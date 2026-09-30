"""Online P1 controls on a frozen Metric-Grasp-Field source.

No feature/action cache is created. P1-1 isolates ray-profile / relative-prior
inputs. P1-3 tests correction parameterizations while preserving the same
frozen candidate generator and Base-CVA logits.
"""
from __future__ import annotations
import copy
import hashlib
import math
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mgf_p0_core import freeze, assert_frozen
from metric_grasp_field_core import (
    evidence_at, gripper_support, project_points, sample_map,
    compose_monotone_residual_logits,
)

VERSION = "mgf_online_p1_v1"

P11_VARIANTS = (
    "profile_hard",
    "profile_fixed",
    "profile_learned",
    "relative_metric_only",
    "relative_scalar",
    "relative_feature",
)

P13_VARIANTS = (
    "residual6",
    "base_conditioned",
    "evidence_gated",
    "shared_shift",
)


def code_digest():
    from mgf_p0_online import code_digest as p0_digest
    root = Path(__file__).resolve().parent
    h = hashlib.sha256(p0_digest().encode())
    for path in sorted(root.glob("mgf_p1_*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def prepare_p11_context(source, context, variant):
    """Prepare a P1-1 view of one frozen-source forward.

    Profile variants share exactly the same learned ray distribution/mean and
    differ only in the evidence operator used by the trainable residual reader.

    Relative-input variants recompute the *frozen* ray head after zeroing the
    requested official-DAV2 relative cue(s). The same relative-feature ablation
    is applied to the trainable TaskFeatureAdapter. This is a trained-source
    intervention, not a claim about retraining the ray head from scratch.
    """
    if variant not in P11_VARIANTS:
        raise ValueError(f"Unknown P1-1 variant {variant!r}")

    out = dict(context)
    out["p1_evidence_mode"] = "learned"
    out["p1_relative"] = context["relative"]
    out["p1_prob"] = context["prob"]

    if variant.startswith("profile_"):
        out["p1_evidence_mode"] = variant.split("_", 1)[1]
        return out

    use_scalar = variant in ("relative_scalar", "profile_learned")
    use_feature = variant in ("relative_feature", "profile_learned")
    # relative_* variants are explicit; profile_learned is handled above.
    if variant == "relative_metric_only":
        use_scalar = use_feature = False
    elif variant == "relative_scalar":
        use_scalar, use_feature = True, False
    elif variant == "relative_feature":
        use_scalar, use_feature = False, True

    rel = context["relative"] if use_feature else torch.zeros_like(context["relative"])
    rinv = (
        context["relative_inverse"]
        if use_scalar
        else torch.zeros_like(context["relative_inverse"])
    )
    out["p1_relative"] = rel
    with torch.no_grad():
        assert_frozen(source.model)
        _, prob = source.model.ray_head(
            rel,
            rinv,
            context["metric"],
            context["depth"],
            context["hw"],
            context["K"],
        )
    out["p1_prob"] = prob.detach()
    return out


class SupportScorer(nn.Module):
    """Role-aware support encoder shared by all P1 residual controls."""

    def __init__(self, source, evidence_mode="learned"):
        super().__init__()
        self.cfg = replace(source.config, evidence_mode=str(evidence_mode))
        h = int(self.cfg.hidden)
        src = source.model.readout
        self.role_embed = copy.deepcopy(src.role_embed)
        self.point = copy.deepcopy(src.point)
        self.pre = nn.Sequential(
            copy.deepcopy(src.head[0]),
            copy.deepcopy(src.head[1]),
            copy.deepcopy(src.head[2]),
        )
        self.hidden = h

    def _chunk(self, feature, prob, actions, K, image_hw):
        xyz, local, roles = gripper_support(actions.float())
        uv, valid = project_points(xyz, K.float(), image_hw)
        valid = (
            valid
            & (xyz[..., 2] >= self.cfg.min_depth)
            & (xyz[..., 2] <= self.cfg.max_depth)
        )
        visual = sample_map(feature.float(), uv, image_hw)
        p = sample_map(prob.detach().float(), uv, image_hw)
        evidence = evidence_at(p, xyz[..., 2], self.cfg)
        evidence = torch.cat((evidence, valid[..., None].float()), -1)
        tokens = self.point(torch.cat((visual, evidence, local / 0.1), -1))
        tokens = (tokens + self.role_embed(roles)[None, None]) * valid[..., None]

        pooled = []
        for role in range(5):
            mask = (roles == role)[None, None] & valid
            pooled.append(
                (tokens * mask[..., None]).sum(-2)
                / mask.sum(-1, keepdim=True).clamp_min(1)
            )
        size = actions[..., 1:4] / actions.new_tensor([0.1, 0.02, 0.04])
        hidden = self.pre(torch.cat((*pooled, size), -1))

        # Deterministic evidence-sufficiency gate. A candidate falls back to
        # Base when support points are invalid or the queried ray profiles are
        # diffuse. No learned gate can trivially saturate to one.
        pp = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
        entropy = -(pp.clamp_min(1e-8) * pp.clamp_min(1e-8).log()).sum(-1)
        confidence = 1.0 - entropy / math.log(float(pp.shape[-1]))
        gate = (confidence.clamp(0, 1) * valid.float()).sum(-1) / float(valid.shape[-1])
        return hidden, gate.clamp(0, 1)

    def forward(self, feature, prob, actions, K, image_hw):
        if actions.ndim != 3 or actions.shape[-1] != 17:
            raise ValueError("Expected actions [B,N,17]")
        actions, prob, K = actions.detach(), prob.detach(), K.detach()
        hs, gates = [], []
        for part in actions.split(self.cfg.action_chunk, dim=1):
            if self.training and self.cfg.checkpoint_chunks and torch.is_grad_enabled():
                def run(f, p, a, k):
                    return self._chunk(f, p, a, k, image_hw)
                h, g = checkpoint(run, feature, prob, part, K, use_reentrant=False)
            else:
                h, g = self._chunk(feature, prob, part, K, image_hw)
            hs.append(h)
            gates.append(g)
        return torch.cat(hs, 1), torch.cat(gates, 1)


class P11Control(nn.Module):
    def __init__(self, source, variant):
        super().__init__()
        if variant not in P11_VARIANTS:
            raise ValueError(variant)
        self.family = "p1_1"
        self.variant = variant
        self.adapter = copy.deepcopy(source.model.task_adapter).requires_grad_(True)
        mode = variant.split("_", 1)[1] if variant.startswith("profile_") else "learned"
        self.support = SupportScorer(source, mode).requires_grad_(True)
        self.out = nn.Linear(self.support.hidden, 6)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, context):
        b, q, a, d = context["shape"]
        feature = self.adapter(
            context["proposal"],
            context["p1_relative"],
            context["metric"],
            context["p1_prob"].shape[-2:],
        )
        hidden, _ = self.support(
            feature,
            context["p1_prob"],
            context["actions"],
            context["K"],
            context["hw"],
        )
        raw = self.out(hidden).reshape(b, q, a, d, 6)
        base = context["ep"]["grasp_cdf_pred_angle_depth"].detach().movedim(1, -1)
        return compose_monotone_residual_logits(base, raw).movedim(-1, 1).contiguous()


class P13Control(nn.Module):
    def __init__(self, source, variant):
        super().__init__()
        if variant not in P13_VARIANTS:
            raise ValueError(variant)
        self.family = "p1_3"
        self.variant = variant
        self.adapter = copy.deepcopy(source.model.task_adapter).requires_grad_(True)
        self.support = SupportScorer(source, "learned").requires_grad_(True)
        h = self.support.hidden
        if variant == "base_conditioned":
            self.base_proj = nn.Sequential(
                nn.LayerNorm(7), nn.Linear(7, h), nn.GELU()
            )
        out_dim = 1 if variant == "shared_shift" else 6
        self.out = nn.Linear(h, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, context):
        b, q, a, d = context["shape"]
        feature = self.adapter(
            context["proposal"],
            context["relative"],
            context["metric"],
            context["prob"].shape[-2:],
        )
        hidden, gate = self.support(
            feature,
            context["prob"],
            context["actions"],
            context["K"],
            context["hw"],
        )
        base = context["ep"]["grasp_cdf_pred_angle_depth"].detach()
        base_last = base.movedim(1, -1)
        base_flat = base_last.reshape(b, q * a * d, 6)

        if self.variant == "base_conditioned":
            utility = base_flat.sigmoid().mean(-1, keepdim=True)
            hidden = hidden + self.base_proj(torch.cat((base_flat, utility), -1))

        raw = self.out(hidden)
        if self.variant == "shared_shift":
            final = base_flat + raw
            return final.reshape(b, q, a, d, 6).movedim(-1, 1).contiguous()

        if self.variant == "evidence_gated":
            raw = raw * gate[..., None]

        raw = raw.reshape(b, q, a, d, 6)
        return compose_monotone_residual_logits(base_last, raw).movedim(-1, 1).contiguous()


def make_control(source, family, variant):
    if family == "p1_1":
        return P11Control(source, variant).to(next(source.parameters()).device)
    if family == "p1_3":
        return P13Control(source, variant).to(next(source.parameters()).device)
    raise ValueError(f"Unknown P1 family {family!r}")


def prepare_context(source, context, family, variant):
    if family == "p1_1":
        return prepare_p11_context(source, context, variant)
    if family == "p1_3":
        return context
    raise ValueError(family)


def load_control(source, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("version") != VERSION:
        raise RuntimeError("Not a P1 control checkpoint")
    protocol = ck["protocol"]
    if protocol["source_sha256"] != source.checkpoint_sha:
        raise RuntimeError("P1 checkpoint/source mismatch")
    control = make_control(source, protocol["family"], protocol["variant"])
    control.load_state_dict(ck["control"], strict=True)
    control.eval()
    return control, ck
