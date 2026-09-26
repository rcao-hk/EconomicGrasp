"""Rep-P1 action-conditioned evidence readers and evaluation contracts.

Rep-P1 is a representation diagnostic.  It reuses the *same* K=7 physical
actions and exact CAD/DexNet labels mined by Rep-P0, and changes only the
evidence representation used to predict action quality.

Formal variants:
  action_only: complete physical action only; controls action/dataset priors.\n  geo_pred   : Rep-P0 predicted-geometry descriptor + MLP.\n  img_point  : 13 action-aligned point samples from the frozen pre-enhancer
               image feature map.
  img_region : structured finger/closing/palm/approach region samples from the
               same frozen image feature map.

Image variants never consume sensor depth, rendered depth, CAD geometry,
corruption identity, or the Rep-P0 predicted-depth descriptor.  Geometry enters
only through the explicit physical grasp action that is being scored.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import rankdata

from rep_p0_geometry_common import (
    FRICTION_THRESHOLDS,
    GeometrySourceProbe,
    friction_to_cdf_targets,
    friction_utility,
    predicted_utility_from_logits,
    success08,
)

REP_P1_VERSION = "rep_p1_action_conditioned_evidence_v1"
REP_P1_VARIANTS = ("action_only", "geo_pred", "img_point", "img_region")
IMAGE_VARIANTS = ("img_point", "img_region")
IMAGE_FEATURE_SOURCE = "stage1_proposal_head_pre_enhancer"
POINT_KEYPOINTS = 13
REGION_NAMES = ("closing", "left_finger", "right_finger", "palm", "approach")
REGION_POINT_COUNTS = (18, 6, 6, 6, 12)
REGION_POINTS = sum(REGION_POINT_COUNTS)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def action_digest(actions: np.ndarray, valid: np.ndarray, offsets: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in (
        np.asarray(actions, dtype=np.float32),
        np.asarray(valid, dtype=np.uint8),
        np.asarray(offsets, dtype=np.float32),
    ):
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def load_p0_frame(path: str | Path, *, need_geo: bool = False) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        required = {
            "actions", "valid", "friction", "utility", "offsets_mm",
            "zero_index", "scene_id", "anno_id", "query_ids", "native_score",
        }
        missing = sorted(required - set(d.files))
        if missing:
            raise RuntimeError(f"{path} missing Rep-P0 keys: {missing}")
        out = {
            "actions": d["actions"].astype(np.float32),
            "valid": d["valid"].astype(bool),
            "friction": d["friction"].astype(np.float32),
            "utility": d["utility"].astype(np.float32),
            "offsets": d["offsets_mm"].astype(np.float32),
            "zero": int(np.asarray(d["zero_index"]).reshape(-1)[0]),
            "scene_id": int(np.asarray(d["scene_id"]).reshape(-1)[0]),
            "anno_id": int(np.asarray(d["anno_id"]).reshape(-1)[0]),
            "query_ids": d["query_ids"].astype(np.int64),
            "native_score": d["native_score"].astype(np.float32),
        }
        if need_geo:
            if "feat_pred" not in d.files:
                raise RuntimeError(f"{path} missing feat_pred")
            out["feat_pred"] = d["feat_pred"].astype(np.float32)
    K, Q, D = out["actions"].shape
    if D != 17 or out["valid"].shape != (K, Q):
        raise RuntimeError(f"Malformed actions/valid in {path}")
    if out["friction"].shape != (K, Q) or out["utility"].shape != (K, Q):
        raise RuntimeError(f"Malformed exact labels in {path}")
    if not (0 <= out["zero"] < K):
        raise RuntimeError(f"Bad zero index in {path}")
    if len(out["native_score"]) != Q or len(out["query_ids"]) != Q:
        raise RuntimeError(f"Query alignment mismatch in {path}")
    return out


def image_cache_path(image_root: str | Path, p0_path: str | Path, p0_root: str | Path) -> Path:
    return Path(image_root) / Path(p0_path).relative_to(Path(p0_root))


def load_image_frame(path: str | Path, p0: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        required = {
            "pre_feature", "K", "image_hw", "scene_id", "anno_id",
            "action_digest", "feature_source",
        }
        missing = sorted(required - set(d.files))
        if missing:
            raise RuntimeError(f"{path} missing Rep-P1 image keys: {missing}")
        scene = int(np.asarray(d["scene_id"]).reshape(-1)[0])
        anno = int(np.asarray(d["anno_id"]).reshape(-1)[0])
        if (scene, anno) != (p0["scene_id"], p0["anno_id"]):
            raise RuntimeError(
                f"Image/P0 frame mismatch {scene}/{anno} vs "
                f"{p0['scene_id']}/{p0['anno_id']}"
            )
        cached_digest = str(np.asarray(d["action_digest"]).reshape(-1)[0])
        expected_digest = action_digest(p0["actions"], p0["valid"], p0["offsets"])
        if cached_digest != expected_digest:
            raise RuntimeError(f"Image/P0 action contract mismatch: {path}")
        feature_source = str(np.asarray(d["feature_source"]).reshape(-1)[0])
        if feature_source != IMAGE_FEATURE_SOURCE:
            raise RuntimeError(
                f"Unexpected image feature source {feature_source!r} in {path}"
            )
        feature = d["pre_feature"].astype(np.float32)
        Kcam = d["K"].astype(np.float32)
        image_hw = d["image_hw"].astype(np.int64)
    if feature.ndim != 3 or Kcam.shape != (3, 3) or image_hw.shape != (2,):
        raise RuntimeError(f"Malformed Rep-P1 image cache: {path}")
    if not np.isfinite(feature).all() or not np.isfinite(Kcam).all():
        raise RuntimeError(f"Non-finite Rep-P1 image cache: {path}")
    return {
        "pre_feature": feature,
        "K": Kcam,
        "image_hw": image_hw,
    }


def cache_paths(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("scene_*/ann_*.npz"))


def action_vector(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 3 or actions.shape[-1] != 17:
        raise ValueError("actions must be [K,Q,17]")
    return actions[..., 1:16].float()


def _transform_local(actions: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Transform [K,Q,P,3] gripper-local points into camera coordinates."""
    rot = actions[..., 4:13].float().reshape(*actions.shape[:-1], 3, 3)
    trans = actions[..., 13:16].float()
    return trans.unsqueeze(-2) + torch.matmul(local, rot.transpose(-1, -2))


def action_point_keypoints_camera(
    actions: torch.Tensor,
    finger_width: float = 0.01,
    finger_length: float = 0.06,
    approach_dist: float = 0.03,
) -> torch.Tensor:
    """Return the 13 sparse keypoints used by the failed AIR diagnostic."""
    if actions.ndim != 3 or actions.shape[-1] != 17:
        raise ValueError("actions must be [K,Q,17]")
    a = actions.float()
    width, height, depth = a[..., 1], a[..., 2], a[..., 3]
    z = torch.zeros_like(width)
    x_mid = depth - finger_length / 2.0
    x_tip = depth
    x_root = depth - finger_length
    x_approach = depth - finger_length - finger_width - approach_dist / 2.0
    y_inner = width / 2.0
    y_outer = width / 2.0 + finger_width / 2.0
    z_half = height / 2.0

    def p(x, y, zz):
        if not torch.is_tensor(x):
            x = z + float(x)
        if not torch.is_tensor(y):
            y = z + float(y)
        if not torch.is_tensor(zz):
            zz = z + float(zz)
        return torch.stack((x, y, zz), -1)

    local = torch.stack((
        p(z, z, z),
        p(x_mid, z, z),
        p(x_approach, z, z),
        p(x_mid, -y_inner, z),
        p(x_mid, y_inner, z),
        p(x_tip, -y_outer, z),
        p(x_tip, y_outer, z),
        p(x_root, -y_outer, z),
        p(x_root, y_outer, z),
        p(x_mid, z, -z_half),
        p(x_mid, z, z_half),
        p(x_approach, z, -z_half),
        p(x_approach, z, z_half),
    ), -2)
    return _transform_local(actions, local)


def _append_region_points(
    points: list[torch.Tensor],
    slots: list[Tuple[float, float, float]],
    region_ids: list[int],
    region: int,
    xs: Iterable[Tuple[torch.Tensor, float]],
    ys: Iterable[Tuple[torch.Tensor, float]],
    zs: Iterable[Tuple[torch.Tensor, float]],
):
    for x, sx in xs:
        for y, sy in ys:
            for z, sz in zs:
                points.append(torch.stack((x, y, z), -1))
                slots.append((float(sx), float(sy), float(sz)))
                region_ids.append(int(region))


def action_region_points_camera(
    actions: torch.Tensor,
    finger_width: float = 0.01,
    finger_length: float = 0.06,
    approach_dist: float = 0.03,
):
    """Return 48 structured points covering five physical gripper regions.

    Counts are [18 closing, 6 left finger, 6 right finger, 6 palm, 12 approach].
    The accompanying slot coordinates are normalized region-local positions,
    used only as positional encoding by the image reader.
    """
    if actions.ndim != 3 or actions.shape[-1] != 17:
        raise ValueError("actions must be [K,Q,17]")
    a = actions.float()
    width, height, depth = a[..., 1], a[..., 2], a[..., 3]
    if bool((width <= 0).any()) or bool((height <= 0).any()):
        raise ValueError("Invalid gripper width/height")

    z0 = torch.zeros_like(width)
    half_h = height / 2.0
    half_w = width / 2.0
    outer_w = half_w + finger_width
    x_back = depth - finger_length
    x_front = depth
    x_palm = x_back - finger_width / 2.0
    x_app_back = x_back - finger_width - approach_dist
    x_app_front = x_back - finger_width

    def interp(lo, hi, frac):
        return lo + (hi - lo) * float(frac)

    points: list[torch.Tensor] = []
    slots: list[Tuple[float, float, float]] = []
    region_ids: list[int] = []

    # Closing volume: 3 x 3 x 2 = 18.
    _append_region_points(
        points, slots, region_ids, 0,
        [(interp(x_back, x_front, f), 2*f-1) for f in (0.15, 0.5, 0.85)],
        [(half_w * f, f) for f in (-0.75, 0.0, 0.75)],
        [(half_h * f, f) for f in (-0.6, 0.6)],
    )
    # Finger bodies: 3 x 1 x 2 each.
    _append_region_points(
        points, slots, region_ids, 1,
        [(interp(x_back, x_front, f), 2*f-1) for f in (0.15, 0.5, 0.85)],
        [(-(half_w + finger_width / 2.0), 0.0)],
        [(half_h * f, f) for f in (-0.6, 0.6)],
    )
    _append_region_points(
        points, slots, region_ids, 2,
        [(interp(x_back, x_front, f), 2*f-1) for f in (0.15, 0.5, 0.85)],
        [((half_w + finger_width / 2.0), 0.0)],
        [(half_h * f, f) for f in (-0.6, 0.6)],
    )
    # Palm/bottom body: 1 x 3 x 2 = 6.
    _append_region_points(
        points, slots, region_ids, 3,
        [(x_palm, 0.0)],
        [(outer_w * f, f) for f in (-0.75, 0.0, 0.75)],
        [(half_h * f, f) for f in (-0.6, 0.6)],
    )
    # Approach corridor: 3 x 2 x 2 = 12.
    _append_region_points(
        points, slots, region_ids, 4,
        [(interp(x_app_back, x_app_front, f), 2*f-1) for f in (0.2, 0.5, 0.8)],
        [(outer_w * f, f) for f in (-0.6, 0.6)],
        [(half_h * f, f) for f in (-0.6, 0.6)],
    )

    local = torch.stack(points, -2)
    if local.shape[-2] != REGION_POINTS:
        raise RuntimeError(f"Region point contract changed: {local.shape}")
    slot = torch.tensor(slots, dtype=local.dtype, device=local.device)
    rid = torch.tensor(region_ids, dtype=torch.long, device=local.device)
    return _transform_local(actions, local), slot, rid


def project_points(points: torch.Tensor, K: torch.Tensor, image_hw):
    if points.shape[-1] != 3:
        raise ValueError("points must end in xyz")
    if K.ndim == 3:
        if K.shape[0] != 1:
            raise ValueError("Rep-P1 uses one frame per call")
        K = K[0]
    if K.shape != (3, 3):
        raise ValueError("K must be [3,3] or [1,3,3]")
    h, w = int(image_hw[0]), int(image_hw[1])
    xyz = points.float()
    depth = xyz[..., 2]
    safe = depth.clamp_min(1e-6)
    u = K[0, 0].float() * xyz[..., 0] / safe + K[0, 2].float()
    v = K[1, 1].float() * xyz[..., 1] / safe + K[1, 2].float()
    uv = torch.stack((u, v), -1)
    visible = (
        (depth > 1e-6) & (u >= 0) & (u <= w - 1) &
        (v >= 0) & (v <= h - 1)
    )
    return uv, visible


def sample_image_features(
    feature: torch.Tensor,
    uv: torch.Tensor,
    visible: torch.Tensor,
    image_hw,
):
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("feature must be [1,C,Hf,Wf]")
    h, w = int(image_hw[0]), int(image_hw[1])
    grid = uv.float().clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    flat = grid.reshape(1, -1, 1, 2).to(feature)
    sampled = F.grid_sample(
        feature, flat, mode="bilinear", padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled[0, :, :, 0].T.reshape(*uv.shape[:-1], feature.shape[1])
    return sampled * visible.to(sampled.dtype).unsqueeze(-1)


class _ActionEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(
                [0.10, 0.02, 0.04] + [1.0] * 9 + [1.0, 1.0, 1.0],
                dtype=torch.float32,
            ),
        )
        self.net = nn.Sequential(
            nn.Linear(15, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, actions):
        return self.net(action_vector(actions) / self.scale.to(actions))


class ActionOnlyProbe(nn.Module):
    """Control probe: predict exact-action CDF from action parameters alone.

    All evidence readers receive the explicit action, so this control quantifies
    how much performance can be explained by action/depth priors without any
    image or point-cloud observation.
    """

    def __init__(self, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.action_encoder = _ActionEncoder(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(FRICTION_THRESHOLDS)),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.head(self.action_encoder(actions))


class ActionPointImageProbe(nn.Module):
    """Standalone six-threshold scorer from 13 action-aligned image samples."""

    def __init__(self, feature_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.action_encoder = _ActionEncoder(hidden)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
        )
        self.point_embed = nn.Parameter(torch.empty(POINT_KEYPOINTS, hidden))
        nn.init.normal_(self.point_embed, std=0.02)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden + 1),
            nn.Linear(2 * hidden + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(FRICTION_THRESHOLDS)),
        )

    def forward(self, feature, K, image_hw, actions, valid):
        pts = action_point_keypoints_camera(actions)
        uv, visible = project_points(pts, K, image_hw)
        visible = visible & valid[..., None]
        sampled = sample_image_features(feature, uv, visible, image_hw)
        if sampled.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f"Feature dim mismatch {sampled.shape[-1]} != {self.feature_dim}"
            )
        ah = self.action_encoder(actions)
        token = self.feature_proj(sampled)
        token = token + self.point_embed.to(token)[None, None] + ah.unsqueeze(-2)
        score = self.attn(torch.tanh(token)).squeeze(-1).masked_fill(~visible, -1e4)
        weight = torch.softmax(score, -1) * visible.to(score.dtype)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (weight.unsqueeze(-1) * token).sum(-2)
        vr = visible.float().mean(-1, keepdim=True)
        logits = self.head(torch.cat((pooled, ah, vr), -1))
        return logits, {
            "visible_ratio": vr.squeeze(-1),
            "attention_max": weight.max(-1).values,
        }


class ActionRegionImageProbe(nn.Module):
    """Structured region reader over finger/closing/palm/approach image evidence."""

    def __init__(self, feature_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.action_encoder = _ActionEncoder(hidden)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
        )
        self.position_proj = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.region_embed = nn.Parameter(torch.empty(len(REGION_NAMES), hidden))
        nn.init.normal_(self.region_embed, std=0.02)
        self.point_attn = nn.Linear(hidden, 1)
        self.region_attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden + len(REGION_NAMES) + 1),
            nn.Linear(2 * hidden + len(REGION_NAMES) + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(FRICTION_THRESHOLDS)),
        )

    def forward(self, feature, K, image_hw, actions, valid):
        pts, slot, rid = action_region_points_camera(actions)
        uv, visible = project_points(pts, K, image_hw)
        visible = visible & valid[..., None]
        sampled = sample_image_features(feature, uv, visible, image_hw)
        if sampled.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f"Feature dim mismatch {sampled.shape[-1]} != {self.feature_dim}"
            )
        ah = self.action_encoder(actions)
        token = self.feature_proj(sampled)
        token = token + self.position_proj(slot.to(token))[None, None]
        token = token + self.region_embed.to(token)[rid][None, None]
        token = token + ah.unsqueeze(-2)

        region_tokens = []
        region_visible = []
        for region in range(len(REGION_NAMES)):
            mask_r = (rid == region)
            tr = token[..., mask_r, :]
            vr = visible[..., mask_r]
            score = self.point_attn(torch.tanh(tr)).squeeze(-1)
            score = score.masked_fill(~vr, -1e4)
            weight = torch.softmax(score, -1) * vr.to(score.dtype)
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
            region_tokens.append((weight.unsqueeze(-1) * tr).sum(-2))
            region_visible.append(vr.any(-1))

        regions = torch.stack(region_tokens, -2)
        rvis = torch.stack(region_visible, -1)
        rscore = self.region_attn(torch.tanh(regions + ah.unsqueeze(-2))).squeeze(-1)
        rscore = rscore.masked_fill(~rvis, -1e4)
        rweight = torch.softmax(rscore, -1) * rvis.to(rscore.dtype)
        rweight = rweight / rweight.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (rweight.unsqueeze(-1) * regions).sum(-2)
        point_vr = visible.float().mean(-1, keepdim=True)
        region_vr = rvis.float()
        logits = self.head(torch.cat((pooled, ah, region_vr, point_vr), -1))
        return logits, {
            "visible_ratio": point_vr.squeeze(-1),
            "region_visible_ratio": region_vr.mean(-1),
            "region_attention_max": rweight.max(-1).values,
        }


def build_probe(
    variant: str,
    *,
    feature_dim: int,
    hidden: int = 256,
    dropout: float = 0.1,
):
    if variant == "action_only":\n        return ActionOnlyProbe(hidden, dropout)\n    if variant == "geo_pred":\n        return GeometrySourceProbe(feature_dim, hidden, dropout)\n    if variant == "img_point":
        return ActionPointImageProbe(feature_dim, hidden, dropout)
    if variant == "img_region":
        return ActionRegionImageProbe(feature_dim, hidden, dropout)
    raise ValueError(f"Unknown Rep-P1 variant {variant!r}")


def pairwise_rank_loss(
    logits: torch.Tensor,
    exact_utility: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Weighted within-ray logistic ranking loss over non-tied exact utilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    pred = torch.sigmoid(logits.float()).mean(-1)
    exact = exact_utility.float()
    if pred.shape != exact.shape or valid.shape != exact.shape:
        raise ValueError("pairwise loss shape mismatch")
    K = pred.shape[0]
    upper = torch.triu(
        torch.ones(K, K, device=pred.device, dtype=torch.bool), diagonal=1
    )[:, :, None]
    pair_valid = valid[:, None, :] & valid[None, :, :] & upper
    true_diff = exact[:, None, :] - exact[None, :, :]
    pair_valid = pair_valid & (true_diff.abs() > 1e-8)
    if not bool(pair_valid.any()):
        return pred.sum() * 0.0
    pred_diff = pred[:, None, :] - pred[None, :, :]
    sign = torch.sign(true_diff[pair_valid])
    weight = true_diff.abs()[pair_valid]
    term = F.softplus(-sign * pred_diff[pair_valid] / float(temperature))
    return (term * weight).sum() / weight.sum().clamp_min(1e-8)


def gather_kn(arr: np.ndarray, k: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    k = np.asarray(k, np.int64)
    return arr[k, np.arange(arr.shape[1])]


def pearson(x, y) -> float:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 2:
        return float("nan")
    return pearson(rankdata(x, method="average"), rankdata(y, method="average"))


def auroc(y_true, score) -> float:
    y = np.asarray(y_true, bool)
    s = np.asarray(score, np.float64)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(y_true, score) -> float:
    y = np.asarray(y_true, bool)
    s = np.asarray(score, np.float64)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    yy = y[order].astype(np.float64)
    precision = np.cumsum(yy) / (np.arange(len(yy), dtype=np.float64) + 1.0)
    return float((precision * yy).sum() / pos)


def within_ray_pairwise_accuracy(
    pred_u: np.ndarray,
    exact_u: np.ndarray,
    valid: np.ndarray,
) -> tuple[int, int]:
    pred = np.asarray(pred_u, np.float64)
    exact = np.asarray(exact_u, np.float64)
    valid = np.asarray(valid, bool)
    K, Q = pred.shape
    correct = total = 0
    for q in range(Q):
        ids = np.flatnonzero(valid[:, q])
        for ii in range(len(ids)):
            for jj in range(ii + 1, len(ids)):
                i, j = int(ids[ii]), int(ids[jj])
                td = exact[i, q] - exact[j, q]
                if abs(td) <= 1e-8:
                    continue
                pd = pred[i, q] - pred[j, q]
                correct += int(pd * td > 0)
                total += 1
    return correct, total


def select_policy(
    pred_u: np.ndarray,
    exact_u: np.ndarray,
    valid: np.ndarray,
    zero: int,
    margin: float,
):
    K, Q = pred_u.shape
    alt = valid.copy()
    alt[zero] = False
    score = np.where(alt, pred_u, -np.inf)
    best = np.argmax(score, axis=0).astype(np.int64)
    has_alt = alt.any(axis=0)
    best = np.where(has_alt, best, zero)
    advantage = pred_u[best, np.arange(Q)] - pred_u[zero]
    selected = np.where(
        has_alt & (advantage > float(margin)), best, zero
    ).astype(np.int64)

    masked = np.where(valid, exact_u, -np.inf)
    exact_best = masked.max(axis=0)
    oracle_arg = np.argmax(masked, axis=0).astype(np.int64)
    oracle = np.where(
        exact_best > exact_u[zero] + 1e-8, oracle_arg, zero
    ).astype(np.int64)
    return selected, oracle, best, advantage


def _policy_arrays(frame, margin: float, subset: np.ndarray | None = None):
    selected, oracle, best, advantage = select_policy(
        frame["pred_u"], frame["utility"], frame["valid"],
        frame["zero"], margin,
    )
    q = np.arange(frame["pred_u"].shape[1])
    if subset is not None:
        q = np.asarray(subset, np.int64)
    u_sel = frame["utility"][selected[q], q]
    u_nat = frame["utility"][frame["zero"], q]
    u_orc = frame["utility"][oracle[q], q]
    f_sel = frame["friction"][selected[q], q]
    f_nat = frame["friction"][frame["zero"], q]
    s_sel = success08(f_sel)
    s_nat = success08(f_nat)
    return {
        "u_sel": u_sel, "u_nat": u_nat, "u_orc": u_orc,
        "s_sel": s_sel, "s_nat": s_nat,
        "rescue": (~s_nat) & s_sel,
        "harm": s_nat & (~s_sel),
        "change": selected[q] != frame["zero"],
        "selected": selected[q],
        "oracle": oracle[q],
        "best": best[q],
        "advantage": advantage[q],
        "q": q,
    }


def policy_metrics(
    frames: list[Dict[str, np.ndarray]],
    margin: float,
    *,
    top_native_k: int | None = None,
) -> Dict[str, float]:
    parts = {
        k: [] for k in
        ("u_sel", "u_nat", "u_orc", "s_sel", "s_nat", "rescue", "harm", "change")
    }
    for frame in frames:
        subset = None
        if top_native_k is not None:
            subset = np.argsort(
                -np.asarray(frame["native_score"]), kind="stable"
            )[: min(int(top_native_k), len(frame["native_score"]))]
        p = _policy_arrays(frame, margin, subset)
        for key in parts:
            parts[key].append(np.asarray(p[key]))
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    gain = float((cat["u_sel"] - cat["u_nat"]).mean())
    head = float((cat["u_orc"] - cat["u_nat"]).mean())
    return {
        "margin": float(margin),
        "num_queries": int(len(cat["u_sel"])),
        "selected_utility": float(cat["u_sel"].mean()),
        "native_utility": float(cat["u_nat"].mean()),
        "oracle_utility": float(cat["u_orc"].mean()),
        "utility_gain": gain,
        "utility_headroom_recovery": (
            gain / head if abs(head) > 1e-12 else float("nan")
        ),
        "selected_success08": float(cat["s_sel"].mean()),
        "native_success08": float(cat["s_nat"].mean()),
        "success08_gain": float(
            (cat["s_sel"].astype(np.float32) -
             cat["s_nat"].astype(np.float32)).mean()
        ),
        "rescue08": float(cat["rescue"].mean()),
        "harm08": float(cat["harm"].mean()),
        "change_rate": float(cat["change"].mean()),
    }


def tune_margin(
    frames: list[Dict[str, np.ndarray]],
    margin_max: float,
    margin_steps: int,
):
    margins = np.linspace(0.0, float(margin_max), max(2, int(margin_steps)))
    rows = [policy_metrics(frames, float(m)) for m in margins]
    best = max(
        rows,
        key=lambda r: (
            r["selected_utility"], -r["harm08"], -r["change_rate"]
        ),
    )
    return best, rows


def candidate_metrics(frames: list[Dict[str, np.ndarray]]) -> Dict[str, float]:
    pred_all, exact_all, succ_all = [], [], []
    pair_correct = pair_total = 0
    for frame in frames:
        mask = frame["valid"]
        pred_all.append(frame["pred_u"][mask])
        exact_all.append(frame["utility"][mask])
        succ_all.append(success08(frame["friction"][mask]))
        c, n = within_ray_pairwise_accuracy(
            frame["pred_u"], frame["utility"], frame["valid"]
        )
        pair_correct += c
        pair_total += n
    pred = np.concatenate(pred_all)
    exact = np.concatenate(exact_all)
    succ = np.concatenate(succ_all)
    return {
        "pred_exact_utility_pearson": pearson(pred, exact),
        "pred_exact_utility_spearman": spearman(pred, exact),
        "success08_auroc": auroc(succ, pred),
        "success08_auprc": average_precision(succ, pred),
        "success08_positive_fraction": float(succ.mean()),
        "within_ray_pairwise_accuracy": (
            float(pair_correct / pair_total) if pair_total else float("nan")
        ),
        "within_ray_pairs": int(pair_total),
    }


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def load_stage1_reference(
    checkpoint_path: str,
    device: torch.device,
    *,
    min_depth: float = 0.2,
    max_depth: float = 1.0,
    bin_num: int = 256,
):
    """Load the exact RGB-only Stage-1 model used to mine Rep-P0 actions."""
    from utils.arguments import cfgs
    from models.economicgrasp_dpt_distill import economicgrasp_dpt_student

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError("Rep-P1 requires a full Stage-1 checkpoint")
    source = str(ckpt.get("geometry_depth_source", "pred"))
    if source not in ("", "pred"):
        raise RuntimeError(
            f"Rep-P1 requires RGB-predicted geometry Stage-1, got {source!r}"
        )
    meta = {
        "pose_mode": str(ckpt.get("pose_depth_mode", "global_film")),
        "use_fuse_depth": bool(ckpt.get("use_fuse_depth", False)),
        "camera_pose_key": str(ckpt.get("camera_pose_key", "camera_pose_vec")),
        "camera_gravity_key": str(
            ckpt.get("camera_gravity_key", "camera_gravity_vec")
        ),
        "pose_hidden_dim": int(ckpt.get("pose_hidden_dim", 64)),
        "ray_gravity_hidden_dim": int(
            ckpt.get("ray_gravity_hidden_dim", 64)
        ),
        "ray_gravity_mid_dim": int(
            ckpt.get("ray_gravity_mid_dim", 32)
        ),
    }
    cfgs.use_top4_view_infer = False
    cfgs.kview_mode = "A1"
    cfgs.kview_k = 1
    cfgs.use_cdf = True
    cfgs.use_obs_depth = False
    cfgs.pose_depth_mode = meta["pose_mode"]
    model = economicgrasp_dpt_student(
        min_depth=min_depth,
        max_depth=max_depth,
        bin_num=bin_num,
        is_training=False,
        use_obs_depth=False,
        pose_depth_mode=meta["pose_mode"],
        camera_pose_key=meta["camera_pose_key"],
        camera_gravity_key=meta["camera_gravity_key"],
        pose_hidden_dim=meta["pose_hidden_dim"],
        ray_gravity_hidden_dim=meta["ray_gravity_hidden_dim"],
        ray_gravity_mid_dim=meta["ray_gravity_mid_dim"],
        use_cdf=True,
        vis_dir=None,
    ).to(device)
    state = ckpt["model_state_dict"]
    result = model.load_state_dict(state, strict=False)
    optional = ("rgb_geometry_diagnostics.",)
    missing = [k for k in result.missing_keys if not k.startswith(optional)]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(optional)]
    if missing or unexpected:
        raise RuntimeError(
            f"Stage-1 checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    del state, ckpt
    model.eval().requires_grad_(False)
    return model, meta


@torch.no_grad()
def extract_pre_enhancer_feature(reference, batch: Dict[str, torch.Tensor]):
    """Extract the frozen proposal-head feature *before* spatial depth enhancement."""
    pack = reference.depth_net(
        batch["img"],
        camera_pose_vec=batch[reference.camera_pose_key],
        camera_gravity_vec=None,
        camera_K=batch["K"],
        return_feats=True,
        return_raw=True,
        return_pose_aux=True,
    )
    if not isinstance(pack, (tuple, list)) or len(pack) != 6:
        raise RuntimeError("Stage-1 depth_net contract changed")
    h, w = batch["img"].shape[-2:]
    raw, _ = reference.proposal_head(pack[4], h // 14, w // 14)
    if raw.ndim != 4 or raw.shape[0] != 1:
        raise RuntimeError(f"Unexpected pre-enhancer feature shape {raw.shape}")
    return raw.detach()
