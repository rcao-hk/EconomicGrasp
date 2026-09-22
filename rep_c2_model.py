"""Rep-C2 shared model and evidence extraction.

Rep-C2 does not change A1's proposed correction. It asks whether a lightweight,
equally-sized verifier can tell when that correction is beneficial from:

  score    : A1 score/profile + offset only.
  local    : score + paired gripper-local geometric evidence.
  reliable : local + visibility/depth-discontinuity reliability evidence.

All three verifiers receive a 160-D tensor and have the same architecture and
parameter count. Unavailable evidence blocks are zeroed, so gains cannot be
explained by a larger classifier. The local evidence is derived only from the
predicted depth consumed by A1; no CAD, sensor depth or GT geometry is used.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


SCORE_DIM = 25
LOCAL_DIM = 75
RELIABILITY_DIM = 60
FULL_DIM = SCORE_DIM + LOCAL_DIM + RELIABILITY_DIM
EVIDENCE = ("score", "local", "reliable")


class RepC2Verifier(nn.Module):
    def __init__(self, hidden=128, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FULL_DIM, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        if x.shape[-1] != FULL_DIM:
            raise ValueError(f"Expected {FULL_DIM} features, got {x.shape}")
        return self.net(x).squeeze(-1)


def evidence_mask(kind, *, device=None, dtype=torch.float32):
    if kind not in EVIDENCE:
        raise ValueError(kind)
    mask = torch.zeros(FULL_DIM, device=device, dtype=dtype)
    mask[:SCORE_DIM] = 1
    if kind in ("local", "reliable"):
        mask[SCORE_DIM:SCORE_DIM+LOCAL_DIM] = 1
    if kind == "reliable":
        mask[SCORE_DIM+LOCAL_DIM:] = 1
    return mask


def apply_evidence(x, kind):
    return x * evidence_mask(kind, device=x.device, dtype=x.dtype)


def gripper_region_samples(actions):
    """Return local/world points for 5 roles x 8 corners per grasp.

    Role boxes match Rep-A's IndependentImageReader exactly:
    closing volume, left finger, right finger, palm, approach.
    """
    a = actions.reshape(-1, 17)
    w, h, d = a[:, 1], a[:, 2], a[:, 3]
    lo_x, hi_x = d-.06, d
    lo = torch.stack((
        torch.stack((lo_x, -w/2, -h/2), -1),
        torch.stack((lo_x, -w/2-.01, -h/2), -1),
        torch.stack((lo_x,  w/2,     -h/2), -1),
        torch.stack((lo_x-.01, -w/2-.01, -h/2), -1),
        torch.stack((lo_x-.04, -w/2-.01, -h/2), -1),
    ), 1)
    hi = torch.stack((
        torch.stack((hi_x,  w/2,      h/2), -1),
        torch.stack((hi_x, -w/2,      h/2), -1),
        torch.stack((hi_x,  w/2+.01,  h/2), -1),
        torch.stack((lo_x,  w/2+.01,  h/2), -1),
        torch.stack((lo_x-.01, w/2+.01, h/2), -1),
    ), 1)
    corners = torch.cartesian_prod(
        torch.tensor([.15,.85], device=a.device, dtype=a.dtype),
        torch.tensor([.15,.85], device=a.device, dtype=a.dtype),
        torch.tensor([.15,.85], device=a.device, dtype=a.dtype),
    )
    local = lo[:, :, None] + (hi-lo)[:, :, None] * corners[None, None]
    local = local.reshape(-1, 40, 3)
    rot = a[:, 4:13].reshape(-1, 3, 3)
    xyz = torch.einsum("nmc,njc->nmj", local, rot) + a[:, None, 13:16]
    return local.reshape(-1,5,8,3), xyz.reshape(-1,5,8,3)


def _sample_scalar_map(map_hw, xyz, K):
    """Sample [H,W] map at projected xyz [Q,5,8,3]."""
    q = xyz.shape[0]
    flat = xyz.reshape(q, 40, 3)
    proj = flat @ K.T
    uv = proj[..., :2] / proj[..., 2:].clamp_min(1e-6)
    h, w = map_hw.shape[-2:]
    valid = (flat[...,2] > .01) & torch.isfinite(uv).all(-1)
    valid &= (uv[...,0] >= 0) & (uv[...,0] <= w-1)
    valid &= (uv[...,1] >= 0) & (uv[...,1] <= h-1)
    grid = 2*(uv+.5)/uv.new_tensor([w,h])-1
    grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
    sampled = F.grid_sample(
        map_hw[None,None],
        grid.reshape(1,q*40,1,2),
        padding_mode="zeros",
        align_corners=False,
    )[0,0,:,0].reshape(q,5,8)
    return sampled, valid.reshape(q,5,8)


def _masked_mean(value, mask, dim=-1):
    weight = mask.to(value.dtype)
    return (value*weight).sum(dim) / weight.sum(dim).clamp_min(1.)


def action_depth_evidence(depth, actions, K):
    """Return task-local geometry [Q,25] and reliability [Q,20]."""
    if depth.ndim == 3:
        d = depth[0]
    elif depth.ndim == 2:
        d = depth
    else:
        raise ValueError(f"depth must be [1,H,W] or [H,W], got {depth.shape}")
    if K.shape != (3,3):
        raise ValueError(f"K must be [3,3], got {K.shape}")
    _, xyz = gripper_region_samples(actions)
    obs, proj_valid = _sample_scalar_map(d, xyz, K)
    z = xyz[...,2]
    valid = proj_valid & torch.isfinite(obs) & (obs > .01)
    gap = torch.where(valid, obs-z, torch.zeros_like(obs))

    # Simple deployment-available reliability proxies: projection validity,
    # local depth gradient, local roughness and strong discontinuity fraction.
    gx = torch.zeros_like(d)
    gy = torch.zeros_like(d)
    gx[:,1:] = (d[:,1:]-d[:,:-1]).abs()
    gy[1:,:] = (d[1:,:]-d[:-1,:]).abs()
    grad = .5*(gx+gy)
    avg = F.avg_pool2d(d[None,None],3,stride=1,padding=1)[0,0]
    rough = (d-avg).abs()
    grad_s, _ = _sample_scalar_map(grad, xyz, K)
    rough_s, _ = _sample_scalar_map(rough, xyz, K)

    local_parts, rel_parts = [], []
    for role in range(5):
        m = valid[:,role]
        g = gap[:,role]
        local_parts += [
            (_masked_mean(g,m)/.05).clamp(-4,4),
            (_masked_mean(g.abs(),m)/.05).clamp(0,4),
            _masked_mean((g < -.005).float(),m),
            _masked_mean((g >  .005).float(),m),
            _masked_mean((g.abs() < .010).float(),m),
        ]
        gs = grad_s[:,role]
        rs = rough_s[:,role]
        rel_parts += [
            m.float().mean(-1),
            (_masked_mean(gs,m)/.05).clamp(0,4),
            (_masked_mean(rs,m)/.05).clamp(0,4),
            _masked_mean((gs > .02).float(),m),
        ]
    local = torch.stack(local_parts,-1)
    reliability = torch.stack(rel_parts,-1)
    if local.shape[-1] != 25 or reliability.shape[-1] != 20:
        raise RuntimeError("Rep-C2 evidence dimension bug")
    return local, reliability


def build_pair_evidence(data, depth, probabilities, proposal):
    """Build full 160-D evidence for native vs A1-proposed candidate."""
    prob = probabilities.float()
    valid = data["valid"].bool()
    k,q,c = prob.shape
    if c != 6 or valid.shape != (k,q):
        raise ValueError("Expected probabilities [K,Q,6] and matching valid")
    z = int(data["zero_index"])
    qq = torch.arange(q, device=prob.device)
    proposal = proposal.long()
    if proposal.shape != (q,) or not bool(valid[proposal,qq].all()):
        raise ValueError("Invalid proposal")

    p0 = prob[z]
    pc = prob[proposal,qq]
    u0, uc = p0.mean(-1), pc.mean(-1)
    offsets = data["offsets_mm"].float()[proposal] / 40.
    native_score = data.get("native_score", None)
    if native_score is None:
        native_score = u0.detach()
    else:
        native_score = native_score.float()
    score = torch.cat([
        p0, pc, pc-p0,
        u0[:,None], uc[:,None], (uc-u0)[:,None],
        offsets[:,None], offsets.abs()[:,None], offsets.square()[:,None],
        native_score[:,None],
    ],-1)
    if score.shape[-1] != SCORE_DIM:
        raise RuntimeError(f"Score evidence dim {score.shape[-1]} != {SCORE_DIM}")

    actions = data["actions"].float()
    a0 = actions[z]
    ac = actions[proposal,qq]
    l0,r0 = action_depth_evidence(depth,a0,data["K"].float())
    lc,rc = action_depth_evidence(depth,ac,data["K"].float())
    local = torch.cat([l0,lc,lc-l0],-1)
    rel = torch.cat([r0,rc,rc-r0],-1)
    x = torch.cat([score,local,rel],-1)
    if x.shape[-1] != FULL_DIM:
        raise RuntimeError(f"Evidence dim {x.shape[-1]} != {FULL_DIM}")
    return x
