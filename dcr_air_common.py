"""Pure contracts for DCR-AIR: Action-aligned Image Readout."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from dcr_cva_common import checked_native_score
from e1e2_common import digest, file_sha, select_centers

AIR_VERSION = 'dcr_action_image_readout_v1'
AIR_METHODS = ('native', 'dcr_stage1', 'air_stage1')


def action_keypoints_camera(actions: torch.Tensor, finger_width: float = .01,
                            finger_length: float = .06,
                            approach_dist: float = .03) -> torch.Tensor:
    """Return 13 gripper-region key points in camera coordinates."""
    if actions.ndim != 3 or actions.shape[-1] != 17:
        raise ValueError('Expected actions [C,Q,17]')
    if min(finger_width, finger_length, approach_dist) <= 0:
        raise ValueError('Gripper dimensions must be positive')
    a = actions.float()
    width, height, depth = a[..., 1], a[..., 2], a[..., 3]
    if not bool(torch.isfinite(a).all()) or bool((width <= 0).any()) or bool((height <= 0).any()):
        raise ValueError('Non-finite/invalid grasp action')

    z = torch.zeros_like(width)
    x_mid = depth - finger_length / 2
    x_tip = depth
    x_root = depth - finger_length
    x_approach = depth - finger_length - finger_width - approach_dist / 2
    y_inner = width / 2
    y_outer = width / 2 + finger_width / 2
    z_half = height / 2

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

    rot = a[..., 4:13].reshape(*a.shape[:-1], 3, 3)
    trans = a[..., 13:16]
    # collision_detector.py uses local=(point-translation)@R.
    return trans.unsqueeze(-2) + torch.matmul(local, rot.transpose(-1, -2))


def project_keypoints(points: torch.Tensor, K: torch.Tensor, image_hw):
    """Project [...,P,3] camera-frame points to pixels and visibility."""
    if points.ndim < 3 or points.shape[-1] != 3:
        raise ValueError('Expected [...,P,3] points')
    h, w = map(int, image_hw)
    if min(h, w) < 2:
        raise ValueError('Invalid image size')
    if K.ndim == 3:
        if K.shape[0] != 1:
            raise ValueError('AIR follows the one-frame E1/DCR protocol')
        K = K[0]
    if K.shape != (3, 3):
        raise ValueError('Expected camera intrinsics [3,3] or [1,3,3]')
    xyz = points.float()
    zz = xyz[..., 2]
    safe_z = zz.clamp_min(1e-6)
    u = K[0, 0].float() * xyz[..., 0] / safe_z + K[0, 2].float()
    v = K[1, 1].float() * xyz[..., 1] / safe_z + K[1, 2].float()
    uv = torch.stack((u, v), -1)
    visible = ((zz > 1e-6) & (u >= 0) & (u <= w - 1) &
               (v >= 0) & (v <= h - 1))
    return uv, visible


def sample_image_features(feature: torch.Tensor, uv: torch.Tensor,
                          visible: torch.Tensor, image_hw):
    """Bilinearly sample a B=1 pre-enhancer feature map."""
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError('Expected feature [1,C,Hf,Wf]')
    h, w = map(int, image_hw)
    if uv.shape[:-1] != visible.shape or uv.shape[-1] != 2:
        raise ValueError('UV/visibility shape mismatch')
    grid = uv.float().clone()
    grid[..., 0] = 2 * grid[..., 0] / max(w - 1, 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / max(h - 1, 1) - 1
    flat = grid.reshape(1, -1, 1, 2).to(feature)
    sampled = F.grid_sample(feature, flat, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    sampled = sampled[0, :, :, 0].T.reshape(*uv.shape[:-1], feature.shape[1])
    return sampled * visible.to(sampled.dtype).unsqueeze(-1)


def evidence_cdf_loss(fused_logits: torch.Tensor, target: torch.Tensor,
                      valid: torch.Tensor, residual: torch.Tensor,
                      anchor_weight: float = .01):
    if fused_logits.shape != target.shape or fused_logits.shape[:-1] != valid.shape:
        raise ValueError('Expected [C,Q,6] logits/target and [C,Q] valid')
    if residual.shape != valid.shape:
        raise ValueError('AIR residual/valid mismatch')
    if not math.isfinite(anchor_weight) or anchor_weight < 0:
        raise ValueError('Invalid AIR anchor weight')
    bce = F.binary_cross_entropy_with_logits(fused_logits[valid], target[valid])
    anchor = residual[valid].square().mean()
    return bce + anchor_weight * anchor, {
        'cdf_bce': float(bce.detach()),
        'air_anchor': float(anchor.detach()),
        'air_abs_residual': float(residual[valid].detach().abs().mean()),
    }


def make_air_outputs(base_logits, fused_logits, bundle, zero):
    """Compare DCR and AIR corrections under identical Stage-1 ranking."""
    if base_logits.shape != fused_logits.shape:
        raise ValueError('Base/AIR CDF shape mismatch')
    base_u = base_logits.float().sigmoid().mean(-1)
    air_u = fused_logits.float().sigmoid().mean(-1)
    base_sel = select_centers(base_u, bundle['valid'], zero)
    air_sel = select_centers(air_u, bundle['valid'], zero)
    q = torch.arange(len(base_sel), device=base_sel.device)
    native = bundle['actions'][zero].detach().clone()
    s0 = checked_native_score(native[:, 0])

    outputs = {'native': native}
    for name, sel in (('dcr_stage1', base_sel), ('air_stage1', air_sel)):
        physical = bundle['actions'][sel, q].detach().clone()
        physical[:, 0] = s0
        outputs[name] = physical
    return outputs, base_sel, air_sel


def code_fingerprint():
    root = Path(__file__).resolve().parent
    files = ('dcr_air_common.py', 'models/economicgrasp_cva_air.py',
             'train_dcr_air.py')
    return digest({p: file_sha(root / p) for p in files})
