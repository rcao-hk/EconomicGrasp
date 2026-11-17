"""Pixel-centre pinhole geometry. Depth is optical-axis z in metres."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def check_K(K):
    if K.shape[-2:] != (3, 3) or not torch.isfinite(K).all():
        raise ValueError('Finite [...,3,3] K required')
    if (K[..., 0, 0] <= 0).any() or (K[..., 1, 1] <= 0).any():
        raise ValueError('Positive focal lengths required')
    if not torch.allclose(K[..., 2, :], K.new_tensor([0., 0., 1.]).expand_as(K[..., 2, :])):
        raise ValueError('Pinhole K required')


def crop_matrix(box, out_hw, *, dtype=torch.float32, device=None):
    """Box [x0,y0,x1,y1) in pixel-edge coordinates; half-pixel resize."""
    x0, y0, x1, y1 = map(float, box)
    h, w = map(int, out_hw)
    if not x1 > x0 or not y1 > y0 or min(h, w) < 2:
        raise ValueError('Nonempty crop and output H,W >= 2 required')
    sx, sy = w / (x1-x0), h / (y1-y0)
    return torch.tensor([[sx, 0., sx*(.5-x0)-.5],
                         [0., sy, sy*(.5-y0)-.5], [0., 0., 1.]], dtype=dtype, device=device)


def transform_uv(uv, A):
    hom = torch.cat((uv, torch.ones_like(uv[..., :1])), -1)
    q = hom @ A.transpose(-1, -2)
    return q[..., :2] / q[..., 2:3]


def rays(uv, K):
    check_K(K)
    hom = torch.cat((uv, torch.ones_like(uv[..., :1])), -1)
    r = hom @ torch.linalg.inv(K).transpose(-1, -2)
    return r / r[..., 2:3]


def pixel_grid(hw, *, device=None, dtype=torch.float32):
    h, w = hw
    v, u = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                          torch.arange(w, device=device, dtype=dtype), indexing='ij')
    return torch.stack((u, v), -1)


def inside(uv, hw, margin=0.):
    h, w = hw
    return (torch.isfinite(uv).all(-1) & (uv[..., 0] >= margin) &
            (uv[..., 0] <= w-1-margin) & (uv[..., 1] >= margin) & (uv[..., 1] <= h-1-margin))


def sample(image, uv, mode='bilinear'):
    """[B,C,H,W], [B,N,2] -> [B,N,C]; no clamping of outside rays."""
    h, w = image.shape[-2:]
    grid = 2*(uv+.5)/uv.new_tensor([w, h])-1
    return F.grid_sample(image, grid[:, :, None], mode=mode, padding_mode='zeros',
                         align_corners=False).squeeze(-1).transpose(1, 2)


def crop_rgb(rgb, K, box, out_hw):
    """Independently resample raw [3,H,W] RGB; return crop, AK, A, valid mask."""
    A = crop_matrix(box, out_hw, device=rgb.device, dtype=rgb.dtype)
    uv = transform_uv(pixel_grid(out_hw, device=rgb.device).reshape(-1, 2), torch.linalg.inv(A))
    valid = inside(uv, rgb.shape[-2:])
    cropped = sample(rgb[None], uv[None])[0].T.reshape(3, *out_hw)
    # ImageNet mean padding -> exactly zero after normalization.
    mean = rgb.new_tensor([.485, .456, .406])[:, None, None]
    cropped = torch.where(valid.reshape(1, *out_hw), cropped, mean)
    std = rgb.new_tensor([.229, .224, .225])[:, None, None]
    return (cropped-mean)/std, A @ K, A, valid.reshape(*out_hw)


def plane_from_table(T):
    """T maps camera metres to calibrated TABLE frame, whose z=0 is the plane."""
    if T.shape[-2:] != (4, 4) or not torch.isfinite(T).all():
        raise ValueError('Finite T_table_from_cam required')
    R = T[..., :3, :3]
    eye = torch.eye(3, device=T.device, dtype=T.dtype).expand_as(R)
    if (not torch.allclose(R.transpose(-1,-2) @ R, eye, atol=2e-4) or
        not torch.allclose(torch.linalg.det(R), torch.ones_like(T[..., 0, 0]), atol=2e-4) or
        not torch.allclose(T[..., 3, :], T.new_tensor([0.,0.,0.,1.]).expand_as(T[...,3,:]))):
        raise ValueError('A proper rigid camera-to-table transform is required')
    n, b = T[..., 2, :3], T[..., 2, 3]
    if (b <= 0).any():
        raise ValueError('Camera must be above calibrated table; b > 0 metres')
    return torch.cat((n, b[..., None]), -1)


def ray_dot_plane(uv, K, plane):
    return (rays(uv, K)*plane[..., None, :3]).sum(-1)


def height_from_z(z, uv, K, plane):
    return ray_dot_plane(uv, K, plane)*z + plane[..., None, 3]


def z_from_height(height, uv, K, plane, grazing=.1):
    a = ray_dot_plane(uv, K, plane)
    eligible = a < -float(grazing)  # fixed input-only mask, shared by ALL table arms
    safe = torch.where(eligible, a, -torch.ones_like(a))
    z = (height-plane[..., None, 3])/safe
    return z, eligible  # preserve negative z on eligible rays for loss/invalid statistics


def balanced_mask_mean(values, mask):
    """Equal image weights; empty images fail, rather than silently losing a scene."""
    count = mask.sum(-1)
    if (count == 0).any():
        raise ValueError('An image has no eligible common rays')
    return ((values*mask).sum(-1)/count).mean()


def losses(z_a, z_b, target_z, raw_rays, mask, consistency_weight, scale_m=.01):
    """Identical two-crop supervision in all arms; only consistency coefficient changes."""
    if not all(torch.isfinite(v).all() for v in (z_a, z_b, target_z, raw_rays)):
        raise FloatingPointError('Nonfinite depth/ray prediction')
    absolute = balanced_mask_mean(((z_a-target_z).abs()+(z_b-target_z).abs())*.5/scale_m, mask)
    xyz = (z_a-z_b).abs()*torch.linalg.vector_norm(raw_rays, dim=-1)/scale_m
    consistency = balanced_mask_mean(xyz, mask)
    positivity = balanced_mask_mean((F.relu(.001-z_a)+F.relu(.001-z_b))*.5/scale_m, mask)
    return absolute + consistency_weight*consistency + positivity, {
        'metric_l1': absolute.detach(), 'xyz_consistency': consistency.detach(),
        'negative_depth_penalty': positivity.detach()}
