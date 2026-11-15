"""Camera-frame geometry contracts; no watertightness or hidden occupancy claims.

The field is an unsigned distance to VISIBLE surface samples, plus a signed
PROJECTIVE depth residual and a ray-observation state. Negative projective
residual means behind the first visible surface, NOT inside a solid object.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.spatial import cKDTree

STATES = ('free', 'surface_band', 'behind_visible_surface', 'unobserved')
DEFAULT_OFFSETS_M = (0., -.005, .005, -.010, .010, -.020, .020)


def check_camera(K, hw):
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(hw) < 2:
        raise ValueError('Finite 3x3 intrinsics and H,W>=2 required')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('Pinhole intrinsics with positive focal lengths required')
    return K


def pixels_and_valid(xyz, K, hw):
    x = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    K = check_camera(K, hw)
    p = x @ K.T
    z = x[:, 2]
    uv = p[:, :2] / np.maximum(z[:, None], 1e-8)
    h, w = map(int, hw)
    valid = np.isfinite(x).all(1) & (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
    return uv, valid


def align_depth(raw_depth_m, raw_K, crop_K, crop_hw):
    """Resample SAME-camera depth to the model crop, using calibrated rays.

    Handles crop/resize intrinsics. No extrinsic change is silently inferred.
    Nearest-neighbour sampling does not blend foreground and background depths.
    """
    raw = np.asarray(raw_depth_m, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Depth must be finite, nonnegative HxW metres; missing=0')
    kr = check_camera(raw_K, raw.shape)
    kc = check_camera(crop_K, crop_hw)
    h, w = map(int, crop_hw)
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
    source = pix @ (kr @ np.linalg.inv(kc)).T
    source = source[:, :2] / source[:, 2:3]
    valid = (source[:, 0] >= 0) & (source[:, 0] <= raw.shape[1]-1) & (source[:, 1] >= 0) & (source[:, 1] <= raw.shape[0]-1)
    ij = np.rint(source).astype(np.int64)
    out = np.zeros(h*w, dtype=np.float32)
    out[valid] = raw[ij[valid, 1], ij[valid, 0]]
    return out.reshape(h, w)


class VisibleSurfaceTeacher:
    """Labels relative to one clean rendered visible-depth image (not full CAD)."""
    def __init__(self, depth_m, K, truncation_m=.03, surface_band_m=.002):
        self.depth = np.asarray(depth_m, dtype=np.float32)
        if self.depth.ndim != 2 or not np.isfinite(self.depth).all() or (self.depth < 0).any():
            raise ValueError('Invalid depth map')
        self.K = check_camera(K, self.depth.shape)
        self.tau, self.band = float(truncation_m), float(surface_band_m)
        if not 0 < self.band < self.tau:
            raise ValueError('Require 0 < band < truncation')
        v, u = np.where(self.depth > 1e-6)
        if len(u) == 0:
            raise ValueError('No visible surface in teacher depth')
        rays = np.stack([u, v, np.ones_like(u)], 1) @ np.linalg.inv(self.K).T
        self.points = (rays * self.depth[v, u, None]).astype(np.float32)
        self.tree = cKDTree(self.points)

    def label(self, xyz):
        x = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite query coordinates')
        uv, in_image = pixels_and_valid(x, self.K, self.depth.shape)
        ids = np.flatnonzero(in_image)
        obs = np.zeros(len(x), dtype=np.float32)
        if len(ids):
            ij = np.rint(uv[ids]).astype(np.int64)
            obs[ids] = self.depth[ij[:, 1], ij[:, 0]]
        known = in_image & (obs > 1e-6)
        residual = obs - x[:, 2]
        state = np.full(len(x), 3, dtype=np.int64)
        state[known & (residual > self.band)] = 0
        state[known & (np.abs(residual) <= self.band)] = 1
        state[known & (residual < -self.band)] = 2
        distance = self.tree.query(x, k=1, workers=1)[0]
        # UDF is to sampled visible surface, and is only supervised for in-image
        # queries. No claim about Euclidean distance to an unobserved object back.
        return dict(udf=np.minimum(distance/self.tau, 1).astype(np.float32),
                    projective=np.clip(residual/self.tau, -1, 1).astype(np.float32),
                    state=state, udf_mask=in_image, projective_mask=known)

    def sample(self, n, seed, support_xyz=None):
        """Surface-near + free/behind rays + action support; labels remain offline."""
        rng = np.random.default_rng(seed)
        parts = []
        if support_xyz is not None and len(support_xyz):
            support = np.asarray(support_xyz).reshape(-1, 3)
            parts.append(support[rng.integers(0, len(support), size=n//2)])
        left = n - sum(len(x) for x in parts)
        near = self.points[rng.integers(0, len(self.points), size=left//2)].copy()
        near += rng.normal(0, self.tau*.5, size=near.shape).astype(np.float32)
        parts.append(near)
        count = n - sum(len(x) for x in parts)
        h, w = self.depth.shape
        uv = np.stack([rng.uniform(-.05*w, 1.05*w, count), rng.uniform(-.05*h, 1.05*h, count), np.ones(count)], 1)
        rays = uv @ np.linalg.inv(self.K).T
        z = rng.uniform(max(.05, float(np.quantile(self.points[:, 2], .05))-.15),
                        float(np.quantile(self.points[:, 2], .95))+.15, count)
        parts.append((rays*z[:, None]).astype(np.float32))
        x = np.concatenate(parts).astype(np.float32)
        return x, self.label(x)


def action_bank(grasps, offsets=DEFAULT_OFFSETS_M):
    """One controlled step along the camera ray; R, width, height, insertion fixed.

    Offsets are Euclidean metres along the unit ray, NOT an insertion-bin change.
    Native action is index zero. This is a finite search, not continuous SE(3).
    """
    g = np.asarray(grasps, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    if g.ndim != 2 or g.shape[1] != 17 or not np.isfinite(g).all():
        raise ValueError('Expected finite N x 17 grasps')
    if len(offsets) < 2 or offsets[0] != 0 or len(np.unique(offsets)) != len(offsets) or np.max(np.abs(offsets)) > .05:
        raise ValueError('Unique bounded offsets, native first, required')
    direction = g[:, 13:16] / np.maximum(np.linalg.norm(g[:, 13:16], axis=1, keepdims=True), 1e-8)
    bank = np.repeat(g[:, None], len(offsets), axis=1)
    bank[:, :, 13:16] += direction[:, None]*offsets[None, :, None]
    if (bank[:, :, 15] <= 0).any():
        raise ValueError('Action refinement crossed camera plane')
    if not np.array_equal(bank[:, 0], g):
        raise RuntimeError('Native action drift')
    return bank


def support_points(grasps):
    """5x5x3 grid covering the executed-width gripper envelope.

    Official convention: local x approach, y opening, z height. Palm spans
    x=[-.03,-.02], fingers extend to insertion depth; finger width .01m.
    These samples are a measurement stencil, not a collision certificate.
    """
    g = grasps
    if g.shape[-1] != 17:
        raise ValueError('17D grasp required')
    axes = [torch.linspace(0, 1, n, device=g.device, dtype=g.dtype) for n in (5, 5, 3)]
    u = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(75, 3)
    local = torch.empty((*g.shape[:-1], 75, 3), device=g.device, dtype=g.dtype)
    local[..., 0] = -.03 + (g[..., 3, None]+.03)*u[:, 0]
    local[..., 1] = (g[..., 1, None]+.02)*(u[:, 1]-.5)
    local[..., 2] = g[..., 2, None]*(u[:, 2]-.5)
    rot = g[..., 4:13].reshape(*g.shape[:-1], 3, 3)
    world = torch.matmul(local, rot.transpose(-1, -2)) + g[..., None, 13:16]
    return world, local


def relative_coordinates(points, grasps):
    return torch.matmul(points - grasps[..., None, 13:16],
                        grasps[..., 4:13].reshape(*grasps.shape[:-1], 3, 3))
