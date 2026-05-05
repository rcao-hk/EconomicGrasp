import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW tensors."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCHW -> BHWC -> BCHW
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_perm = F.layer_norm(
            x_perm,
            normalized_shape=(self.num_channels,),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
        return x_perm.permute(0, 3, 1, 2).contiguous()


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class GraspSpatialEnhancer(nn.Module):
    """
    Single-scale BIP3D-style Spatial Enhancer for economicgrasp_dpt.

    It injects camera-aware probabilistic 3D position information into a dense
    2D feature map, using the depth distribution already predicted by DPT.

    Inputs:
        feat_2d:
            (B, C, Hf, Wf), e.g. proposal_path1 from DPTHead.
        depth_prob:
            (B, D, Hd, Wd), depth distribution over fixed bins.
            Usually depth_prob_448 from DINOv2DepthDistributionNet.
        K:
            (B, 3, 3), camera intrinsics in the same image coordinate system
            as image_hw.
        image_hw:
            (H_img, W_img), usually (448, 448).

    Output:
        enhanced_feat:
            (B, C, Hf, Wf), same shape as feat_2d.
        aux:
            scalar diagnostics for logging.
    """

    def __init__(
        self,
        embed_dims: int = 128,
        feature_3d_dim: int = 32,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        num_depth: int = 256,
        detach_depth_prob: bool = True,
        use_prob_embedding: bool = False,
        use_post_norm: bool = False,
        prob_eps: float = 1e-6,
        
        # sensor depth feature
        use_sensor_depth_feat: bool = True,
        depth_feat_dim: int = 128,
        depth_norm_fn: str = "layer",
        depth_downsample: int = 2,
        
        # visualization
        vis_dir: Optional[str] = None,
        vis_every: int = 500,
        vis_dpi: int = 150,
        vis_rank0_only: bool = True,
        save_vis_npz: bool = True,
    ):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.feature_3d_dim = int(feature_3d_dim)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.num_depth = int(num_depth)
        self.detach_depth_prob = bool(detach_depth_prob)
        self.use_prob_embedding = bool(use_prob_embedding)
        self.use_post_norm = bool(use_post_norm)
        self.prob_eps = float(prob_eps)

        self.use_sensor_depth_feat = bool(use_sensor_depth_feat)
        self.depth_feat_dim = int(depth_feat_dim)
        
        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.vis_dpi = int(vis_dpi)
        self.vis_rank0_only = bool(vis_rank0_only)
        self.save_vis_npz = bool(save_vis_npz)
        self._vis_iter = 0

        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)
            
        depth_bins = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.num_depth,
            dtype=torch.float32,
        )
        self.register_buffer("depth_bins", depth_bins, persistent=False)

        # This mirrors BIP3D's linear 3D point embedding.
        # Because it is linear, we can use E[xyz] instead of enumerating all xyz_k.
        self.pts_fc = nn.Linear(3, self.feature_3d_dim)

        if self.use_sensor_depth_feat:
            self.depth_adapter = SensorDepthAdapter(
                feature_3d_dim=self.depth_feat_dim,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                norm_fn=depth_norm_fn,
                downsample=depth_downsample,
            )
        else:
            self.depth_adapter = None

        # Extra grasp-friendly scalar geometry:
        # ray_unit(3), normalized mean depth(1), normalized std depth(1), entropy(1)
        # scalar_in_dim = 6
        # self.scalar_geom_fc = nn.Sequential(
        #     nn.Conv2d(scalar_in_dim, self.feature_3d_dim, kernel_size=1, bias=False),
        #     _make_group_norm(self.feature_3d_dim),
        #     nn.GELU(),
        #     nn.Conv2d(self.feature_3d_dim, self.feature_3d_dim, kernel_size=1),
        # )

        fusion_in_dim = (
            self.embed_dims
            + self.feature_3d_dim  # probabilistic 3D position embedding from predicted depth
        )

        if self.use_sensor_depth_feat:
            fusion_in_dim += self.depth_feat_dim
            
        if self.use_prob_embedding:
            # Optional. More expressive, but heavier at 448x448 and D=256.
            self.prob_fc = nn.Sequential(
                nn.Conv2d(self.num_depth, self.feature_3d_dim, kernel_size=1, bias=False),
                _make_group_norm(self.feature_3d_dim),
                nn.GELU(),
                nn.Conv2d(self.feature_3d_dim, self.feature_3d_dim, kernel_size=1),
            )
            fusion_in_dim += self.feature_3d_dim
        else:
            self.prob_fc = None

        self.delta_fc = nn.Sequential(
            nn.Conv2d(fusion_in_dim, self.embed_dims, kernel_size=1, bias=False),
            _make_group_norm(self.embed_dims),
            nn.GELU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
        )

        self.gate_fc = nn.Sequential(
            nn.Conv2d(fusion_in_dim, self.embed_dims, kernel_size=1),
            nn.Sigmoid(),
        )
        
        if self.use_post_norm:
            self.post_norm = LayerNorm2d(self.embed_dims)
        else:
            self.post_norm = nn.Identity()

    @staticmethod
    def _resize_and_normalize_prob(
        depth_prob: torch.Tensor,
        size_hw: Tuple[int, int],
        eps: float,
    ) -> torch.Tensor:
        """
        Resize depth distribution to feature resolution, then re-normalize
        over depth bins. Bilinear interpolation does not strictly preserve
        sum-to-one, so renormalization is necessary.
        """
        if depth_prob.shape[-2:] != size_hw:
            depth_prob = F.interpolate(
                depth_prob,
                size=size_hw,
                mode="bilinear",
                align_corners=False,
            )

        depth_prob = depth_prob.clamp_min(0.0)
        depth_prob = depth_prob / depth_prob.sum(dim=1, keepdim=True).clamp_min(eps)
        return depth_prob

    @staticmethod
    def _camera_rays(
        B: int,
        Hf: int,
        Wf: int,
        image_hw: Tuple[int, int],
        K: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ray:
                (B, 3, Hf, Wf), unnormalized pinhole ray [x/z, y/z, 1].
                This should be multiplied by depth to obtain xyz.
            ray_unit:
                (B, 3, Hf, Wf), normalized direction, useful as a feature.
        """
        H_img, W_img = int(image_hw[0]), int(image_hw[1])

        # Feature-cell centers in the original image coordinate system.
        ys = (torch.arange(Hf, device=device, dtype=dtype) + 0.5) * (H_img / Hf) - 0.5
        xs = (torch.arange(Wf, device=device, dtype=dtype) + 0.5) * (W_img / Wf) - 0.5
        v, u = torch.meshgrid(ys, xs, indexing="ij")

        u = u.view(1, 1, Hf, Wf).expand(B, -1, -1, -1)
        v = v.view(1, 1, Hf, Wf).expand(B, -1, -1, -1)

        if K.shape[-2:] == (4, 4):
            K = K[:, :3, :3]
        K = K.to(device=device, dtype=dtype)

        fx = K[:, 0, 0].view(B, 1, 1, 1).clamp_min(1e-6)
        fy = K[:, 1, 1].view(B, 1, 1, 1).clamp_min(1e-6)
        cx = K[:, 0, 2].view(B, 1, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1, 1)

        x_over_z = (u - cx) / fx
        y_over_z = (v - cy) / fy
        ones = torch.ones_like(x_over_z)

        ray = torch.cat([x_over_z, y_over_z, ones], dim=1)
        ray_unit = F.normalize(ray, dim=1, eps=1e-6)
        return ray, ray_unit

    def _depth_moments(
        self,
        depth_prob: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean depth, std depth, and entropy from depth distribution.

        Returns:
            mean_z:  (B,1,H,W)
            std_z:   (B,1,H,W)
            entropy: (B,1,H,W), normalized to [0,1]
        """
        p = depth_prob.float()
        bins = self.depth_bins.to(device=p.device, dtype=p.dtype).view(1, self.num_depth, 1, 1)

        mean_z = (p * bins).sum(dim=1, keepdim=True)
        mean_z2 = (p * bins.square()).sum(dim=1, keepdim=True)
        std_z = (mean_z2 - mean_z.square()).clamp_min(1e-10).sqrt()

        p_safe = p.clamp_min(1e-8)
        entropy = -(p_safe * p_safe.log()).sum(dim=1, keepdim=True) / math.log(self.num_depth)
        entropy = entropy.clamp(0.0, 1.0)

        return mean_z.to(out_dtype), std_z.to(out_dtype), entropy.to(out_dtype)

    # -------------------------------------------------------------------------
    # visualization helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _lazy_plt():
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        return plt

    @staticmethod
    def _is_main_process() -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x):
            x = x.detach().float().cpu()
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x.numpy()
        x = np.asarray(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    @staticmethod
    def _robust_minmax(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0):
        x = np.asarray(x)
        finite = np.isfinite(x)
        if finite.sum() == 0:
            return 0.0, 1.0
        lo = float(np.nanpercentile(x[finite], p_low))
        hi = float(np.nanpercentile(x[finite], p_high))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    def _to_rgb_np(self, img_chw: torch.Tensor) -> np.ndarray:
        """
        Convert CHW image to displayable RGB.
        Works even if image was ImageNet-normalized, because it uses robust min-max.
        """
        x = img_chw.detach().float().cpu()

        if x.dim() != 3:
            raise ValueError(f"Expected CHW image, got {tuple(x.shape)}")

        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] > 3:
            x = x[:3]

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.permute(1, 2, 0).numpy()

        lo, hi = self._robust_minmax(x, 1.0, 99.0)
        x = (x - lo) / (hi - lo + 1e-6)
        return np.clip(x, 0.0, 1.0)

    def _map_to_hw_np(self, map_2d, size_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert map to 2D numpy. If size_hw is provided, resize using bilinear.
        """
        if torch.is_tensor(map_2d):
            t = map_2d.detach().float()

            while t.dim() > 2:
                t = t[0]

            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

            if size_hw is not None and tuple(t.shape[-2:]) != tuple(size_hw):
                t = F.interpolate(
                    t.view(1, 1, t.shape[-2], t.shape[-1]),
                    size=size_hw,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

            return t.cpu().numpy()

        arr = np.asarray(map_2d)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim > 2:
            arr = arr.squeeze()
        return arr

    def _save_map_png(
        self,
        arr2d,
        out_path: str,
        title: Optional[str] = None,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        colorbar: bool = True,
    ):
        plt = self._lazy_plt()
        arr = self._map_to_hw_np(arr2d)

        if vmin is None or vmax is None:
            lo, hi = self._robust_minmax(arr)
            if vmin is None:
                vmin = lo
            if vmax is None:
                vmax = hi

        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6

        plt.figure(figsize=(6, 5))
        im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        if colorbar:
            plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout(pad=0.1)
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _save_overlay_png(
        self,
        img_chw: torch.Tensor,
        map_2d,
        out_path: str,
        title: Optional[str] = None,
        cmap: str = "magma",
        alpha: float = 0.45,
        vmin=None,
        vmax=None,
    ):
        plt = self._lazy_plt()

        img_np = self._to_rgb_np(img_chw)
        H, W = img_np.shape[:2]
        arr = self._map_to_hw_np(map_2d, size_hw=(H, W))

        if vmin is None or vmax is None:
            lo, hi = self._robust_minmax(arr)
            if vmin is None:
                vmin = lo
            if vmax is None:
                vmax = hi

        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6

        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        im = plt.imshow(arr, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout(pad=0.0)
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _save_hist_png(
        self,
        values,
        out_path: str,
        title: Optional[str] = None,
        xlabel: str = "value",
        bins: int = 80,
    ):
        plt = self._lazy_plt()
        arr = self._to_numpy(values).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return

        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _feature_pca_rgb(
        self,
        feat_chw: torch.Tensor,
        max_points: int = 4096,
    ) -> Optional[np.ndarray]:
        """
        Visualize a CxHxW feature map using PCA to RGB.
        This is useful for seeing whether GSE changes feature separability.
        """
        x = feat_chw.detach().float().cpu()
        if x.dim() != 3:
            return None

        C, H, W = x.shape
        if C < 1 or H < 2 or W < 2:
            return None

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mat = x.permute(1, 2, 0).reshape(-1, C)  # (HW,C)
        mat = mat - mat.mean(dim=0, keepdim=True)

        N = mat.shape[0]
        if N > max_points:
            idx = torch.linspace(0, N - 1, max_points).long()
            fit = mat[idx]
        else:
            fit = mat

        try:
            # fit: (N,C), Vh: (min(N,C), C)
            _, _, Vh = torch.linalg.svd(fit, full_matrices=False)
            comp = Vh[: min(3, Vh.shape[0])].T  # (C,<=3)
            rgb = mat @ comp                    # (HW,<=3)
        except Exception:
            # Fallback: use first channels.
            rgb = mat[:, : min(3, C)]

        if rgb.shape[1] < 3:
            pad = torch.zeros(rgb.shape[0], 3 - rgb.shape[1])
            rgb = torch.cat([rgb, pad], dim=1)

        rgb = rgb.reshape(H, W, 3).numpy()
        for c in range(3):
            lo, hi = self._robust_minmax(rgb[..., c], 1.0, 99.0)
            rgb[..., c] = (rgb[..., c] - lo) / (hi - lo + 1e-6)

        return np.clip(rgb, 0.0, 1.0)

    def _save_rgb_png(self, rgb: np.ndarray, out_path: str, title: Optional[str] = None):
        plt = self._lazy_plt()
        plt.figure(figsize=(6, 6))
        plt.imshow(np.clip(rgb, 0.0, 1.0))
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0.0)
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    def _save_summary_grid(
        self,
        items,
        out_path: str,
        cols: int = 4,
    ):
        """
        items: list of dict:
            {
                "title": str,
                "data": tensor or np.ndarray,
                "cmap": str,
                "vmin": optional,
                "vmax": optional,
                "rgb": bool,
            }
        """
        plt = self._lazy_plt()

        if len(items) == 0:
            return

        rows = int(math.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
        if rows == 1:
            axes = np.asarray([axes])
        axes = axes.reshape(rows, cols)

        for ax in axes.reshape(-1):
            ax.axis("off")

        for i, item in enumerate(items):
            ax = axes.reshape(-1)[i]
            ax.axis("off")

            title = item.get("title", "")
            is_rgb = item.get("rgb", False)

            if is_rgb:
                arr = item["data"]
                ax.imshow(np.clip(arr, 0.0, 1.0))
                ax.set_title(title)
                continue

            arr = self._map_to_hw_np(item["data"])
            vmin = item.get("vmin", None)
            vmax = item.get("vmax", None)
            if vmin is None or vmax is None:
                lo, hi = self._robust_minmax(arr)
                if vmin is None:
                    vmin = lo
                if vmax is None:
                    vmax = hi
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1e-6

            im = ax.imshow(
                arr,
                cmap=item.get("cmap", "viridis"),
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(out_path, dpi=self.vis_dpi)
        plt.close()

    @torch.no_grad()
    def _maybe_visualize(
        self,
        *,
        feat_2d: torch.Tensor,
        prob: torch.Tensor,
        mean_z: torch.Tensor,
        std_z: torch.Tensor,
        entropy: torch.Tensor,
        ray_unit: torch.Tensor,
        ipe: torch.Tensor,
        # scalar_geom_feat: torch.Tensor,
        gate: torch.Tensor,
        delta: torch.Tensor,
        out: torch.Tensor,
        depth_feat: Optional[torch.Tensor] = None,
        sensor_depth: Optional[torch.Tensor] = None,
        img: Optional[torch.Tensor] = None,
        vis_prefix: Optional[str] = None,
    ):
        if self.vis_dir is None:
            return
        if self.vis_every <= 0:
            return
        if self._vis_iter % self.vis_every != 0:
            return
        if self.vis_rank0_only and not self._is_main_process():
            return

        try:
            b0 = 0
            rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()

            if vis_prefix is None:
                prefix = f"gse_r{rank}_it{self._vis_iter:06d}"
            else:
                prefix = f"{vis_prefix}_gse_r{rank}_it{self._vis_iter:06d}"

            out_dir = self.vis_dir
            os.makedirs(out_dir, exist_ok=True)

            # ------------------------------------------------------------------
            # Derived maps
            # ------------------------------------------------------------------
            prob_peak = prob.max(dim=1, keepdim=True).values  # (B,1,Hf,Wf)

            top_idx = prob.argmax(dim=1, keepdim=True)        # (B,1,Hf,Wf)
            bins = self.depth_bins.to(device=prob.device, dtype=prob.dtype).view(1, self.num_depth, 1, 1)
            top_z = bins.expand(prob.shape[0], -1, prob.shape[-2], prob.shape[-1]).gather(1, top_idx)

            feat_norm = feat_2d.detach().float().norm(dim=1, keepdim=True)
            out_norm = out.detach().float().norm(dim=1, keepdim=True)
            ipe_norm = ipe.detach().float().norm(dim=1, keepdim=True)
            # geom_norm = scalar_geom_feat.detach().float().norm(dim=1, keepdim=True)

            gate_mean = gate.detach().float().mean(dim=1, keepdim=True)
            delta_abs = delta.detach().float().abs().mean(dim=1, keepdim=True)
            update_abs = (gate * delta).detach().float().abs().mean(dim=1, keepdim=True)
            feat_abs = feat_2d.detach().float().abs().mean(dim=1, keepdim=True)
            update_ratio = update_abs / feat_abs.clamp_min(1e-6)

            ray_x = ray_unit[:, 0:1]
            ray_y = ray_unit[:, 1:2]
            ray_z = ray_unit[:, 2:3]

            # ------------------------------------------------------------------
            # Sensor depth visualization maps
            # ------------------------------------------------------------------
            sensor_depth_vis = None          # raw/resized sensor depth map, invalid = 0
            sensor_valid_vis = None          # raw valid mask
            sensor_depth_feat_rs = None      # sensor depth resized to feature map resolution
            sensor_valid_feat = None         # valid mask resized to feature map resolution
            pred_sensor_absdiff = None       # |DPT mean_z - sensor depth| at feature resolution
            pred_sensor_signed = None        # DPT mean_z - sensor depth

            if sensor_depth is not None:
                sd = sensor_depth.detach()
                if sd.dim() == 3:
                    sd = sd.unsqueeze(1)
                sd = sd.to(device=mean_z.device, dtype=mean_z.dtype)

                sensor_valid_vis = (
                    (sd > self.min_depth)
                    & (sd < self.max_depth)
                    & torch.isfinite(sd)
                ).float()

                sd_clamped = sd.clamp(self.min_depth, self.max_depth)
                sensor_depth_vis = torch.where(
                    sensor_valid_vis > 0.5,
                    sd_clamped,
                    torch.zeros_like(sd_clamped),
                )

                if sensor_depth_vis.shape[-2:] != mean_z.shape[-2:]:
                    sensor_depth_feat_rs = F.interpolate(
                        sensor_depth_vis,
                        size=mean_z.shape[-2:],
                        mode="nearest",
                    )
                    sensor_valid_feat = F.interpolate(
                        sensor_valid_vis,
                        size=mean_z.shape[-2:],
                        mode="nearest",
                    )
                else:
                    sensor_depth_feat_rs = sensor_depth_vis
                    sensor_valid_feat = sensor_valid_vis

                pred_sensor_signed = (mean_z - sensor_depth_feat_rs) * sensor_valid_feat
                pred_sensor_absdiff = pred_sensor_signed.abs()
                
            # ------------------------------------------------------------------
            # Save compact summary grid
            # ------------------------------------------------------------------
            grid_items = []

            if img is not None:
                img_b0 = img[b0]
                rgb_np = self._to_rgb_np(img_b0)
                self._save_rgb_png(
                    rgb_np,
                    os.path.join(out_dir, f"{prefix}_rgb.png"),
                    title="RGB input",
                )
                grid_items.append({"title": "RGB", "data": rgb_np, "rgb": True})
            else:
                img_b0 = None
                
            if depth_feat is not None:
                depth_feat_norm = depth_feat.detach().float().norm(dim=1, keepdim=True)
            else:
                depth_feat_norm = None

            # grid_items.extend([
            #     {
            #         "title": "mean_z",
            #         "data": mean_z[b0, 0],
            #         "cmap": "viridis",
            #         "vmin": self.min_depth,
            #         "vmax": self.max_depth,
            #     },
            #     {
            #         "title": "top_z(argmax prob)",
            #         "data": top_z[b0, 0],
            #         "cmap": "viridis",
            #         "vmin": self.min_depth,
            #         "vmax": self.max_depth,
            #     },
            #     {
            #         "title": "std_z",
            #         "data": std_z[b0, 0],
            #         "cmap": "magma",
            #         "vmin": 0.0,
            #     },
            #     {
            #         "title": "entropy",
            #         "data": entropy[b0, 0],
            #         "cmap": "magma",
            #         "vmin": 0.0,
            #         "vmax": 1.0,
            #     },
            #     {
            #         "title": "prob_peak",
            #         "data": prob_peak[b0, 0],
            #         "cmap": "viridis",
            #         "vmin": 0.0,
            #         "vmax": 1.0,
            #     },
            #     {
            #         "title": "gate_mean",
            #         "data": gate_mean[b0, 0],
            #         "cmap": "viridis",
            #         "vmin": 0.0,
            #         "vmax": 1.0,
            #     },
            #     {
            #         "title": "delta_abs",
            #         "data": delta_abs[b0, 0],
            #         "cmap": "plasma",
            #         "vmin": 0.0,
            #     },
            #     {
            #         "title": "update_abs=|gate*delta|",
            #         "data": update_abs[b0, 0],
            #         "cmap": "plasma",
            #         "vmin": 0.0,
            #     },
            #     {
            #         "title": "update_ratio",
            #         "data": update_ratio[b0, 0],
            #         "cmap": "plasma",
            #         "vmin": 0.0,
            #     },
            #     {
            #         "title": "feat_norm_before",
            #         "data": feat_norm[b0, 0],
            #         "cmap": "viridis",
            #         "vmin": 0.0,
            #     },
            #     {
            #         "title": "feat_norm_after",
            #         "data": out_norm[b0, 0],
            #         "cmap": "viridis",
            #         "vmin": 0.0,
            #     },
            #     {
            #         "title": "ipe_norm",
            #         "data": ipe_norm[b0, 0],
            #         "cmap": "viridis",
            #         "vmin": 0.0,
            #     },
            #     # {
            #     #     "title": "scalar_geom_norm",
            #     #     "data": geom_norm[b0, 0],
            #     #     "cmap": "viridis",
            #     #     "vmin": 0.0,
            #     # },
            #     {
            #         "title": "ray_x",
            #         "data": ray_x[b0, 0],
            #         "cmap": "coolwarm",
            #         "vmin": -1.0,
            #         "vmax": 1.0,
            #     },
            #     {
            #         "title": "ray_y",
            #         "data": ray_y[b0, 0],
            #         "cmap": "coolwarm",
            #         "vmin": -1.0,
            #         "vmax": 1.0,
            #     },
            #     {
            #         "title": "ray_z",
            #         "data": ray_z[b0, 0],
            #         "cmap": "viridis",
            #         "vmin": 0.0,
            #         "vmax": 1.0,
            #     },
            # ])
            
            # if depth_feat_norm is not None:
            #     grid_items.append({
            #         "title": "sensor_depth_feat_norm",
            #         "data": depth_feat_norm[b0, 0],
            #         "cmap": "viridis",
            #         "vmin": 0.0,
            #     })

            # if sensor_depth_vis is not None:
            #     grid_items.extend([
            #         {
            #             "title": "sensor_depth",
            #             "data": sensor_depth_vis[b0, 0],
            #             "cmap": "viridis",
            #             "vmin": self.min_depth,
            #             "vmax": self.max_depth,
            #         },
            #         {
            #             "title": "sensor_valid",
            #             "data": sensor_valid_vis[b0, 0],
            #             "cmap": "gray",
            #             "vmin": 0.0,
            #             "vmax": 1.0,
            #         },
            #         {
            #             "title": "|pred_z - sensor_z|",
            #             "data": pred_sensor_absdiff[b0, 0],
            #             "cmap": "magma",
            #             "vmin": 0.0,
            #         },
            #         {
            #             "title": "pred_z - sensor_z",
            #             "data": pred_sensor_signed[b0, 0],
            #             "cmap": "coolwarm",
            #         },
            #     ])
                                                
            # self._save_summary_grid(
            #     grid_items,
            #     os.path.join(out_dir, f"{prefix}_summary.png"),
            #     cols=4,
            # )

            # ------------------------------------------------------------------
            # Individual maps
            # ------------------------------------------------------------------
            self._save_map_png(
                mean_z[b0, 0],
                os.path.join(out_dir, f"{prefix}_mean_z.png"),
                title="expected depth E[z]",
                cmap="viridis",
                vmin=self.min_depth,
                vmax=self.max_depth,
            )
            self._save_map_png(
                top_z[b0, 0],
                os.path.join(out_dir, f"{prefix}_top_z.png"),
                title="argmax depth",
                cmap="viridis",
                vmin=self.min_depth,
                vmax=self.max_depth,
            )
            self._save_map_png(
                std_z[b0, 0],
                os.path.join(out_dir, f"{prefix}_std_z.png"),
                title="depth std",
                cmap="magma",
                vmin=0.0,
            )
            self._save_map_png(
                entropy[b0, 0],
                os.path.join(out_dir, f"{prefix}_entropy.png"),
                title="depth distribution entropy",
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            )
            self._save_map_png(
                prob_peak[b0, 0],
                os.path.join(out_dir, f"{prefix}_prob_peak.png"),
                title="max depth probability",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            self._save_map_png(
                gate_mean[b0, 0],
                os.path.join(out_dir, f"{prefix}_gate_mean.png"),
                title="GSE gate mean",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            self._save_map_png(
                delta_abs[b0, 0],
                os.path.join(out_dir, f"{prefix}_delta_abs.png"),
                title="GSE delta abs mean",
                cmap="plasma",
                vmin=0.0,
            )
            self._save_map_png(
                update_ratio[b0, 0],
                os.path.join(out_dir, f"{prefix}_update_ratio.png"),
                title="|gate*delta| / |feat|",
                cmap="plasma",
                vmin=0.0,
            )

            # ------------------------------------------------------------------
            # Overlay maps on RGB
            # ------------------------------------------------------------------
            if img_b0 is not None:
                self._save_overlay_png(
                    img_b0,
                    mean_z[b0, 0],
                    os.path.join(out_dir, f"{prefix}_overlay_mean_z.png"),
                    title="E[z] overlay",
                    cmap="viridis",
                    vmin=self.min_depth,
                    vmax=self.max_depth,
                )
                self._save_overlay_png(
                    img_b0,
                    entropy[b0, 0],
                    os.path.join(out_dir, f"{prefix}_overlay_entropy.png"),
                    title="depth entropy overlay",
                    cmap="magma",
                    vmin=0.0,
                    vmax=1.0,
                )
                self._save_overlay_png(
                    img_b0,
                    gate_mean[b0, 0],
                    os.path.join(out_dir, f"{prefix}_overlay_gate.png"),
                    title="GSE gate overlay",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                )
                self._save_overlay_png(
                    img_b0,
                    update_ratio[b0, 0],
                    os.path.join(out_dir, f"{prefix}_overlay_update_ratio.png"),
                    title="update ratio overlay",
                    cmap="plasma",
                    vmin=0.0,
                )

            if sensor_depth_vis is not None:
                self._save_map_png(
                    sensor_depth_vis[b0, 0],
                    os.path.join(out_dir, f"{prefix}_sensor_depth.png"),
                    title="sensor depth",
                    cmap="viridis",
                    vmin=self.min_depth,
                    vmax=self.max_depth,
                )
                self._save_map_png(
                    sensor_valid_vis[b0, 0],
                    os.path.join(out_dir, f"{prefix}_sensor_valid.png"),
                    title="sensor depth valid mask",
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                self._save_map_png(
                    pred_sensor_absdiff[b0, 0],
                    os.path.join(out_dir, f"{prefix}_pred_sensor_absdiff.png"),
                    title="|predicted mean_z - sensor depth|",
                    cmap="magma",
                    vmin=0.0,
                )
                self._save_map_png(
                    pred_sensor_signed[b0, 0],
                    os.path.join(out_dir, f"{prefix}_pred_sensor_signeddiff.png"),
                    title="predicted mean_z - sensor depth",
                    cmap="coolwarm",
                )

                if depth_feat_norm is not None:
                    self._save_map_png(
                        depth_feat_norm[b0, 0],
                        os.path.join(out_dir, f"{prefix}_sensor_depth_feat_norm.png"),
                        title="sensor depth feature norm",
                        cmap="viridis",
                        vmin=0.0,
                    )
                    
            # ------------------------------------------------------------------
            # Feature PCA maps
            # ------------------------------------------------------------------
            pca_before = self._feature_pca_rgb(feat_2d[b0])
            pca_after = self._feature_pca_rgb(out[b0])
            pca_delta = self._feature_pca_rgb(delta[b0])
            pca_ipe = self._feature_pca_rgb(ipe[b0])
            # pca_geom = self._feature_pca_rgb(scalar_geom_feat[b0])

            if pca_before is not None:
                self._save_rgb_png(
                    pca_before,
                    os.path.join(out_dir, f"{prefix}_feat_pca_before.png"),
                    title="feature PCA before GSE",
                )
            if pca_after is not None:
                self._save_rgb_png(
                    pca_after,
                    os.path.join(out_dir, f"{prefix}_feat_pca_after.png"),
                    title="feature PCA after GSE",
                )
            if pca_delta is not None:
                self._save_rgb_png(
                    pca_delta,
                    os.path.join(out_dir, f"{prefix}_delta_pca.png"),
                    title="delta PCA",
                )
            if pca_ipe is not None:
                self._save_rgb_png(
                    pca_ipe,
                    os.path.join(out_dir, f"{prefix}_ipe_pca.png"),
                    title="IPE PCA",
                )
            # if pca_geom is not None:
            #     self._save_rgb_png(
            #         pca_geom,
            #         os.path.join(out_dir, f"{prefix}_scalar_geom_pca.png"),
            #         title="scalar geometry feature PCA",
            #     )

            # ------------------------------------------------------------------
            # Histograms
            # ------------------------------------------------------------------
            self._save_hist_png(
                entropy[b0],
                os.path.join(out_dir, f"{prefix}_hist_entropy.png"),
                title="entropy histogram",
                xlabel="entropy",
            )
            self._save_hist_png(
                std_z[b0],
                os.path.join(out_dir, f"{prefix}_hist_std_z.png"),
                title="depth std histogram",
                xlabel="std_z",
            )
            self._save_hist_png(
                prob_peak[b0],
                os.path.join(out_dir, f"{prefix}_hist_prob_peak.png"),
                title="depth prob peak histogram",
                xlabel="max prob",
            )
            self._save_hist_png(
                gate_mean[b0],
                os.path.join(out_dir, f"{prefix}_hist_gate_mean.png"),
                title="gate mean histogram",
                xlabel="gate",
            )
            self._save_hist_png(
                update_ratio[b0],
                os.path.join(out_dir, f"{prefix}_hist_update_ratio.png"),
                title="update ratio histogram",
                xlabel="|gate*delta| / |feat|",
            )

            # ------------------------------------------------------------------
            # Raw dump for offline inspection
            # ------------------------------------------------------------------
            if self.save_vis_npz:
                np.savez_compressed(
                    os.path.join(out_dir, f"{prefix}_maps.npz"),
                    mean_z=self._to_numpy(mean_z[b0, 0]),
                    top_z=self._to_numpy(top_z[b0, 0]),
                    std_z=self._to_numpy(std_z[b0, 0]),
                    entropy=self._to_numpy(entropy[b0, 0]),
                    prob_peak=self._to_numpy(prob_peak[b0, 0]),
                    gate_mean=self._to_numpy(gate_mean[b0, 0]),
                    delta_abs=self._to_numpy(delta_abs[b0, 0]),
                    update_abs=self._to_numpy(update_abs[b0, 0]),
                    update_ratio=self._to_numpy(update_ratio[b0, 0]),
                    feat_norm=self._to_numpy(feat_norm[b0, 0]),
                    out_norm=self._to_numpy(out_norm[b0, 0]),
                    ipe_norm=self._to_numpy(ipe_norm[b0, 0]),
                    # scalar_geom_norm=self._to_numpy(geom_norm[b0, 0]),
                    ray_x=self._to_numpy(ray_x[b0, 0]),
                    ray_y=self._to_numpy(ray_y[b0, 0]),
                    ray_z=self._to_numpy(ray_z[b0, 0]),
                )

        except Exception as e:
            print(f"[GraspSpatialEnhancer vis] failed at iter {self._vis_iter}: {repr(e)}")
            
    def forward(
        self,
        feat_2d: torch.Tensor,
        depth_prob: torch.Tensor,
        K: torch.Tensor,
        image_hw: Optional[Tuple[int, int]] = None,
        sensor_depth: Optional[torch.Tensor] = None,
        return_maps: bool = False,
        img: Optional[torch.Tensor] = None,
        vis_prefix: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert feat_2d.dim() == 4, f"feat_2d should be BCHW, got {feat_2d.shape}"
        assert depth_prob.dim() == 4, f"depth_prob should be BDHW, got {depth_prob.shape}"

        B, C, Hf, Wf = feat_2d.shape
        if C != self.embed_dims:
            raise ValueError(f"feat channel C={C} does not match embed_dims={self.embed_dims}")

        if depth_prob.size(1) != self.num_depth:
            raise ValueError(
                f"depth_prob has D={depth_prob.size(1)}, but enhancer expects num_depth={self.num_depth}"
            )

        if image_hw is None:
            image_hw = depth_prob.shape[-2:]

        dtype = feat_2d.dtype
        device = feat_2d.device

        prob = self._resize_and_normalize_prob(depth_prob, (Hf, Wf), self.prob_eps)
        if self.detach_depth_prob:
            prob = prob.detach()

        mean_z, std_z, entropy = self._depth_moments(prob, out_dtype=dtype)

        ray, ray_unit = self._camera_rays(
            B=B,
            Hf=Hf,
            Wf=Wf,
            image_hw=image_hw,
            K=K,
            device=device,
            dtype=dtype,
        )

        # E[xyz] = ray * E[z]
        mean_xyz = ray * mean_z  # (B,3,Hf,Wf)

        # BIP3D-style image position embedding.
        # Since pts_fc is linear, this equals sum_k p_k * pts_fc(xyz_k).
        ipe = self.pts_fc(
            mean_xyz.permute(0, 2, 3, 1).contiguous()
        ).permute(0, 3, 1, 2).contiguous()

        # z_range = max(self.max_depth - self.min_depth, 1e-6)
        # mean_z_norm = ((mean_z - self.min_depth) / z_range).clamp(0.0, 1.0)
        # std_z_norm = (std_z / z_range).clamp(0.0, 1.0)

        # scalar_geom = torch.cat(
        #     [
        #         ray_unit,
        #         mean_z_norm,
        #         std_z_norm,
        #         entropy,
        #     ],
        #     dim=1,
        # )
        # scalar_geom_feat = self.scalar_geom_fc(scalar_geom)
        # fuse_list = [feat_2d, ipe]

        # ------------------------------------------------------------------
        # Sensor depth feature only.
        # IPE still comes from predicted depth_prob.
        # Sensor depth does NOT generate depth distribution or depth prior.
        # ------------------------------------------------------------------
        depth_feat = None
        if self.use_sensor_depth_feat:
            if sensor_depth is None:
                raise ValueError(
                    "use_sensor_depth_feat=True, but sensor_depth is None. "
                    "Please pass sensor_depth to GraspSpatialEnhancer.forward()."
                )

            if sensor_depth.dim() == 3:
                sensor_depth = sensor_depth.unsqueeze(1)

            depth_feat = self.depth_adapter(
                sensor_depth=sensor_depth,
                out_hw=(Hf, Wf),
            )  # (B, depth_feat_dim, Hf, Wf)

        fuse_list = [feat_2d, ipe]

        if depth_feat is not None:
            fuse_list.append(depth_feat)

        if self.use_prob_embedding:
            # Keep this off for the first run. It is more expensive and can overfit.
            prob_feat = self.prob_fc(prob.to(dtype))
            fuse_list.append(prob_feat)

        fused = torch.cat(fuse_list, dim=1)

        delta = self.delta_fc(fused)
        gate = self.gate_fc(fused)

        out = feat_2d + gate * delta
        out = self.post_norm(out)

        self._maybe_visualize(
            feat_2d=feat_2d,
            prob=prob,
            mean_z=mean_z,
            std_z=std_z,
            entropy=entropy,
            ray_unit=ray_unit,
            ipe=ipe,
            # scalar_geom_feat=scalar_geom_feat,
            gate=gate,
            delta=delta,
            out=out,
            depth_feat=depth_feat,
            sensor_depth=sensor_depth,   # add this
            img=img,
            vis_prefix=vis_prefix,
        )
        self._vis_iter += 1
        
        aux: Dict[str, torch.Tensor] = {
            "D: GSE mean_z": mean_z.detach().mean(),
            "D: GSE std_z": std_z.detach().mean(),
            "D: GSE entropy": entropy.detach().mean(),
            "D: GSE gate_mean": gate.detach().mean(),
            "D: GSE delta_abs": delta.detach().abs().mean(),
        }

        if depth_feat is not None:
            aux["D: GSE depth_feat_abs"] = depth_feat.detach().abs().mean()
            aux["D: GSE depth_feat_norm"] = depth_feat.detach().norm(dim=1).mean()

        if return_maps:
            prob_peak = prob.max(dim=1, keepdim=True).values
            top_idx = prob.argmax(dim=1, keepdim=True)
            bins = self.depth_bins.to(device=prob.device, dtype=prob.dtype).view(1, self.num_depth, 1, 1)
            top_z = bins.expand(prob.shape[0], -1, prob.shape[-2], prob.shape[-1]).gather(1, top_idx)

            aux["spatial_mean_z_map"] = mean_z.detach()
            aux["spatial_std_z_map"] = std_z.detach()
            aux["spatial_entropy_map"] = entropy.detach()
            aux["spatial_prob_peak_map"] = prob_peak.detach()
            aux["spatial_top_z_map"] = top_z.detach()
            aux["spatial_gate_mean_map"] = gate.detach().mean(dim=1, keepdim=True)
            aux["spatial_delta_abs_map"] = delta.detach().abs().mean(dim=1, keepdim=True)
            aux["spatial_update_abs_map"] = (gate * delta).detach().abs().mean(dim=1, keepdim=True)
            if depth_feat is not None:
                aux["spatial_depth_feat_norm_map"] = depth_feat.detach().norm(dim=1, keepdim=True)

            if sensor_depth is not None:
                sd = sensor_depth.detach()
                if sd.dim() == 3:
                    sd = sd.unsqueeze(1)
                sd = sd.to(device=mean_z.device, dtype=mean_z.dtype)

                sensor_valid = (
                    (sd > self.min_depth)
                    & (sd < self.max_depth)
                    & torch.isfinite(sd)
                ).float()
                sensor_depth_vis = torch.where(
                    sensor_valid > 0.5,
                    sd.clamp(self.min_depth, self.max_depth),
                    torch.zeros_like(sd),
                )

                sensor_depth_feat_rs = F.interpolate(
                    sensor_depth_vis,
                    size=mean_z.shape[-2:],
                    mode="nearest",
                )
                sensor_valid_feat = F.interpolate(
                    sensor_valid,
                    size=mean_z.shape[-2:],
                    mode="nearest",
                )

                aux["spatial_sensor_depth_map"] = sensor_depth_vis
                aux["spatial_sensor_valid_map"] = sensor_valid
                aux["spatial_pred_sensor_absdiff_map"] = (
                    (mean_z.detach() - sensor_depth_feat_rs).abs() * sensor_valid_feat
                )
                
        return out, aux


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'layer':
            self.norm1 = LayerNorm2d(planes)
            self.norm2 = LayerNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = LayerNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)
            
    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    
    
class MultiBasicEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn=='layer':
            self.norm1 = LayerNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []

        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1:
            return (outputs04, v) if dual_inp else (outputs04,)

        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        if num_layers == 2:
            return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)

        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16, v) if dual_inp else (outputs04, outputs08, outputs16)
    

class SensorDepthAdapter(nn.Module):
    """
    Use MultiBasicEncoder and only take the largest-resolution feature map.

    Input:
        sensor_depth: (B,1,H,W), metric depth in meters
    Output:
        depth_feat:   (B,Cd,Hf,Wf)
    """
    def __init__(
        self,
        feature_3d_dim: int = 128,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        norm_fn: str = "group",
        downsample: int = 2,
    ):
        super().__init__()
        self.feature_3d_dim = int(feature_3d_dim)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        # output_dim must be tuple-like because MultiBasicEncoder uses dim[0/1/2].
        self.encoder = MultiBasicEncoder(
            input_dim=1,
            output_dim=[(feature_3d_dim, feature_3d_dim, feature_3d_dim)],
            norm_fn=norm_fn,
            dropout=0.0,
            downsample=downsample,
        )

    def forward(self, sensor_depth: torch.Tensor, out_hw):
        if sensor_depth.dim() == 3:
            sensor_depth = sensor_depth.unsqueeze(1)
        assert sensor_depth.dim() == 4 and sensor_depth.size(1) == 1

        valid = (
            (sensor_depth > self.min_depth)
            & (sensor_depth < self.max_depth)
            & torch.isfinite(sensor_depth)
        ).float()

        z = sensor_depth.clamp(self.min_depth, self.max_depth)
        # 朴素处理：invalid 直接置 0，不额外输入 valid mask。
        x = z * valid

        outputs04, outputs08, outputs16 = self.encoder(x, num_layers=3)

        # 最大空间分辨率 feature map
        depth_feat = outputs04[0]

        if depth_feat.shape[-2:] != tuple(out_hw):
            depth_feat = F.interpolate(
                depth_feat,
                size=out_hw,
                mode="bilinear",
                align_corners=False,
            )

        return depth_feat