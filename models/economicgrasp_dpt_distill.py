"""Minimal privileged-depth teacher -> RGB student distillation.

All stages use the same deterministic image-space FPS selector so output KD is
not confounded by 3D-FPS/image-query seed mismatch.  The modality distinction
is explicit and restricted to the geometry depth consumed by the grasp model:

* Stage 0 teacher: RGB proposal features + clean synthetic ``gt_depth_m``; the
  DPT metric-depth decoder remains checkpoint-compatible but is frozen and
  bypassed.
* Stage 1 student: RGB -> DPT metric depth, trained with the existing GT losses.
* Stage 2 student: the same RGB-only model plus frozen Stage-0 output KD; the
  student reuses the teacher's exact ordered image-FPS base indices.

Thus the experiment isolates whether task-specific grasp outputs from privileged
clean geometry can improve the RGB-only CVA-CDF student, without adding feature
KD, ray losses, collision heads, material augmentation, or multi-view training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Version 2 is the first contract in which Stage 0 is a true privileged
# clean-depth teacher rather than a predicted-depth self-distillation model.
DISTILL_CONTRACT_VERSION = 2


from utils.arguments import cfgs
from models.dinov2_dpt import DPTHead
from models.grasp_spatial_enhancer import GraspSpatialEnhancer
from models.rgb_geometry_diagnostics import RGBGeometryDiagnostics
from models.kview_query_transformer import (
    KViewQueryTransformerConfig,
    CenterViewAngleQueryTransformerLocalGraspModule,
)
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from models.economicgrasp_depth import (
    DINOv2DepthRegressionNet,
    DepthRefine,
)
from utils.label_generation import (
    generate_grasp_views,
    batch_viewpoint_params_to_matrix,
    process_grasp_labels_cdf_width,
    process_grasp_labels_extend_angle,
)

from models.economicgrasp_bip3d import GeometryAwareDenseFieldViewNet

class economicgrasp_dpt(nn.Module):
    """EconomicGrasp-DPT CVA model with an explicit geometry-depth source.

    The RGB proposal path is shared by both roles. ``geometry_depth_source``
    controls only the metric depth consumed by spatial enhancement, sparse
    backprojection, ViewNet, and CVA local analysis:

    * ``pred``: execute the DPT metric-depth decoder (RGB-only student);
    * ``gt``: use ``end_points["gt_depth_m"]`` and bypass/freeze the DPT
      decoder (privileged clean-depth teacher).

    Seed selection can be switched independently; the Stage-0--2 distillation
    experiment forces deterministic image-space FPS for both roles.
    """
    def __init__(
        self,
        encoder: str = 'vitb',
        tok_feat_dim: int = 128,
        cylinder_radius: float = 0.05,
        min_depth: float = 0.2,
        max_depth: float = 1.0,
        bin_num: int = 256,
        freeze_backbone: bool = True,
        use_gt_xyz_for_train: bool = False,
        is_training: bool = True,
        use_obs_depth: bool = False,
        pose_depth_mode: str = "none",
        camera_pose_key: str = "camera_pose_vec",
        camera_gravity_key: str = "camera_gravity_vec",
        pose_hidden_dim: int = 64,
        ray_gravity_hidden_dim: int = 64,
        ray_gravity_mid_dim: int = 32,
        use_depth_comp: bool = False,
        use_cdf: bool = False,
        vis_dir: Optional[str] = 'vis_dpt',
        vis_every: int = 500,
        debug_print_every: int = 50,
        seed_selection_mode: str = "point_fps",
        geometry_depth_source: str = "pred",
    ):
        super().__init__()
        self.is_training = bool(is_training)
        self.use_gt_xyz_for_train = bool(use_gt_xyz_for_train)
        self.seed_feature_dim = int(tok_feat_dim)
        self.num_depth = int(cfgs.num_depth)
        self.num_angle = int(cfgs.num_angle)
        self.M_points = int(cfgs.m_point)
        self.num_view = int(cfgs.num_view)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bin_num = int(bin_num)
        self.use_obs_depth = bool(use_obs_depth)
        self.geometry_depth_source = str(geometry_depth_source).strip().lower()
        if self.geometry_depth_source not in {"pred", "gt"}:
            raise ValueError(
                "geometry_depth_source must be 'pred' or 'gt', got "
                f"{geometry_depth_source!r}."
            )
        self.pose_depth_mode = str(pose_depth_mode or "none")
        self.camera_pose_key = str(camera_pose_key)
        self.camera_gravity_key = str(camera_gravity_key)
        self.pose_hidden_dim = int(pose_hidden_dim)
        self.ray_gravity_hidden_dim = int(ray_gravity_hidden_dim)
        self.ray_gravity_mid_dim = int(ray_gravity_mid_dim)
        if self.pose_depth_mode != "none" and self.use_obs_depth:
            raise ValueError(
                f"pose_depth_mode={self.pose_depth_mode!r} is an RGB-only "
                "depth setting and cannot be combined with "
                "use_obs_depth=True."
            )
        if self.geometry_depth_source == "gt":
            if self.use_obs_depth:
                raise ValueError(
                    "geometry_depth_source='gt' is the privileged clean-depth "
                    "teacher path and cannot be combined with use_obs_depth=True."
                )
            if self.pose_depth_mode != "none":
                raise ValueError(
                    "geometry_depth_source='gt' bypasses the metric-depth head; "
                    "pose_depth_mode must therefore be 'none'."
                )
        self.use_depth_comp = bool(use_depth_comp)
        # CDF is an explicit model choice. Geometry diagnostics follow
        # visualization automatically: a non-empty vis_dir enables them.
        self.use_cdf = bool(use_cdf)
        self.use_geometry_diagnostics = bool(vis_dir)

        self.stride = 1
        self.vis_dir = vis_dir
        self.vis_every = int(vis_every)
        self.debug_print_every = int(debug_print_every)
        self.seed_selection_mode = str(seed_selection_mode).strip().lower()
        if self.seed_selection_mode not in {"point_fps", "image_fps"}:
            raise ValueError(
                "seed_selection_mode must be 'point_fps' or 'image_fps', "
                f"got {seed_selection_mode!r}."
            )
        self._vis_iter = 0
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)
            
        # self.depth_net = DINOv2DepthDistributionNet(
        #     encoder=encoder,
        #     stride=self.stride,
        #     min_depth=self.min_depth,
        #     max_depth=self.max_depth,
        #     bin_num=self.bin_num,
        #     freeze_backbone=freeze_backbone,
        # )
        self.depth_net = DINOv2DepthRegressionNet(
            encoder=encoder,
            stride=self.stride,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            freeze_backbone=freeze_backbone,
            pose_depth_mode=self.pose_depth_mode,
            pose_hidden_dim=self.pose_hidden_dim,
            ray_gravity_hidden_dim=self.ray_gravity_hidden_dim,
            ray_gravity_mid_dim=self.ray_gravity_mid_dim,
        )
        if self.geometry_depth_source == "gt":
            # Keep the module/state-dict structure checkpoint-compatible, but the
            # privileged teacher never executes or optimizes the depth decoder.
            self.depth_net.requires_grad_(False)

        model_configs = {
            'vits': {'embed_dim': 384, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'embed_dim': 768, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'embed_dim': 1024, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'embed_dim': 1536, 'out_channels': [1536, 1536, 1536, 1536]},
        }
        cfg = model_configs[encoder]

        # One DPT head predicts [objectness_logit_0, objectness_logit_1, graspness]
        self.proposal_head = DPTHead(
            in_channels=cfg['embed_dim'],
            features=tok_feat_dim,
            use_bn=False,
            out_channels=cfg['out_channels'],
            out_dim=3,
            use_clstoken=True,
        )

        self.depth_refine_dim=32
        if self.use_obs_depth:
            depth_feat_dim_map = {
                "vits": 64,
                "vitb": 128,
                "vitl": 256,
                "vitg": 384,
            }
            self.depth_feat_dim = depth_feat_dim_map[encoder]
            self.depth_refine = DepthRefine(
                rgb_feat_dim=self.depth_feat_dim,
                obs_feat_dim=self.depth_refine_dim,
                hidden_dim=self.depth_refine_dim,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                downsample=self.stride,
            )
        else:
            self.depth_refine = None
        self.spatial_enhancer = GraspSpatialEnhancer(
            embed_dims=tok_feat_dim,
            feature_3d_dim=32,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            num_depth=self.bin_num,
            detach_depth_grad=True,      # 第一轮建议 True，避免破坏 depth_net
            use_post_norm=False,         # 第一轮建议 False，保持 path_1 分布
            vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'spatial_enhancer'),
            vis_every=self.vis_every,
            vis_rank0_only=True,
            save_vis_npz=True,
            )
        
        self.view_dirs = generate_grasp_views(self.num_view)
        # self.view_net = ViewNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.view = GeometryAwareDenseFieldViewNet(
            num_view=self.num_view,
            seed_feature_dim=self.seed_feature_dim,
            hidden_dim=self.seed_feature_dim,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            view_dirs=self.view_dirs,
            vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'geom_viewnet'),
            vis_every=self.vis_every,
            is_training=self.is_training,
        )
        # self.view_net = GeometryAwareDenseFieldAttnViewNet(
        #     num_view=self.num_view,
        #     seed_feature_dim=self.seed_feature_dim,
        #     hidden_dim=self.seed_feature_dim,
        #     min_depth=self.min_depth,
        #     max_depth=self.max_depth,
        #     bin_num=self.bin_num,
        #     view_dirs=generate_grasp_views(self.num_view),
        #     vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'geom_attn_viewnet'),
        #     vis_every=self.vis_every,
        #     num_heads=4,
        #     attn_dropout=0.01,
        #     use_depth_prob=False,
        # )
        # self.cy_group = Cylinder_Grouping_Global_Interaction(
        #     nsample=16,
        #     cylinder_radius=cylinder_radius,
        #     seed_feature_dim=self.seed_feature_dim,
        # )
        # self.local_region_group = MetricRegionCropGrouping(
        #     seed_feature_dim=self.seed_feature_dim,
        #     feat_dim=self.seed_feature_dim,
        #     out_dim=256,
        #     hidden_dim=128,
        #     patch_size=12,
        #     metric_radius=0.08,
        #     radius_px_min=8.0,
        #     radius_px_max=64.0,
        #     train_scale_min=0.80,
        #     train_scale_max=1.25,
        #     min_depth=self.min_depth,
        #     max_depth=self.max_depth,
        #     depth_norm_scale=0.08,
        #     detach_depth=True,
        #     detach_aux_maps=True,
        #     use_view_conditioned_pool=True,
        #     vis_dir=None if self.vis_dir is None else os.path.join(self.vis_dir, 'local_region_crop'),
        #     vis_every=self.vis_every,
        #     vis_num_seeds=4,
        #     vis_seed_mode='first',
        #     save_npz=True,
        # )
        
        # self.grasp_head = Grasp_Head_Local_Interaction(
        #     num_angle=self.num_angle,
        #     num_depth=self.num_depth,
        # )
        # self.grasp_head = Grasp_Head_Local_Interaction_Collision(
        #     num_angle=self.num_angle,
        #     num_depth=self.num_depth,
        # )
        # self.grasp_head = Grasp_Head_Local_Interaction_Dropout(
        #     num_angle=self.num_angle,
        #     num_depth=self.num_depth,
        # )

        # Shared K-view selector/grouping configuration.
        self.kview_config = KViewQueryTransformerConfig(
            mode=(
                "A2"
                if bool(
                    getattr(
                        cfgs,
                        "use_top4_view_infer",
                        False,
                    )
                )
                else getattr(cfgs, "kview_mode", "A1")
            ),
            num_query_views=(
                4
                if bool(
                    getattr(
                        cfgs,
                        "use_top4_view_infer",
                        False,
                    )
                )
                else getattr(cfgs, "kview_k", 1)
            ),
            sample_temperature=getattr(
                cfgs,
                "kview_tau",
                1.0,
            ),
            sample_from=getattr(
                cfgs,
                "kview_sample_from",
                "minmax_norm",
            ),
            patch_size=getattr(
                cfgs,
                "kview_patch_size",
                6,
            ),
            metric_radius=getattr(
                cfgs,
                "kview_metric_radius",
                0.08,
            ),
            radius_px_min=getattr(
                cfgs,
                "kview_radius_px_min",
                8.0,
            ),
            radius_px_max=getattr(
                cfgs,
                "kview_radius_px_max",
                64.0,
            ),
            grouping_model_dim=getattr(
                cfgs,
                "kview_group_dim",
                256,
            ),
            grouping_num_heads=getattr(
                cfgs,
                "kview_group_heads",
                4,
            ),
            grouping_dropout=getattr(
                cfgs,
                "kview_group_dropout",
                0.05,
            ),
            grouping_max_queries_per_chunk=getattr(
                cfgs,
                "kview_group_chunk",
                2048,
            ),
            use_gripper_projected_axes=True,
            head_model_dim=getattr(
                cfgs,
                "kview_head_dim",
                128,
            ),
            head_hidden_dim=getattr(
                cfgs,
                "kview_head_hidden_dim",
                64,
            ),
            head_num_layers=getattr(
                cfgs,
                "kview_head_layers",
                2,
            ),
            head_num_heads=getattr(
                cfgs,
                "kview_head_heads",
                4,
            ),
            head_attn_dropout=getattr(
                cfgs,
                "kview_attn_dropout",
                0.05,
            ),
            head_dropout_p=getattr(
                cfgs,
                "kview_head_dropout",
                0.15,
            ),
            use_collision_head=False,
            # Used only by the CDF candidate decoder.
            num_cdf_thresholds=int(
                getattr(
                    cfgs,
                    "num_cdf_thresholds",
                    6,
                )
            ),
            cdf_increment_bias=float(
                getattr(
                    cfgs,
                    "cdf_increment_bias",
                    -4.0,
                )
            ),
            vis_dir=(
                None
                if self.vis_dir is None
                else os.path.join(
                    self.vis_dir,
                    (
                        "kview_query_grasp_cdf"
                        if self.use_cdf
                        else "kview_query_grasp_legacy"
                    ),
                )
            ),
            vis_every=self.vis_every,
            vis_num_queries=int(
                getattr(
                    cfgs,
                    "cdf_vis_num_queries",
                    32,
                )
            ),
            save_npz=False,
        )

        common_kview_kwargs = dict(
            view_net=self.view,
            num_view=self.num_view,
            num_angle=self.num_angle,
            num_depth=self.num_depth,
            seed_feature_dim=self.seed_feature_dim,
            feat_dim=self.seed_feature_dim,
            view_dirs=self.view_dirs,
            batch_viewpoint_params_to_matrix_fn=(
                batch_viewpoint_params_to_matrix
            ),
            config=self.kview_config,
        )

        # Both variants are the same Center-View-Angle Transformer. Only
        # the final candidate decoder/supervision changes with use_cdf.
        self.kview_grasp_module = (
            CenterViewAngleQueryTransformerLocalGraspModule(
                **common_kview_kwargs,
                use_cdf=self.use_cdf,
            )
        )

        # Geometry diagnostics are independent of the prediction-head choice
        # and are enabled automatically whenever visualization is enabled.
        # RGBGeometryDiagnostics supports both CDF and legacy outputs.
        if self.use_geometry_diagnostics:
            self.rgb_geometry_diagnostics = RGBGeometryDiagnostics(
                num_angle=self.num_angle,
                num_depth=self.num_depth,
                batch_viewpoint_params_to_matrix_fn=(
                    batch_viewpoint_params_to_matrix
                ),
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                patch_size=int(
                    getattr(
                        cfgs,
                        "geometry_diag_patch_size",
                        5,
                    )
                ),
                metric_radius=float(
                    getattr(
                        cfgs,
                        "geometry_diag_metric_radius",
                        0.04,
                    )
                ),
                radius_px_min=float(
                    getattr(
                        cfgs,
                        "geometry_diag_radius_px_min",
                        4.0,
                    )
                ),
                radius_px_max=float(
                    getattr(
                        cfgs,
                        "geometry_diag_radius_px_max",
                        32.0,
                    )
                ),
                topk=int(
                    getattr(
                        cfgs,
                        "geometry_diag_topk",
                        50,
                    )
                ),
                high_center_error_m=float(
                    getattr(
                        cfgs,
                        "geometry_diag_high_center_error",
                        0.02,
                    )
                ),
                high_patch_error_m=float(
                    getattr(
                        cfgs,
                        "geometry_diag_high_patch_error",
                        0.02,
                    )
                ),
                vis_dir=(
                    None
                    if self.vis_dir is None
                    else os.path.join(
                        self.vis_dir,
                        "rgb_geometry",
                    )
                ),
                vis_every=self.vis_every,
                vis_num_queries=int(
                    getattr(
                        cfgs,
                        "geometry_diag_vis_queries",
                        256,
                    )
                ),
                vis_num_cases=int(
                    getattr(
                        cfgs,
                        "geometry_diag_vis_cases",
                        4,
                    )
                ),
                save_npz=True,
            )
        else:
            self.rgb_geometry_diagnostics = None

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            print(
                "[economicgrasp_dpt] CVA head="
                + ("CDF" if self.use_cdf else "legacy explicit-angle")
                + ", geometry_diagnostics="
                + str(int(self.use_geometry_diagnostics))
                + ", obs_depth="
                + str(int(self.use_obs_depth))
                + ", pose_depth_mode="
                + self.pose_depth_mode
                + ", geometry_depth_source="
                + self.geometry_depth_source,
                flush=True,
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.geometry_depth_source == "gt":
            # The entire depth module is frozen in the privileged teacher. Keep
            # its DINO feature extractor in eval mode as well, so Stage-0 RGB
            # proposal features are deterministic and match teacher inference.
            self.depth_net.eval()
        return self

    def _prepare_gt_geometry_depth(
        self,
        end_points: dict,
        *,
        image_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalize the privileged clean depth to ``[B,1,H,W]`` meters.

        ``GraspNetMultiDataset`` always provides ``gt_depth_m`` independently of
        its legacy ``use_gt_depth`` switch.  This lets the teacher use clean
        geometry while teacher and student retain identical RGB crops, token
        labels, and image-FPS coordinates.  Invalid/unknown pixels remain zero
        and are removed later by the common depth-valid mask.
        """
        depth = end_points.get("gt_depth_m", None)
        if depth is None:
            raise KeyError(
                "geometry_depth_source='gt' requires end_points['gt_depth_m']."
            )
        if not torch.is_tensor(depth):
            raise TypeError(
                "end_points['gt_depth_m'] must be a tensor, got "
                f"{type(depth)}."
            )
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4:
            depth = depth[:, :1]
        else:
            raise ValueError(
                "gt_depth_m must be [B,H,W] or [B,1,H,W], got "
                f"{tuple(depth.shape)}."
            )
        depth = depth.to(device=device, dtype=dtype)
        if tuple(depth.shape[-2:]) != tuple(image_hw):
            depth = F.interpolate(depth, size=image_hw, mode="nearest")
        return torch.nan_to_num(
            depth, nan=0.0, posinf=0.0, neginf=0.0
        ).contiguous()


    @staticmethod
    def _deterministic_fill_indices(
        selected: torch.Tensor,
        fallback: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        """Return exactly ``target_count`` indices without stochastic sampling."""
        target_count = int(target_count)
        if target_count <= 0:
            raise ValueError(f"target_count must be positive, got {target_count}.")
        if selected.numel() >= target_count:
            return selected[:target_count].contiguous()
        if selected.numel() == 0:
            selected = fallback
        if selected.numel() == 0:
            raise RuntimeError("No valid image token is available for seed selection.")
        repeat = (target_count + selected.numel() - 1) // selected.numel()
        return selected.repeat(repeat)[:target_count].contiguous()

    @staticmethod
    def _image_uv_coordinates(
        token_idx: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return normalized image coordinates [K,2] for flat token indices."""
        if token_idx.dim() != 1:
            raise ValueError(
                f"token_idx must be one-dimensional, got {tuple(token_idx.shape)}"
            )
        u = (token_idx % int(width)).to(dtype=dtype)
        v = (token_idx // int(width)).to(dtype=dtype)
        u = u / max(float(width - 1), 1.0)
        v = v / max(float(height - 1), 1.0)
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _deterministic_fps_2d(
        coordinates: torch.Tensor,
        scores: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Deterministic 2D farthest-point sampling.

        The candidate list is reordered so its highest-graspness token is the
        first FPS seed. On CUDA, the repository's PointNet2 FPS kernel is reused
        on ``[u, v, 0]`` image coordinates for efficiency; no depth or 3D scene
        coordinate enters the distance. A pure PyTorch fallback is retained for
        CPU unit tests.
        """
        if coordinates.dim() != 2 or coordinates.shape[-1] != 2:
            raise ValueError(
                "coordinates must be [K,2], got "
                f"{tuple(coordinates.shape)}"
            )
        if scores.dim() != 1 or scores.shape[0] != coordinates.shape[0]:
            raise ValueError(
                "scores must be [K] and aligned with coordinates; got "
                f"scores={tuple(scores.shape)}, coords={tuple(coordinates.shape)}"
            )
        K = int(coordinates.shape[0])
        count = min(max(int(num_samples), 0), K)
        if count == 0:
            return torch.empty(0, device=coordinates.device, dtype=torch.long)

        coords = coordinates.float()
        score_clean = torch.nan_to_num(
            scores.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        first = torch.argmax(score_clean)
        all_local = torch.arange(K, device=coords.device, dtype=torch.long)
        order = torch.cat(
            [first.view(1), all_local[all_local != first]], dim=0
        )
        coords_ordered = coords[order]

        if coords_ordered.is_cuda:
            coords_3d = torch.cat(
                [
                    coords_ordered,
                    torch.zeros(
                        K,
                        1,
                        device=coords.device,
                        dtype=coords_ordered.dtype,
                    ),
                ],
                dim=-1,
            ).unsqueeze(0).contiguous()
            selected_ordered = furthest_point_sample(
                coords_3d, count
            ).squeeze(0).long()
            return order[selected_ordered].contiguous()

        selected = torch.empty(count, device=coords.device, dtype=torch.long)
        selected_mask = torch.zeros(K, device=coords.device, dtype=torch.bool)
        min_dist = torch.full(
            (K,), float("inf"), device=coords.device, dtype=coords.dtype
        )
        farthest = first
        for sample_i in range(count):
            selected[sample_i] = farthest
            selected_mask[farthest] = True
            center = coords[farthest].view(1, 2)
            dist = ((coords - center) ** 2).sum(dim=-1)
            min_dist = torch.minimum(min_dist, dist)
            min_dist = min_dist.masked_fill(selected_mask, -1.0)
            farthest = torch.argmax(min_dist)
        return selected.contiguous()

    def _select_image_fps_indices(
        self,
        graspable_mask: torch.Tensor,
        valid_tok: torch.Tensor,
        grasp_score: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Select image queries with deterministic 2D FPS.

        FPS runs over the predicted graspable foreground. If fewer than M
        graspable tokens exist, all of them are kept and the remaining slots are
        filled by image-FPS over other valid tokens. Only when fewer than M valid
        pixels exist are indices repeated deterministically.
        """
        B, N = grasp_score.shape
        if N != int(height) * int(width):
            raise ValueError(
                f"grasp_score has N={N}, expected H*W={height * width}."
            )
        if graspable_mask.shape != (B, N) or valid_tok.shape != (B, N):
            raise ValueError(
                "graspable_mask, valid_tok and grasp_score must share [B,H*W]."
            )

        score = torch.nan_to_num(
            grasp_score.detach(), nan=0.0, posinf=0.0, neginf=0.0
        )
        outputs = []
        all_idx = torch.arange(N, device=score.device, dtype=torch.long)

        for batch_i in range(B):
            primary = graspable_mask[batch_i] & valid_tok[batch_i]
            fallback = valid_tok[batch_i]
            if not bool(fallback.any()):
                fallback = torch.ones_like(fallback)
            primary_idx = torch.nonzero(primary, as_tuple=False).squeeze(1)
            fallback_idx = torch.nonzero(fallback, as_tuple=False).squeeze(1)

            selected_parts = []
            if primary_idx.numel() > 0:
                primary_uv = self._image_uv_coordinates(
                    primary_idx, height, width, score.dtype
                )
                primary_local = self._deterministic_fps_2d(
                    primary_uv,
                    score[batch_i, primary_idx],
                    min(self.M_points, int(primary_idx.numel())),
                )
                selected_parts.append(primary_idx[primary_local])

            selected = (
                torch.cat(selected_parts, dim=0)
                if selected_parts
                else torch.empty(0, device=score.device, dtype=torch.long)
            )

            if selected.numel() < self.M_points:
                used = torch.zeros(N, device=score.device, dtype=torch.bool)
                if selected.numel() > 0:
                    used[selected] = True
                remaining_idx = fallback_idx[~used[fallback_idx]]
                if remaining_idx.numel() > 0:
                    remaining_uv = self._image_uv_coordinates(
                        remaining_idx, height, width, score.dtype
                    )
                    remaining_local = self._deterministic_fps_2d(
                        remaining_uv,
                        score[batch_i, remaining_idx],
                        min(
                            self.M_points - int(selected.numel()),
                            int(remaining_idx.numel()),
                        ),
                    )
                    selected = torch.cat(
                        [selected, remaining_idx[remaining_local]], dim=0
                    )

            selected = self._deterministic_fill_indices(
                selected,
                fallback_idx if fallback_idx.numel() > 0 else all_idx,
                self.M_points,
            )
            outputs.append(selected)

        return torch.stack(outputs, dim=0).contiguous()

    def _validate_image_fps_override(
        self,
        override: torch.Tensor,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Validate exact teacher image-FPS indices reused by the student."""
        if not torch.is_tensor(override):
            raise TypeError(
                "image_fps_seed_idx_override must be a tensor [B,M]."
            )
        override = override.to(device=device, dtype=torch.long)
        expected = (int(batch_size), int(self.M_points))
        if tuple(override.shape) != expected:
            raise ValueError(
                "image_fps_seed_idx_override must have shape "
                f"{expected}, got {tuple(override.shape)}."
            )
        if bool(((override < 0) | (override >= int(num_tokens))).any()):
            raise ValueError(
                "image_fps_seed_idx_override contains an out-of-range token index."
            )
        return override.contiguous()

    def _select_graspable_seed_queries(
        self,
        feat_grid: torch.Tensor,
        depth_map: torch.Tensor,
        camera_K: torch.Tensor,
        graspable_mask: torch.Tensor,
        valid_tok: torch.Tensor,
        grasp_score: torch.Tensor,
        end_points: dict,
    ):
        """Build the sparse grasp-query set for either teacher or student.

        ``image_fps`` is the shared teacher/student path. It performs FPS only
        in normalized image coordinates and then backprojects the selected M
        depths. Stage 2 may provide ``image_fps_seed_idx_override`` so the
        student reuses the teacher's exact image-FPS indices. ``point_fps`` is
        retained only for the original non-distillation baseline.
        """
        B, C, H, W = feat_grid.shape
        N = H * W
        M = self.M_points
        feat_flat = feat_grid.view(B, C, N).contiguous()
        graspable_num_batch = float(graspable_mask.float().sum().item())

        use_gt_xyz = (
            self.is_training
            and self.use_gt_xyz_for_train
            and ("gt_depth_m" in end_points)
        )
        depth_for_xyz = depth_map
        if use_gt_xyz:
            depth_for_xyz = end_points["gt_depth_m"]
            if depth_for_xyz.dim() == 3:
                depth_for_xyz = depth_for_xyz.unsqueeze(1)
            elif depth_for_xyz.dim() == 4:
                depth_for_xyz = depth_for_xyz[:, :1]
            if depth_for_xyz.shape[-2:] != (H, W):
                depth_for_xyz = F.interpolate(
                    depth_for_xyz,
                    size=(H, W),
                    mode="nearest",
                )
            depth_for_xyz = depth_for_xyz.to(
                device=feat_grid.device,
                dtype=depth_map.dtype,
            )

        if self.seed_selection_mode == "image_fps":
            override = end_points.get("image_fps_seed_idx_override", None)
            if override is not None:
                if self.seed_selection_mode != "image_fps":
                    raise RuntimeError(
                        "image_fps_seed_idx_override is valid only when "
                        "seed_selection_mode='image_fps'."
                    )
                token_sel_idx = self._validate_image_fps_override(
                    override, B, N, feat_grid.device
                )
                shared_override = True
            else:
                token_sel_idx = self._select_image_fps_indices(
                    graspable_mask=graspable_mask,
                    valid_tok=valid_tok,
                    grasp_score=grasp_score,
                    height=H,
                    width=W,
                )
                shared_override = False

            gather_idx = token_sel_idx.unsqueeze(1).expand(-1, C, -1)
            seed_features = torch.gather(feat_flat, 2, gather_idx).contiguous()

            z_flat = depth_for_xyz[:, 0].reshape(B, N)
            z_seed = torch.gather(z_flat, 1, token_sel_idx).unsqueeze(-1)
            z_seed = torch.nan_to_num(
                z_seed,
                nan=self.min_depth,
                posinf=self.max_depth,
                neginf=self.min_depth,
            ).clamp(self.min_depth, self.max_depth)
            u = (token_sel_idx % W).to(dtype=z_seed.dtype)
            v = (token_sel_idx // W).to(dtype=z_seed.dtype)
            uv = torch.stack([u, v], dim=-1)
            seed_xyz = self._backproject_uvz(
                uv,
                z_seed if use_gt_xyz else z_seed.detach(),
                camera_K,
            )

            end_points["D: Image-FPS enabled"] = depth_map.new_tensor(
                float(self.seed_selection_mode == "image_fps")
            ).reshape(())
            end_points["D: Shared image-FPS seeds"] = depth_map.new_tensor(
                float(shared_override)
            ).reshape(())
            if shared_override:
                shared_valid = torch.gather(valid_tok, 1, token_sel_idx)
                end_points["D: Shared image-FPS valid ratio"] = (
                    shared_valid.float().mean().reshape(())
                )

            return (
                seed_features,
                seed_xyz.contiguous(),
                token_sel_idx.contiguous(),
                None,
                None,
                graspable_num_batch,
            )

        flat_all = torch.arange(
            N,
            device=feat_grid.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1).contiguous()
        u_all = (flat_all % W).float()
        v_all = (flat_all // W).float()
        uv_all = torch.stack([u_all, v_all], dim=-1)

        z_all_pred = depth_map.view(B, -1, 1).contiguous()
        z_all_pred = torch.nan_to_num(
            z_all_pred,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(1e-6)
        xyz_all_pred = self._backproject_uvz(
            uv_all,
            z_all_pred.detach(),
            camera_K,
        )
        if use_gt_xyz:
            z_all_match = depth_for_xyz.view(B, -1, 1).contiguous()
            z_all_match = torch.nan_to_num(
                z_all_match,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(1e-6)
            xyz_all_match = self._backproject_uvz(
                uv_all,
                z_all_match,
                camera_K,
            )
        else:
            xyz_all_match = xyz_all_pred

        seed_features = []
        seed_xyz = []
        token_sel_idx = []
        for batch_i in range(B):
            cur_idx = torch.nonzero(
                graspable_mask[batch_i], as_tuple=False
            ).squeeze(1)
            if cur_idx.numel() == 0:
                cur_idx = torch.nonzero(
                    valid_tok[batch_i], as_tuple=False
                ).squeeze(1)
            if cur_idx.numel() == 0:
                cur_idx = torch.arange(N, device=feat_grid.device)

            cur_feat = feat_flat[batch_i][:, cur_idx]
            cur_xyz = xyz_all_match[batch_i][cur_idx]
            if cur_xyz.shape[0] >= M:
                fps_idx = furthest_point_sample(
                    cur_xyz.unsqueeze(0).contiguous(), M
                )
                cur_xyz = gather_operation(
                    cur_xyz.unsqueeze(0).transpose(1, 2).contiguous(),
                    fps_idx,
                ).transpose(1, 2).squeeze(0).contiguous()
                cur_feat = gather_operation(
                    cur_feat.unsqueeze(0).contiguous(), fps_idx
                ).squeeze(0).contiguous()
                cur_sel = cur_idx[fps_idx.squeeze(0).long()]
            else:
                rep = torch.randint(
                    0,
                    cur_xyz.shape[0],
                    (M,),
                    device=feat_grid.device,
                )
                cur_xyz = cur_xyz[rep]
                cur_feat = cur_feat[:, rep]
                cur_sel = cur_idx[rep]

            seed_features.append(cur_feat)
            seed_xyz.append(cur_xyz)
            token_sel_idx.append(cur_sel)

        return (
            torch.stack(seed_features, dim=0),
            torch.stack(seed_xyz, dim=0),
            torch.stack(token_sel_idx, dim=0),
            xyz_all_pred,
            uv_all,
            graspable_num_batch,
        )

    def _assert_cva_output_contract(
        self,
        end_points: dict,
    ) -> None:
        """Fail immediately when CDF and legacy endpoints are mixed."""
        if self.use_cdf:
            required = (
                "grasp_cdf_pred_angle_depth",
                "grasp_width_pred_angle_depth",
            )
            forbidden = (
                "grasp_depth_pred_angle",
                "grasp_score_pred_angle",
                "grasp_width_pred_angle",
                "grasp_angle_pred",
                "grasp_depth_pred",
                "grasp_score_pred",
                "grasp_width_pred",
            )
        else:
            required = (
                "grasp_depth_pred_angle",
                "grasp_score_pred_angle",
                "grasp_width_pred_angle",
                "grasp_angle_pred",
                "grasp_depth_pred",
                "grasp_score_pred",
                "grasp_width_pred",
            )
            forbidden = (
                "grasp_cdf_pred_angle_depth",
                "grasp_width_pred_angle_depth",
            )

        missing = [
            key for key in required
            if key not in end_points
        ]
        if missing:
            raise KeyError(
                f"CVA head={'CDF' if self.use_cdf else 'legacy'} "
                f"is missing endpoint(s): {missing}"
            )

        incompatible = [
            key for key in forbidden
            if key in end_points
        ]
        if incompatible:
            raise RuntimeError(
                f"CVA head={'CDF' if self.use_cdf else 'legacy'} "
                f"received incompatible endpoint(s): {incompatible}. "
                "Check model construction, labels, loss and decoder."
            )

    @staticmethod
    def _backproject_uvz(uv_b_n2, z_b_n1, K_b_33):
        fx = K_b_33[:, 0, 0].unsqueeze(1)
        fy = K_b_33[:, 1, 1].unsqueeze(1)
        cx = K_b_33[:, 0, 2].unsqueeze(1)
        cy = K_b_33[:, 1, 2].unsqueeze(1)
        u = uv_b_n2[..., 0]
        v = uv_b_n2[..., 1]
        z = z_b_n1.squeeze(-1)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)

    def _save_map_png(self, arr2d, out_path, vmin=None, vmax=None, cmap='Spectral', title=None):
        if torch.is_tensor(arr2d):
            arr2d = arr2d.detach().float().cpu().numpy()
        plt.figure(figsize=(6, 6))
        if vmin is None:
            vmin = float(np.nanmin(arr2d))
        if vmax is None:
            vmax = float(np.nanmax(arr2d))
        plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_overlay_points(self, img_448, pts_uv, out_path, radius=1, color=(0, 0, 255)):
        import cv2
        x = img_448.detach().float().cpu()
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        x_bgr = x[..., ::-1].copy()

        pts = pts_uv.detach().cpu().numpy()
        H, W = x_bgr.shape[:2]
        for (u, v) in pts:
            uu = int(round(float(u)))
            vv = int(round(float(v)))
            if 0 <= uu < W and 0 <= vv < H:
                cv2.circle(x_bgr, (uu, vv), radius, color, thickness=-1)
        cv2.imwrite(out_path, x_bgr)

    @torch.no_grad()
    def _save_pred_gt_cloud_ply(
        self,
        cloud_pred: torch.Tensor,
        cloud_gt: torch.Tensor,
        end_points: dict,
    ):
        if self.vis_dir is None:
            return

        # ------------------------------------------------------------
        # Avoid duplicated visualization under DDP / multi-process.
        # Only rank0 writes point clouds.
        # ------------------------------------------------------------
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        def _valid(x: np.ndarray):
            m = np.isfinite(x).all(axis=1)
            m &= (x[:, 2] > self.min_depth)
            m &= (x[:, 2] < self.max_depth)
            return x[m]

        def _make_color(n: int, color):
            c = np.zeros((n, 3), dtype=np.float32)
            c[:, 0] = float(color[0])
            c[:, 1] = float(color[1])
            c[:, 2] = float(color[2])
            return c

        def _write_ply(items, out_path: str):
            """
            items: list of (points_np, color_tuple)
            """
            pts_list = []
            col_list = []

            for pts_np, color in items:
                if pts_np is None:
                    continue

                pts_np = _valid(pts_np)
                if pts_np.shape[0] == 0:
                    continue

                pts_list.append(pts_np.astype(np.float32))
                col_list.append(_make_color(pts_np.shape[0], color))

            if len(pts_list) == 0:
                return False

            pts = np.concatenate(pts_list, axis=0)
            cols = np.concatenate(col_list, axis=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

            o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
            return True

        # ------------------------------------------------------------
        # Use batch item 0 only.
        # ------------------------------------------------------------
        p = cloud_pred[0].detach().float().cpu().numpy()
        g = cloud_gt[0].detach().float().cpu().numpy()

        scene = int(end_points.get('scene_idx', -1)[0].item()) \
            if torch.is_tensor(end_points.get('scene_idx', None)) \
            else int(end_points.get('scene_idx', -1))

        anno = int(end_points.get('anno_idx', -1)[0].item()) \
            if torch.is_tensor(end_points.get('anno_idx', None)) \
            else int(end_points.get('anno_idx', -1))

        # ------------------------------------------------------------
        # Case 1: RGB mode, save pred + gt only.
        #   red  = predicted final depth cloud
        #   blue = GT depth cloud
        # ------------------------------------------------------------
        if not self.use_obs_depth:
            out_path = os.path.join(
                self.vis_dir,
                f'dpt_pred_gt_xyz_scene{scene:04d}_anno{anno:04d}_it{self._vis_iter:06d}.ply'
            )

            _write_ply(
                [
                    (p, (1.0, 0.0, 0.0)),  # pred: red
                    (g, (0.0, 0.0, 1.0)),  # gt: blue
                ],
                out_path,
            )
            return

        # ------------------------------------------------------------
        # Case 2: RGB-D mode, save pred + gt + obs in one PLY only.
        #   red   = predicted final depth cloud
        #   blue  = GT depth cloud
        #   green = observed depth cloud
        # ------------------------------------------------------------
        if "sensor_depth_m" not in end_points:
            # In principle should not happen when self.use_obs_depth=True.
            out_path = os.path.join(
                self.vis_dir,
                f'dpt_pred_gt_xyz_scene{scene:04d}_anno{anno:04d}_it{self._vis_iter:06d}.ply'
            )
            _write_ply(
                [
                    (p, (1.0, 0.0, 0.0)),
                    (g, (0.0, 0.0, 1.0)),
                ],
                out_path,
            )
            return

        obs_depth = end_points["sensor_depth_m"]

        if obs_depth.dim() == 3:
            obs_depth = obs_depth.unsqueeze(1)
        elif obs_depth.dim() == 4:
            obs_depth = obs_depth[:, :1]
        else:
            return

        K = end_points["K"]
        device = obs_depth.device

        obs_depth = obs_depth.to(device=device, dtype=K.dtype)
        K = K.to(device=device, dtype=obs_depth.dtype)

        # Use model input resolution when available.
        if "img" in end_points and torch.is_tensor(end_points["img"]):
            H_img, W_img = end_points["img"].shape[-2:]
            if obs_depth.shape[-2:] != (H_img, W_img):
                obs_depth = F.interpolate(
                    obs_depth,
                    size=(H_img, W_img),
                    mode="nearest",
                )
        else:
            H_img, W_img = obs_depth.shape[-2:]

        B, _, Hobs, Wobs = obs_depth.shape

        flat_all = torch.arange(
            Hobs * Wobs,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1).contiguous()

        u_all = (flat_all % Wobs).float()
        v_all = (flat_all // Wobs).float()
        uv_all = torch.stack([u_all, v_all], dim=-1)

        z_obs = obs_depth.view(B, -1, 1).contiguous()
        z_obs = torch.nan_to_num(
            z_obs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(1e-6)

        xyz_obs = self._backproject_uvz(uv_all, z_obs, K)
        o = xyz_obs[0].detach().float().cpu().numpy()

        out_path = os.path.join(
            self.vis_dir,
            f'dpt_pred_gt_xyz_scene{scene:04d}_anno{anno:04d}_it{self._vis_iter:06d}.ply'
        )

        _write_ply(
            [
                (p, (1.0, 0.0, 0.0)),  # pred: red
                (g, (0.0, 0.0, 1.0)),  # gt: blue
                (o, (0.0, 1.0, 0.0)),  # obs: green
            ],
            out_path,
        )

    @torch.no_grad()
    def _add_topview_quality_logs(self, end_points: dict):
        """
        Diagnose whether dense view regression improves the actual argmax top-view.

        Required:
        - view_score: (B,M,V)
        - grasp_top_view_inds: (B,M)
        - batch_grasp_view_graspness: (B,M,V)

        Logs:
        - GT score of predicted top view
        - oracle top-view GT score
        - regret = oracle - predicted
        - top-k agreement with GT view field
        - angular error between predicted top view and oracle top view
        - predicted top1-top2 angular distance
        """
        if not isinstance(end_points, dict):
            return end_points

        required = [
            "view_score",
            "grasp_top_view_inds",
            "batch_grasp_view_graspness",
        ]
        for k in required:
            if k not in end_points:
                return end_points

        view_score = end_points["view_score"].detach()
        view_label = end_points["batch_grasp_view_graspness"].detach()
        top_idx = end_points["grasp_top_view_inds"].detach().long()

        if view_score.dim() != 3 or view_label.dim() != 3 or top_idx.dim() != 2:
            return end_points

        # Expected: (B,M,V). If someone accidentally returns (B,V,M), fix if unambiguous.
        if view_score.shape != view_label.shape:
            if view_score.transpose(1, 2).shape == view_label.shape:
                view_score = view_score.transpose(1, 2).contiguous()
            else:
                return end_points

        B, M, V = view_label.shape
        if top_idx.shape != (B, M):
            return end_points

        device = view_label.device
        top_idx = top_idx.clamp(0, V - 1)

        # ------------------------------------------------------------
        # 1) GT score of predicted top-view vs. oracle top-view
        # ------------------------------------------------------------
        pred_top_gt = torch.gather(
            view_label,
            dim=2,
            index=top_idx.unsqueeze(-1),
        ).squeeze(-1)  # (B,M)

        oracle_top_gt, oracle_idx = view_label.max(dim=-1)  # (B,M), (B,M)

        finite_mask = (
            torch.isfinite(pred_top_gt)
            & torch.isfinite(oracle_top_gt)
            & torch.isfinite(view_label).all(dim=-1)
        )

        # If a selected seed has all-zero view labels, it is not informative for top-view quality.
        label_valid = finite_mask & (oracle_top_gt > 1e-6)

        # Use valid labels if available; otherwise fall back to finite mask to avoid empty logs.
        stat_mask = label_valid
        if not bool(stat_mask.any()):
            stat_mask = finite_mask

        def masked_mean(x: torch.Tensor):
            if bool(stat_mask.any()):
                return x[stat_mask].float().mean()
            return x.new_tensor(0.0).float()

        def masked_ratio(cond: torch.Tensor):
            if bool(stat_mask.any()):
                return cond[stat_mask].float().mean()
            return cond.new_tensor(0.0).float()

        regret = (oracle_top_gt - pred_top_gt).clamp_min(0.0)

        end_points["D: TopView LabelValid"] = label_valid.float().mean().reshape(())
        end_points["D: TopView PredGT"] = masked_mean(pred_top_gt).reshape(())
        end_points["D: TopView OracleGT"] = masked_mean(oracle_top_gt).reshape(())
        end_points["D: TopView Regret"] = masked_mean(regret).reshape(())

        end_points["D: TopView PredGT>0.1"] = masked_ratio(pred_top_gt > 0.1).reshape(())
        end_points["D: TopView PredGT>0.3"] = masked_ratio(pred_top_gt > 0.3).reshape(())
        end_points["D: TopView PredGT>0.5"] = masked_ratio(pred_top_gt > 0.5).reshape(())

        # ------------------------------------------------------------
        # 2) Whether predicted top-view is among GT top-k modes
        # ------------------------------------------------------------
        for k in (1, 5, 10, 20):
            kk = min(k, V)
            gt_topk_idx = torch.topk(view_label, k=kk, dim=-1).indices  # (B,M,kk)
            hit = (gt_topk_idx == top_idx.unsqueeze(-1)).any(dim=-1)    # (B,M)
            end_points[f"D: TopView InGTTop{k}"] = masked_ratio(hit).reshape(())

        # ------------------------------------------------------------
        # 3) Predicted-score diagnostics
        # ------------------------------------------------------------
        pred_top_score = torch.gather(
            view_score,
            dim=2,
            index=top_idx.unsqueeze(-1),
        ).squeeze(-1)

        top2_vals, top2_idx = torch.topk(view_score, k=min(2, V), dim=-1)
        pred_margin = top2_vals[..., 0] - top2_vals[..., 1] if V >= 2 else torch.zeros_like(pred_top_score)

        end_points["D: TopView PredScore"] = masked_mean(pred_top_score).reshape(())
        end_points["D: TopView PredMargin"] = masked_mean(pred_margin).reshape(())

        # ------------------------------------------------------------
        # 4) Angular diagnostics on view anchors
        # ------------------------------------------------------------
        if hasattr(self.view, "view_dirs"):
            view_dirs = self.view.view_dirs.detach().to(device=device, dtype=torch.float32)
        else:
            view_dirs = generate_grasp_views(V).to(device=device, dtype=torch.float32)

        view_dirs = F.normalize(view_dirs, dim=-1)

        pred_dir = view_dirs.index_select(0, top_idx.reshape(-1)).view(B, M, 3)
        oracle_dir = view_dirs.index_select(0, oracle_idx.reshape(-1)).view(B, M, 3)

        cos_po = (pred_dir * oracle_dir).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        ang_po = torch.rad2deg(torch.acos(cos_po))  # (B,M)

        end_points["D: TopView AngErr"] = masked_mean(ang_po).reshape(())
        end_points["D: TopView Ang<5"] = masked_ratio(ang_po < 5.0).reshape(())
        end_points["D: TopView Ang<10"] = masked_ratio(ang_po < 10.0).reshape(())
        end_points["D: TopView Ang<15"] = masked_ratio(ang_po < 15.0).reshape(())
        end_points["D: TopView Ang<30"] = masked_ratio(ang_po < 30.0).reshape(())

        if V >= 2:
            top1_idx = top2_idx[..., 0].reshape(-1)
            top2_idx_flat = top2_idx[..., 1].reshape(-1)

            top1_dir = view_dirs.index_select(0, top1_idx).view(B, M, 3)
            top2_dir = view_dirs.index_select(0, top2_idx_flat).view(B, M, 3)

            cos_12 = (top1_dir * top2_dir).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            ang_12 = torch.rad2deg(torch.acos(cos_12))

            end_points["D: TopView Top1Top2Ang"] = masked_mean(ang_12).reshape(())

        return end_points

    def forward(self, end_points: dict):
        img = end_points['img']
        K = end_points['K']
        B, _, H, W = img.shape
        assert (H, W) == (448, 448)
        Ntok = H * W
        M = self.M_points

        # depth_448, depth_tok, _, depth_prob_448, depth_logits_448, depth_prob_pred, feats = self.depth_net(
        #     img,
        #     return_prob=True,
        #     return_tok_prob=True,
        #     return_feats=True,
        # )

        camera_pose_vec = None
        camera_gravity_vec = None
        depth_net_pred_448 = None
        depth_img_feat = None
        depth_head_raw_448 = None
        pose_depth_aux = {}

        if self.geometry_depth_source == "gt":
            # Privileged Stage-0/teacher path: retain the same frozen DINO RGB
            # features for proposal prediction, but do not execute the DPT metric
            # depth decoder. All downstream geometry consumes clean synthetic
            # ``gt_depth_m`` from the data loader.
            feats = self.depth_net.extract_backbone_features(img)
            depth_448 = self._prepare_gt_geometry_depth(
                end_points,
                image_hw=(H, W),
                device=img.device,
                dtype=img.dtype,
            )
        else:
            if self.pose_depth_mode == "global_film":
                if self.camera_pose_key not in end_points:
                    raise KeyError(
                        "pose_depth_mode='global_film' requires "
                        f"end_points['{self.camera_pose_key}'] with shape (B,3)."
                    )
                camera_pose_vec = end_points[self.camera_pose_key]
            elif self.pose_depth_mode == "ray_gravity_film":
                if self.camera_gravity_key not in end_points:
                    raise KeyError(
                        "pose_depth_mode='ray_gravity_film' requires "
                        f"end_points['{self.camera_gravity_key}'] with shape (B,3)."
                    )
                camera_gravity_vec = end_points[self.camera_gravity_key]

            (
                depth_net_pred_448,
                _,
                depth_img_feat,
                depth_head_raw_448,
                feats,
                pose_depth_aux,
            ) = self.depth_net(
                img,
                camera_pose_vec=camera_pose_vec,
                camera_gravity_vec=camera_gravity_vec,
                camera_K=K,
                return_feats=True,
                return_raw=True,
                return_pose_aux=True,
            )

        obs_depth_448 = None
        depth_confidence_448 = None

        if self.geometry_depth_source == "gt":
            # ``depth_448`` is already the privileged geometry input.
            pass
        elif not self.use_obs_depth:
            # RGB-only student: RGB -> predicted absolute metric depth.
            depth_448 = depth_net_pred_448
        else:
            # RGB-D compatibility path, not used in the Stage-0--2 experiment.
            obs_depth_448 = end_points.get("sensor_depth_m", None)
            if obs_depth_448 is None:
                raise ValueError("use_obs_depth=True requires end_points['sensor_depth_m'].")

            if obs_depth_448.dim() == 3:
                obs_depth_448 = obs_depth_448.unsqueeze(1)
            elif obs_depth_448.dim() == 4:
                obs_depth_448 = obs_depth_448[:, :1]
            else:
                raise ValueError(f"Unexpected sensor_depth_m shape: {obs_depth_448.shape}")

            obs_depth_448 = obs_depth_448.to(device=img.device, dtype=depth_net_pred_448.dtype)

            if obs_depth_448.shape[-2:] != (H, W):
                obs_depth_448 = F.interpolate(
                    obs_depth_448,
                    size=(H, W),
                    mode="nearest",
                )

            depth_448, fusion_aux = self.depth_refine(
                rgb_feat=depth_img_feat,
                net_depth=depth_head_raw_448,
                obs_depth=obs_depth_448,
            )

            depth_confidence_448 = fusion_aux["depth_confidence"]
            depth_refined_correction_448 = depth_448 - obs_depth_448             # For debug

            # depth_448 = torch.nan_to_num(
            #     depth_448_raw,
            #     nan=self.min_depth,
            #     posinf=self.max_depth,
            #     neginf=self.min_depth,
            # )

        # This diagnostic tensor must be created only after the active geometry
        # depth has been selected.  In the RGB-only branch ``depth_448`` does not
        # exist until the DPT prediction is assigned above.
        if not self.use_obs_depth:
            depth_refined_correction_448 = torch.zeros_like(depth_448)

        if self.stride > 1:
            depth_tok = F.interpolate(
                depth_448,
                size=(H // self.stride, W // self.stride),
                mode="nearest",
            )
        else:
            depth_tok = depth_448
    
        patch_h, patch_w = H // 14, W // 14
        proposal_path1, proposal_logits_448 = self.proposal_head(feats, patch_h, patch_w)

        # ------------------------------------------------------------------
        # Grasp Spatial Enhancer
        # ------------------------------------------------------------------
        # proposal_path1_enh, spatial_aux = self.spatial_enhancer(
        #     feat_2d=proposal_path1,       # (B,C,Hf,Wf)
        #     depth_prob=depth_prob_448,    # (B,D,448,448)
        #     K=K,                          # K must match resized/cropped 448x448 image
        #     image_hw=(H, W),              # usually (448,448)
        #     return_maps=False,
        #     img=end_points.get("img", img) if isinstance(end_points, dict) else img,
        #     vis_prefix=None,
        # )

        proposal_path1_enh, spatial_aux = self.spatial_enhancer(
            feat_2d=proposal_path1,
            depth_prob=None,
            depth_map=depth_448,     # final depth: RGB direct or observed + residual
            K=K,
            image_hw=(H, W),
            return_maps=False,
            img=end_points.get("img", img),
        )

        for k, v in spatial_aux.items():
            end_points[k] = v

        feat_grid = F.interpolate(proposal_path1_enh, size=(H, W), mode='bilinear', align_corners=False)

        objectness_logits_448 = proposal_logits_448[:, :2, :, :]
        graspness_logits_448 = proposal_logits_448[:, 2:3, :, :]

        end_points['img_feat_dpt'] = feat_grid
        # ``depth_map_pred`` is retained as the historical geometry/loss key.
        # For the privileged teacher it is the clean GT input, not a prediction.
        end_points["depth_map_pred"] = depth_448
        end_points["depth_map_used_for_geometry"] = depth_448
        end_points["depth_tok_pred"] = depth_tok
        end_points["D: Geometry depth source GT"] = depth_448.new_tensor(
            float(self.geometry_depth_source == "gt")
        ).reshape(())
        end_points["D: Depth head executed"] = depth_448.new_tensor(
            float(depth_net_pred_448 is not None)
        ).reshape(())

        if depth_net_pred_448 is not None:
            # Network-predicted absolute depth and raw DPT output exist only for
            # the RGB-only student / legacy RGB-D compatibility path.
            end_points["depth_net_pred"] = depth_net_pred_448
            end_points["depth_head_raw_pred"] = depth_head_raw_448

        if self.pose_depth_mode != "none":
            # Common mode-independent diagnostics.
            for key in (
                "pose_depth_gamma_abs_mean",
                "pose_depth_beta_abs_mean",
                "pose_depth_gamma_abs_mean_levels",
                "pose_depth_beta_abs_mean_levels",
                "pose_depth_gamma_abs_max_levels",
                "pose_depth_beta_abs_max_levels",
                "pose_depth_gamma_spatial_std",
                "pose_depth_beta_spatial_std",
                "pose_depth_gamma_spatial_std_levels",
                "pose_depth_beta_spatial_std_levels",
            ):
                if key in pose_depth_aux:
                    end_points[key] = pose_depth_aux[key]

            # Preserve the old logger keys for global-FiLM checkpoints/scripts.
            end_points["pose_film_gamma_abs_mean"] = pose_depth_aux[
                "pose_film_gamma_abs_mean"
            ]
            end_points["pose_film_beta_abs_mean"] = pose_depth_aux[
                "pose_film_beta_abs_mean"
            ]
            end_points["pose_film_gamma_abs_mean_levels"] = pose_depth_aux[
                "pose_film_gamma_abs_mean_levels"
            ]
            end_points["pose_film_beta_abs_mean_levels"] = pose_depth_aux[
                "pose_film_beta_abs_mean_levels"
            ]

            if "camera_pose_unit" in pose_depth_aux:
                end_points["camera_pose_unit"] = pose_depth_aux[
                    "camera_pose_unit"
                ]
            if "camera_gravity_unit" in pose_depth_aux:
                end_points["camera_gravity_unit"] = pose_depth_aux[
                    "camera_gravity_unit"
                ]
            for key in (
                "ray_gravity_alignment_mean",
                "ray_gravity_alignment_min",
                "ray_gravity_alignment_max",
                "ray_gravity_alignment_map",
            ):
                if key in pose_depth_aux:
                    end_points[key] = pose_depth_aux[key]

            end_points["D: PoseDepth gamma abs mean"] = pose_depth_aux[
                "pose_depth_gamma_abs_mean"
            ].reshape(())
            end_points["D: PoseDepth beta abs mean"] = pose_depth_aux[
                "pose_depth_beta_abs_mean"
            ].reshape(())
            end_points["D: PoseDepth gamma spatial std"] = pose_depth_aux[
                "pose_depth_gamma_spatial_std"
            ].reshape(())
            end_points["D: PoseDepth beta spatial std"] = pose_depth_aux[
                "pose_depth_beta_spatial_std"
            ].reshape(())
            # Legacy names retained for existing training-log tooling.
            end_points["D: PoseFiLM gamma abs mean"] = pose_depth_aux[
                "pose_film_gamma_abs_mean"
            ].reshape(())
            end_points["D: PoseFiLM beta abs mean"] = pose_depth_aux[
                "pose_film_beta_abs_mean"
            ].reshape(())
            if "ray_gravity_alignment_mean" in pose_depth_aux:
                end_points["D: RayGravity alignment mean"] = pose_depth_aux[
                    "ray_gravity_alignment_mean"
                ].reshape(())
                end_points["D: RayGravity alignment min"] = pose_depth_aux[
                    "ray_gravity_alignment_min"
                ].reshape(())
                end_points["D: RayGravity alignment max"] = pose_depth_aux[
                    "ray_gravity_alignment_max"
                ].reshape(())

        if self.use_obs_depth:
            end_points["obs_depth_m_used"] = obs_depth_448
            end_points["sensor_depth_m_used"] = obs_depth_448  # compatibility
            end_points["depth_confidence_pred"] = depth_confidence_448

            # Compatibility/debug: correction relative to observed depth.
            end_points["depth_refined_correction"] = depth_refined_correction_448
            end_points["depth_residual_pred"] = depth_refined_correction_448
        else:
            if depth_net_pred_448 is not None:
                end_points["D: Depth net pred mean"] = (
                    depth_net_pred_448.detach().mean()
                )
            end_points["depth_residual_pred"] = torch.zeros_like(depth_448)

        objectness_score = objectness_logits_448.view(B, 2, -1).contiguous()
        graspness_score = graspness_logits_448.view(B, 1, -1).contiguous()
        end_points['objectness_score'] = objectness_score
        end_points['graspness_score'] = graspness_score

        objectness_pred = torch.argmax(objectness_score, dim=1)
        grasp_raw = graspness_score.squeeze(1)
        grasp_sel = grasp_raw.clamp(0.0, 1.0)

        if 'token_valid_mask' in end_points:
            valid_tok = end_points['token_valid_mask'].bool()
            if valid_tok.shape[1] != Ntok:
                raise ValueError(f'Expected token_valid_mask with {Ntok}, got {tuple(valid_tok.shape)}')
        else:
            valid_tok = torch.ones((B, Ntok), device=img.device, dtype=torch.bool)

        depth_valid_tok = (
            torch.isfinite(depth_448)
            & (depth_448 > self.min_depth)
            & (depth_448 < self.max_depth)
        ).view(B, -1)

        valid_tok = valid_tok & depth_valid_tok

        end_points['dbg_depth_valid'] = depth_valid_tok.detach()
        end_points['D: DepthValid#'] = depth_valid_tok.float().sum(dim=1).mean().reshape(())
        end_points['D: DepthValid ratio'] = depth_valid_tok.float().mean().reshape(())

        mask_obj_pred = valid_tok & (objectness_pred == 1)
        mask_thr_pred = mask_obj_pred & (grasp_sel > float(cfgs.graspness_threshold))

        end_points['dbg_grasp_raw'] = grasp_raw.detach()
        end_points['dbg_grasp_sel'] = grasp_sel.detach()
        end_points['dbg_mask_obj'] = mask_obj_pred.detach()
        end_points['dbg_mask_pred'] = mask_thr_pred.detach()
        end_points['dbg_objectness_pred'] = objectness_pred.detach()
        end_points['D: PredCand#(thr)'] = mask_thr_pred.float().sum(dim=1).mean().reshape(())
        end_points['D: PredObj#'] = mask_obj_pred.float().sum(dim=1).mean().reshape(())
        end_points['D: GraspRaw min'] = grasp_raw.min().reshape(())
        end_points['D: GraspRaw max'] = grasp_raw.max().reshape(())
        end_points['D: GraspSel mean'] = grasp_sel.mean().reshape(())

        graspable_mask = mask_thr_pred
        (
            seed_features_graspable,
            seed_xyz_graspable,
            token_sel_idx,
            xyz_all_pred,
            uv_all,
            graspable_num_batch,
        ) = self._select_graspable_seed_queries(
            feat_grid=feat_grid,
            depth_map=depth_448,
            camera_K=K,
            graspable_mask=graspable_mask,
            valid_tok=valid_tok,
            grasp_score=grasp_sel,
            end_points=end_points,
        )

        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['token_sel_idx'] = token_sel_idx
        end_points['token_sel_xyz'] = seed_xyz_graspable
        end_points['D: Graspable Points'] = torch.tensor(
            graspable_num_batch / float(B), device=img.device
        )
        end_points['D: PointCloudFree Seeds'] = depth_448.new_tensor(
            float(self.seed_selection_mode == 'image_fps')
        ).reshape(())

        if (self.vis_dir is not None) and (self._vis_iter % self.vis_every == 0):
            try:
                self._save_map_png(grasp_sel[0].view(H, W), os.path.join(self.vis_dir, f'dpt_grasp_map_it{self._vis_iter:06d}.png'), cmap='viridis')
                self._save_map_png(objectness_pred[0].view(H, W).float(), os.path.join(self.vis_dir, f'dpt_objectness_it{self._vis_iter:06d}.png'), cmap='gray')
                self._save_map_png(
                    depth_448[0, 0],
                    os.path.join(self.vis_dir, f'dpt_final_depth_it{self._vis_iter:06d}.png'),
                    cmap='magma',
                    vmin=self.min_depth,
                    vmax=self.max_depth,
                )

                self._save_map_png(
                    depth_448[0, 0],
                    os.path.join(self.vis_dir, f'dpt_final_depth_it{self._vis_iter:06d}.png'),
                    cmap='magma',
                    vmin=self.min_depth,
                    vmax=self.max_depth,
                    title='final depth',
                )

                if depth_net_pred_448 is not None:
                    self._save_map_png(
                        depth_net_pred_448[0, 0],
                        os.path.join(self.vis_dir, f'dpt_depth_head_abs_debug_it{self._vis_iter:06d}.png'),
                        cmap='magma',
                        vmin=self.min_depth,
                        vmax=self.max_depth,
                        title='depth head sigmoid(abs) debug',
                    )

                    self._save_map_png(
                        depth_head_raw_448[0, 0],
                        os.path.join(self.vis_dir, f'dpt_depth_head_raw_it{self._vis_iter:06d}.png'),
                        cmap='coolwarm',
                        title='depth head raw output',
                    )

                if self.use_obs_depth:
                    self._save_map_png(
                        obs_depth_448[0, 0],
                        os.path.join(self.vis_dir, f'dpt_obs_depth_it{self._vis_iter:06d}.png'),
                        cmap='magma',
                        vmin=self.min_depth,
                        vmax=self.max_depth,
                        title='observed depth',
                    )

                    self._save_map_png(
                        depth_confidence_448[0, 0],
                        os.path.join(self.vis_dir, f'dpt_depth_confidence_it{self._vis_iter:06d}.png'),
                        cmap='viridis',
                        vmin=0.0,
                        vmax=1.0,
                        title='confidence of network predicted depth',
                    )

                    self._save_map_png(
                        depth_refined_correction_448[0, 0],
                        os.path.join(self.vis_dir, f'dpt_depth_refined_correction_it{self._vis_iter:06d}.png'),
                        cmap='coolwarm',
                        title='final depth - observed depth',
                    )

                    oor_mask = (
                        (~torch.isfinite(depth_448))
                        | (depth_448 <= self.min_depth)
                        | (depth_448 >= self.max_depth)
                    ).float()

                    self._save_map_png(
                        oor_mask[0, 0],
                        os.path.join(self.vis_dir, f'dpt_final_depth_out_of_range_it{self._vis_iter:06d}.png'),
                        cmap='gray',
                        vmin=0.0,
                        vmax=1.0,
                        title='final depth out-of-range mask',
                    )

                if 'gt_depth_m' in end_points:
                    gt_depth = end_points['gt_depth_m']
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.dim() == 4:
                        gt_depth = gt_depth[:, :1]

                    gt_depth = gt_depth.to(device=depth_448.device, dtype=depth_448.dtype)

                    if gt_depth.shape[-2:] != (H, W):
                        gt_depth = F.interpolate(gt_depth, size=(H, W), mode='nearest')

                    gt_valid = (
                        torch.isfinite(gt_depth)
                        & (gt_depth > self.min_depth)
                        & (gt_depth < self.max_depth)
                    ).float()

                    final_abs_err = (depth_448 - gt_depth).abs() * gt_valid

                    self._save_map_png(
                        final_abs_err[0, 0],
                        os.path.join(self.vis_dir, f'dpt_final_depth_abs_err_it{self._vis_iter:06d}.png'),
                        cmap='magma',
                        vmin=0.0,
                        title='|geometry depth - GT|',
                    )

                    if depth_net_pred_448 is not None:
                        net_abs_err = (depth_net_pred_448 - gt_depth).abs() * gt_valid
                        self._save_map_png(
                            net_abs_err[0, 0],
                            os.path.join(self.vis_dir, f'dpt_depth_net_pred_abs_err_it{self._vis_iter:06d}.png'),
                            cmap='magma',
                            vmin=0.0,
                            title='|network predicted depth - GT|',
                        )

                    if self.use_obs_depth:
                        correction_target = gt_depth - obs_depth_448
                        correction_err = (depth_refined_correction_448 - correction_target).abs() * gt_valid

                        self._save_map_png(
                            correction_target[0, 0],
                            os.path.join(self.vis_dir, f'dpt_depth_correction_target_it{self._vis_iter:06d}.png'),
                            cmap='coolwarm',
                            title='GT - observed depth',
                        )

                        self._save_map_png(
                            correction_err[0, 0],
                            os.path.join(self.vis_dir, f'dpt_depth_correction_abs_err_it{self._vis_iter:06d}.png'),
                            cmap='magma',
                            vmin=0.0,
                            title='|fused correction - target correction|',
                        )

                pts_uv = torch.stack([(token_sel_idx[0] % W).float(), (token_sel_idx[0] // W).float()], dim=-1)
                self._save_overlay_points(img[0], pts_uv, os.path.join(self.vis_dir, f'dpt_seed_overlay_it{self._vis_iter:06d}.png'))
                if 'gt_depth_m' in end_points:
                    gt_depth = end_points['gt_depth_m']
                    if gt_depth.dim() == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    if xyz_all_pred is not None and uv_all is not None:
                        z_all_gt = gt_depth.view(B, -1, 1).contiguous().clamp_min(1e-6)
                        xyz_all_gt = self._backproject_uvz(uv_all, z_all_gt, K)
                        self._save_pred_gt_cloud_ply(
                            xyz_all_pred, xyz_all_gt, end_points
                        )
                    
            except Exception:
                pass

        if self.is_training:
            if self.use_depth_comp:
                raise RuntimeError(
                    "The switchable CVA trainer requires extended-angle "
                    "labels. use_depth_comp currently has no extended-angle "
                    "label matcher; disable it or implement a dedicated "
                    "process_grasp_labels_extend_angle_depth_comp function."
                )
            process_fn = (
                process_grasp_labels_cdf_width
                if self.use_cdf
                else process_grasp_labels_extend_angle
            )
            process_kwargs = None
        else:
            process_fn = None
            process_kwargs = None

        end_points = self.kview_grasp_module(
            seed_features=seed_features_graspable,
            seed_xyz=seed_xyz_graspable,
            token_sel_idx=token_sel_idx,
            feat_map=feat_grid,
            depth_map=depth_448,
            camera_K=K,
            end_points=end_points,
            is_training=self.is_training,
            process_grasp_labels_fn=process_fn,
            process_grasp_labels_kwargs=process_kwargs,
            topview_debug_fn=(
                self._add_topview_quality_logs if self.is_training else None
            ),
            depth_prob=None,
            objectness_logits=objectness_logits_448,
            graspness_map=grasp_sel.view(B, 1, H, W).contiguous(),
            img=img,
        )

        self._assert_cva_output_contract(end_points)
        end_points["D: CDF enabled"] = depth_448.new_tensor(
            float(self.use_cdf)
        ).reshape(())
        end_points["D: Geometry diagnostics enabled"] = (
            depth_448.new_tensor(
                float(self.use_geometry_diagnostics)
            ).reshape(())
        )

        if self.use_geometry_diagnostics:
            if self.rgb_geometry_diagnostics is None:
                raise RuntimeError(
                    "use_geometry_diagnostics=True but the "
                    "diagnostic module was not initialized."
                )
            end_points = self.rgb_geometry_diagnostics(
                end_points=end_points,
                depth_pred=depth_448,
                gt_depth=end_points.get("gt_depth_m", None),
                K=K,
                img=img,
                step=self._vis_iter,
                modality=(
                    "rgb_gt_depth"
                    if self.geometry_depth_source == "gt"
                    else ("rgbd" if self.use_obs_depth else "rgb")
                ),
                use_cdf=self.use_cdf,
            )

        with torch.no_grad():
            end_points["D: Depth final mean"] = depth_448.detach().mean()
            end_points["D: Depth final min"] = depth_448.detach().min()
            end_points["D: Depth final max"] = depth_448.detach().max()
            end_points["D: Depth final out-of-range ratio"] = (
                (~torch.isfinite(depth_448))
                | (depth_448 <= self.min_depth)
                | (depth_448 >= self.max_depth)
            ).float().mean()

            if "gt_depth_m" in end_points:
                gt_depth_dbg = end_points["gt_depth_m"]
                if gt_depth_dbg.dim() == 3:
                    gt_depth_dbg = gt_depth_dbg.unsqueeze(1)
                elif gt_depth_dbg.dim() == 4:
                    gt_depth_dbg = gt_depth_dbg[:, :1]

                gt_depth_dbg = gt_depth_dbg.to(depth_448)

                if gt_depth_dbg.shape[-2:] != depth_448.shape[-2:]:
                    gt_depth_dbg = F.interpolate(
                        gt_depth_dbg,
                        size=depth_448.shape[-2:],
                        mode="nearest",
                    )

                valid_dbg = (
                    torch.isfinite(gt_depth_dbg)
                    & (gt_depth_dbg > self.min_depth)
                    & (gt_depth_dbg < self.max_depth)
                )

                if valid_dbg.any():
                    end_points["D: Depth final MAE"] = (
                        depth_448 - gt_depth_dbg
                    ).abs()[valid_dbg].mean()

                    if depth_net_pred_448 is not None:
                        end_points["D: Depth net pred MAE"] = (
                            depth_net_pred_448 - gt_depth_dbg
                        ).abs()[valid_dbg].mean()

                    if self.use_obs_depth:
                        end_points["D: ObsDepth MAE"] = (
                            obs_depth_448 - gt_depth_dbg
                        ).abs()[valid_dbg].mean()

                        correction_target_dbg = gt_depth_dbg - obs_depth_448
                        end_points["D: Depth correction target abs"] = (
                            correction_target_dbg.abs()[valid_dbg].mean()
                        )
                        end_points["D: Depth correction MAE"] = (
                            depth_refined_correction_448 - correction_target_dbg
                        ).abs()[valid_dbg].mean()

                        end_points["D: Depth refine gain"] = (
                            end_points["D: ObsDepth MAE"] -
                            end_points["D: Depth final MAE"]
                        )
                
        if (self._vis_iter % self.debug_print_every == 0):
            with torch.no_grad():
                msg = (
                    f"[economicgrasp_dpt] it={self._vis_iter} "
                    f"cdf={int(self.use_cdf)} "
                    f"geomdiag={int(self.use_geometry_diagnostics)} "
                    f"obs={int(self.use_obs_depth)} "
                    f"poseD={self.pose_depth_mode} "
                    f"zsrc={self.geometry_depth_source} "
                    f"seed={self.seed_selection_mode} "
                    f"graspable={end_points['D: Graspable Points'].item():.1f} "
                    f"cand={end_points['D: PredCand#(thr)'].item():.1f} "
                    f"obj={end_points['D: PredObj#'].item():.1f} "
                    f"grasp_mean={end_points['D: GraspSel mean'].item():.4f} "
                    f"z_mean={end_points['D: Depth final mean'].item():.4f} "
                    f"z_oor={end_points['D: Depth final out-of-range ratio'].item():.4f}"
                )
                if "D: Depth final MAE" in end_points:
                    msg += f" z_mae={end_points['D: Depth final MAE'].item():.4f}"
                if "D: Depth refine gain" in end_points:
                    msg += f" refine_gain={end_points['D: Depth refine gain'].item():.4f}"
                if "D: PoseDepth gamma abs mean" in end_points:
                    msg += (
                        f" pose_gamma={end_points['D: PoseDepth gamma abs mean'].item():.6f}"
                        f" pose_beta={end_points['D: PoseDepth beta abs mean'].item():.6f}"
                        f" pose_gstd={end_points['D: PoseDepth gamma spatial std'].item():.6f}"
                    )
                print(msg)

        self._vis_iter += 1
        return end_points
    

class economicgrasp_dpt_distill(economicgrasp_dpt):
    """EconomicGrasp-DPT with deterministic image-FPS sparse queries."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Ignore selector arguments from the earlier image-topk prototype so an
        # old launcher cannot silently reactivate a different seed path.
        kwargs.pop("seed_selection_mode", None)
        kwargs.pop("image_seed_nms_kernel", None)
        # The distillation experiment never uses the legacy train-only GT-XYZ
        # switch. Geometry privilege is controlled solely by depth source.
        kwargs["use_gt_xyz_for_train"] = False
        super().__init__(*args, seed_selection_mode="image_fps", **kwargs)


class economicgrasp_dpt_teacher(economicgrasp_dpt_distill):
    """Stage-0/2 privileged teacher using clean synthetic depth geometry."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("geometry_depth_source", None)
        kwargs["geometry_depth_source"] = "gt"
        kwargs["use_obs_depth"] = False
        kwargs["pose_depth_mode"] = "none"
        super().__init__(*args, **kwargs)


class economicgrasp_dpt_student(economicgrasp_dpt_distill):
    """Stage-1/2 RGB-only student using its predicted metric depth."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("geometry_depth_source", None)
        kwargs["geometry_depth_source"] = "pred"
        super().__init__(*args, **kwargs)


@dataclass(frozen=True)
class OutputDistillationConfig:
    """Weights and validity thresholds for privileged-teacher output KD."""

    overall_weight: float = 1.0
    objectness_weight: float = 1.0
    graspness_weight: float = 1.0
    depth_weight: float = 0.0
    view_weight: float = 1.0
    cdf_weight: float = 1.0
    width_weight: float = 0.1

    temperature: float = 1.0
    max_query_view_angle_deg: float = 35.0
    width_positive_threshold: float = 0.5
    min_depth: float = 0.2
    max_depth: float = 1.0
    eps: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Retain only tensors required by the KD objective. In particular,
# ``kview_base_token_sel_idx`` is passed back to the student as an exact seed
# override in stage 2.
_DISTILL_TARGET_KEYS = (
    "objectness_score",
    "graspness_score",
    "depth_map_pred",
    "token_valid_mask",
    "view_score",
    "kview_base_token_sel_idx",
    "token_sel_idx",
    "grasp_top_view_xyz",
    "grasp_cdf_pred_angle_depth",
    "grasp_width_pred_angle_depth",
)


def extract_distillation_targets(
    teacher_end_points: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Detach the strict subset needed by stage-2 distillation."""
    targets: Dict[str, torch.Tensor] = {}
    missing = []
    for key in _DISTILL_TARGET_KEYS:
        value = teacher_end_points.get(key, None)
        if value is None:
            if key == "token_valid_mask":
                continue
            missing.append(key)
            continue
        if not torch.is_tensor(value):
            raise TypeError(
                f"Teacher endpoint {key!r} must be a tensor, got {type(value)}."
            )
        targets[key] = value.detach()
    if missing:
        raise KeyError(
            "Stage-2 teacher is missing required CDF endpoint(s): "
            + ", ".join(missing)
        )
    return targets


def load_checkpoint_state(
    model: nn.Module,
    checkpoint_path: str,
    *,
    strict: bool = True,
    checkpoint_data: Any = None,
) -> Dict[str, Any]:
    """Load a state dict, optionally reusing prevalidated checkpoint data."""
    if checkpoint_data is None:
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be non-empty.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    else:
        checkpoint = checkpoint_data
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state, Mapping):
        raise TypeError(
            f"Checkpoint does not contain a state dict: {checkpoint_path}"
        )
    result = model.load_state_dict(state, strict=False)
    optional_prefixes = ("rgb_geometry_diagnostics.",)
    missing = [
        key for key in result.missing_keys
        if not key.startswith(optional_prefixes)
    ]
    unexpected = [
        key for key in result.unexpected_keys
        if not key.startswith(optional_prefixes)
    ]
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Strict checkpoint loading produced incompatible keys: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": state}


def _zero_from_student(student_end_points: Mapping[str, Any]) -> torch.Tensor:
    for key in (
        "grasp_cdf_pred_angle_depth",
        "view_score",
        "objectness_score",
        "depth_map_pred",
    ):
        value = student_end_points.get(key, None)
        if torch.is_tensor(value):
            return value.sum() * 0.0
    raise KeyError("No differentiable student output is available for KD loss.")


def _normalize_view_score_shape(
    score: torch.Tensor,
    num_query: int,
) -> torch.Tensor:
    if score.dim() != 3:
        raise ValueError(
            f"view_score must be [B,Q,V] or [B,V,Q], got {tuple(score.shape)}"
        )
    if score.shape[1] == num_query:
        return score.contiguous()
    if score.shape[2] == num_query:
        return score.transpose(1, 2).contiguous()
    raise ValueError(
        f"Cannot align view_score {tuple(score.shape)} with Q={num_query}."
    )


def _gather_query_dim(
    tensor: torch.Tensor,
    match: torch.Tensor,
    query_dim: int,
) -> torch.Tensor:
    """Gather a teacher tensor along its query dimension with [B,Qs] indices."""
    if tensor.shape[0] != match.shape[0]:
        raise ValueError("Teacher tensor and query match batch sizes differ.")
    shape = list(tensor.shape)
    index_shape = shape.copy()
    index_shape[query_dim] = match.shape[1]
    view = [match.shape[0]] + [1] * (tensor.dim() - 1)
    view[query_dim] = match.shape[1]
    index = match.view(*view).expand(*index_shape)
    return torch.gather(tensor, query_dim, index)


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=torch.bool)
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    if bool(mask.any()):
        return value[mask].mean()
    return zero


def _assert_shared_base_image_fps(
    student_idx: torch.Tensor,
    teacher_idx: torch.Tensor,
) -> torch.Tensor:
    """Require identical ordered image-FPS seeds and return equality mask."""
    if student_idx.dim() != 2 or teacher_idx.dim() != 2:
        raise ValueError(
            "Base image-FPS indices must be [B,M], got "
            f"student={tuple(student_idx.shape)}, teacher={tuple(teacher_idx.shape)}"
        )
    if student_idx.shape != teacher_idx.shape:
        raise RuntimeError(
            "Teacher/student image-FPS seed shape mismatch: "
            f"{tuple(student_idx.shape)} vs {tuple(teacher_idx.shape)}."
        )
    equal = student_idx == teacher_idx
    if not bool(equal.all()):
        mismatch = float((~equal).float().mean().item())
        raise RuntimeError(
            "Stage-2 requires exact shared image-FPS seeds, but the ordered "
            f"base indices differ at {100.0 * mismatch:.3f}% of positions."
        )
    return equal


def _same_seed_query_match(
    student_base_idx: torch.Tensor,
    teacher_base_idx: torch.Tensor,
    student_query_idx: torch.Tensor,
    student_view_xyz: torch.Tensor,
    teacher_query_idx: torch.Tensor,
    teacher_view_xyz: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match teacher views only within the same ordered image-FPS seed.

    K-view expansion is base-major in the CVA selector. After exact base-seed
    sharing, query tensors can be reshaped to [B,M,K,*]. Each student view is
    paired with the closest physical teacher view belonging to that same base
    image location. No global center matching or pixel-distance threshold is
    needed.
    """
    _assert_shared_base_image_fps(student_base_idx, teacher_base_idx)
    B, M = student_base_idx.shape
    if student_query_idx.dim() != 2 or teacher_query_idx.dim() != 2:
        raise ValueError("Expanded token indices must be [B,Q].")
    if student_query_idx.shape[0] != B or teacher_query_idx.shape[0] != B:
        raise ValueError("Expanded query batch size differs from base seeds.")
    Qs = int(student_query_idx.shape[1])
    Qt = int(teacher_query_idx.shape[1])
    if Qs % M != 0 or Qt % M != 0:
        raise RuntimeError(
            f"K-view query count must be divisible by M={M}; got Qs/Qt={Qs}/{Qt}."
        )
    Ks = Qs // M
    Kt = Qt // M
    if Ks <= 0 or Kt <= 0:
        raise RuntimeError("Teacher/student produced no K-view query.")

    s_idx = student_query_idx.view(B, M, Ks)
    t_idx = teacher_query_idx.view(B, M, Kt)
    expected_s = student_base_idx.unsqueeze(-1).expand_as(s_idx)
    expected_t = teacher_base_idx.unsqueeze(-1).expand_as(t_idx)
    if not bool((s_idx == expected_s).all()):
        raise RuntimeError(
            "Student K-view query ordering is not base-major or does not reuse "
            "the shared image-FPS base indices."
        )
    if not bool((t_idx == expected_t).all()):
        raise RuntimeError(
            "Teacher K-view query ordering is not base-major or does not reuse "
            "the shared image-FPS base indices."
        )

    if student_view_xyz.shape != (B, Qs, 3):
        raise ValueError(
            f"student_view_xyz must be {(B, Qs, 3)}, got {tuple(student_view_xyz.shape)}"
        )
    if teacher_view_xyz.shape != (B, Qt, 3):
        raise ValueError(
            f"teacher_view_xyz must be {(B, Qt, 3)}, got {tuple(teacher_view_xyz.shape)}"
        )

    s_view = F.normalize(
        student_view_xyz.detach().float(), dim=-1
    ).view(B, M, Ks, 3)
    t_view = F.normalize(
        teacher_view_xyz.detach().float(), dim=-1
    ).view(B, M, Kt, 3)
    cosine = torch.einsum("bmqc,bmkc->bmqk", s_view, t_view).clamp(-1.0, 1.0)
    local_match = cosine.argmax(dim=-1)
    parent_offset = (
        torch.arange(M, device=local_match.device, dtype=torch.long)
        .view(1, M, 1)
        .expand(B, M, Ks)
        * Kt
    )
    global_match = (parent_offset + local_match).reshape(B, Qs).contiguous()
    matched_cos = torch.gather(
        cosine, dim=-1, index=local_match.unsqueeze(-1)
    ).squeeze(-1)
    matched_cos = matched_cos.clamp(-1.0 + float(eps), 1.0 - float(eps))
    angle_deg = torch.rad2deg(torch.acos(matched_cos)).reshape(B, Qs)
    return global_match, angle_deg


def compute_output_distillation_loss(
    student_end_points: Dict[str, Any],
    teacher_targets: Mapping[str, torch.Tensor],
    config: OutputDistillationConfig,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute stage-2 output KD under exact shared image-FPS seeds."""
    zero = _zero_from_student(student_end_points)
    temperature = max(float(config.temperature), float(config.eps))

    required_student = (
        "objectness_score",
        "graspness_score",
        "depth_map_pred",
        "view_score",
        "kview_base_token_sel_idx",
        "token_sel_idx",
        "grasp_top_view_xyz",
        "grasp_cdf_pred_angle_depth",
        "grasp_width_pred_angle_depth",
    )
    missing = [key for key in required_student if key not in student_end_points]
    if missing:
        raise KeyError(
            "Stage-2 student is missing required CDF endpoint(s): "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # Dense pixel-aligned outputs.
    # ------------------------------------------------------------------
    s_obj = student_end_points["objectness_score"]
    t_obj = teacher_targets["objectness_score"].to(s_obj)
    if s_obj.shape != t_obj.shape:
        raise ValueError(
            f"Objectness KD shape mismatch: {tuple(s_obj.shape)} vs {tuple(t_obj.shape)}"
        )
    valid_tok = student_end_points.get(
        "token_valid_mask",
        teacher_targets.get("token_valid_mask", None),
    )
    if torch.is_tensor(valid_tok):
        valid_tok = valid_tok.to(device=s_obj.device).bool()
    else:
        valid_tok = torch.ones(
            s_obj.shape[0], s_obj.shape[-1], device=s_obj.device, dtype=torch.bool
        )

    obj_kl = F.kl_div(
        F.log_softmax(s_obj / temperature, dim=1),
        F.softmax(t_obj / temperature, dim=1),
        reduction="none",
    ).sum(dim=1) * (temperature ** 2)
    objectness_loss = _masked_mean(obj_kl, valid_tok, zero)

    s_grasp = student_end_points["graspness_score"]
    t_grasp = teacher_targets["graspness_score"].to(s_grasp)
    if s_grasp.shape != t_grasp.shape:
        raise ValueError(
            f"Graspness KD shape mismatch: {tuple(s_grasp.shape)} vs {tuple(t_grasp.shape)}"
        )
    grasp_map = F.smooth_l1_loss(s_grasp, t_grasp, reduction="none").squeeze(1)
    graspness_loss = _masked_mean(grasp_map, valid_tok, zero)

    # The RGB student already receives direct GT metric-depth supervision via
    # the unchanged supervised loss.  By default depth KD is disabled so Stage 2
    # measures task-output transfer rather than duplicating the same GT target.
    if float(config.depth_weight) > 0.0:
        s_depth = student_end_points["depth_map_pred"]
        t_depth = teacher_targets["depth_map_pred"].to(s_depth)
        if s_depth.shape != t_depth.shape:
            t_depth = F.interpolate(
                t_depth,
                size=s_depth.shape[-2:],
                mode="nearest",
            )
        depth_valid = (
            torch.isfinite(t_depth)
            & (t_depth > float(config.min_depth))
            & (t_depth < float(config.max_depth))
            & torch.isfinite(s_depth)
        )
        depth_map = F.smooth_l1_loss(s_depth, t_depth, reduction="none")
        depth_loss = _masked_mean(depth_map, depth_valid, zero)
    else:
        depth_loss = zero

    # ------------------------------------------------------------------
    # Exact same-base view field.
    # ------------------------------------------------------------------
    s_base_idx = student_end_points["kview_base_token_sel_idx"].long()
    t_base_idx = teacher_targets["kview_base_token_sel_idx"].to(
        device=s_base_idx.device, dtype=torch.long
    )
    shared_equal = _assert_shared_base_image_fps(s_base_idx, t_base_idx)
    s_view = _normalize_view_score_shape(
        student_end_points["view_score"], s_base_idx.shape[1]
    )
    t_view = _normalize_view_score_shape(
        teacher_targets["view_score"].to(s_view), t_base_idx.shape[1]
    )
    if s_view.shape != t_view.shape:
        raise ValueError(
            "Shared-seed view field shape mismatch: "
            f"student={tuple(s_view.shape)}, teacher={tuple(t_view.shape)}"
        )
    view_loss = F.smooth_l1_loss(s_view, t_view, reduction="mean")

    # ------------------------------------------------------------------
    # Same-center CVA query CDF and depth-wise width.
    # ------------------------------------------------------------------
    s_query_idx = student_end_points["token_sel_idx"].long()
    t_query_idx = teacher_targets["token_sel_idx"].to(
        device=s_query_idx.device, dtype=torch.long
    )
    s_query_view = student_end_points["grasp_top_view_xyz"]
    t_query_view = teacher_targets["grasp_top_view_xyz"].to(s_query_view)
    query_match, query_angle = _same_seed_query_match(
        student_base_idx=s_base_idx,
        teacher_base_idx=t_base_idx,
        student_query_idx=s_query_idx,
        student_view_xyz=s_query_view,
        teacher_query_idx=t_query_idx,
        teacher_view_xyz=t_query_view,
        eps=float(config.eps),
    )
    query_valid = query_angle <= float(config.max_query_view_angle_deg)

    s_cdf = student_end_points["grasp_cdf_pred_angle_depth"]
    t_cdf = teacher_targets["grasp_cdf_pred_angle_depth"].to(s_cdf)
    if s_cdf.dim() != 5 or t_cdf.dim() != 5:
        raise ValueError(
            "CDF outputs must be [B,T,Q,A,D], got "
            f"student={tuple(s_cdf.shape)}, teacher={tuple(t_cdf.shape)}"
        )
    t_cdf_matched = _gather_query_dim(t_cdf, query_match, query_dim=2)
    if s_cdf.shape != t_cdf_matched.shape:
        raise ValueError(
            "Matched CDF shape mismatch: "
            f"student={tuple(s_cdf.shape)}, teacher={tuple(t_cdf_matched.shape)}"
        )
    cdf_soft_target = torch.sigmoid(t_cdf_matched / temperature)
    cdf_map = F.binary_cross_entropy_with_logits(
        s_cdf / temperature,
        cdf_soft_target,
        reduction="none",
    ) * (temperature ** 2)
    cdf_valid = query_valid[:, None, :, None, None]
    cdf_loss = _masked_mean(cdf_map, cdf_valid, zero)

    s_width = student_end_points["grasp_width_pred_angle_depth"]
    t_width = teacher_targets["grasp_width_pred_angle_depth"].to(s_width)
    if s_width.dim() != 4 or t_width.dim() != 4:
        raise ValueError(
            "Width outputs must be [B,D,Q,A], got "
            f"student={tuple(s_width.shape)}, teacher={tuple(t_width.shape)}"
        )
    t_width_matched = _gather_query_dim(t_width, query_match, query_dim=2)
    if s_width.shape != t_width_matched.shape:
        raise ValueError(
            "Matched width shape mismatch: "
            f"student={tuple(s_width.shape)}, teacher={tuple(t_width_matched.shape)}"
        )
    teacher_positive = (
        cdf_soft_target.mean(dim=1).permute(0, 3, 1, 2)
        >= float(config.width_positive_threshold)
    )
    width_valid = query_valid.unsqueeze(1).unsqueeze(-1) & teacher_positive
    width_map = F.smooth_l1_loss(
        s_width, t_width_matched, reduction="none"
    )
    width_loss = _masked_mean(width_map, width_valid, zero)

    weighted = (
        float(config.objectness_weight) * objectness_loss
        + float(config.graspness_weight) * graspness_loss
        + float(config.depth_weight) * depth_loss
        + float(config.view_weight) * view_loss
        + float(config.cdf_weight) * cdf_loss
        + float(config.width_weight) * width_loss
    )
    total = float(config.overall_weight) * weighted

    student_end_points["B: KD Objectness Loss"] = objectness_loss
    student_end_points["B: KD Graspness Loss"] = graspness_loss
    student_end_points["B: KD Depth Loss"] = depth_loss
    student_end_points["B: KD View Loss"] = view_loss
    student_end_points["B: KD CDF Loss"] = cdf_loss
    student_end_points["B: KD Width Loss"] = width_loss
    student_end_points["A: Distill Loss"] = total

    with torch.no_grad():
        student_end_points["D: KD shared image-FPS ratio"] = (
            shared_equal.float().mean().reshape(())
        )
        student_end_points["D: KD query match ratio"] = (
            query_valid.float().mean().reshape(())
        )
        student_end_points["D: KD query view angle"] = (
            query_angle.mean().reshape(())
        )
        student_end_points["D: KD width positive ratio"] = (
            width_valid.float().mean().reshape(())
        )
        student_end_points["D: KD depth enabled"] = zero.new_tensor(
            float(config.depth_weight > 0.0)
        ).reshape(())

    return total, student_end_points
