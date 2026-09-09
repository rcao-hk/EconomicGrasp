#!/usr/bin/env python3
"""P1-O same-query GT-center oracle for EconomicGrasp-DPT CVA-CDF.

Scientific question
-------------------
Does replacing only the *selected sparse grasp-query center depth* with clean
same-pixel GT depth improve the frozen Stage-1 RGB grasp pipeline?

The intervention is deliberately narrow:
  * seed pixel/query ownership: unchanged Stage-1 deterministic image-FPS;
  * dense geometry/depth map: unchanged RGB-predicted metric depth;
  * selected sparse center only: z_pred(q) -> z_gt(q), same image pixel q;
  * invalid GT query depth: fall back to the native predicted center;
  * ViewNet/CVA/CDF/decoder: unchanged Stage-1 network, recomputed naturally
    after the center substitution (no GT view/angle/depth-bin labels injected);
  * optional collision post-processing: identical protocol for both branches.

The script runs the native Stage-1 baseline and P1-O oracle on the same batch,
asserts identical ordered image-FPS/query pixels, saves paired predictions under
``<save_dir>/baseline`` and ``<save_dir>/gt_center``, and writes a protocol /
diagnostic JSON. GraspNet AP is intentionally evaluated by the existing external
evaluator on those two output directories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Mapping

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup


def _parse_p1o_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--p1o_max_batches",
        type=int,
        default=0,
        help="Debug-only cap; 0 processes the complete sampled split.",
    )
    parser.add_argument(
        "--p1o_assert_atol",
        type=float,
        default=1.0e-6,
        help="Numerical tolerance for exact-query / GT-center assertions.",
    )
    parser.add_argument(
        "--p1o_summary_filename",
        type=str,
        default="p1o_same_query_gt_center_summary.json",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


P1O_ARGS = _parse_p1o_args()
if int(P1O_ARGS.p1o_max_batches) < 0:
    raise ValueError("--p1o_max_batches must be >= 0.")
if float(P1O_ARGS.p1o_assert_atol) <= 0.0:
    raise ValueError("--p1o_assert_atol must be positive.")

# Reuse the controlled Stage-0/1/2 inference contracts and data utilities.
import inference_cva_distill as base

from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student


cfgs = base.cfgs


class SameQueryGTCenterOracleStudent(economicgrasp_dpt_student):
    """Stage-1 student with a parameter-free switch for sparse GT-center use."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.p1o_use_gt_center = False

    @staticmethod
    def _token_uv(
        token_sel_idx: torch.Tensor,
        height: int,
        width: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if token_sel_idx.dim() != 2:
            raise ValueError(
                f"token_sel_idx must be [B,M], got {tuple(token_sel_idx.shape)}"
            )
        if bool(
            ((token_sel_idx < 0) | (token_sel_idx >= int(height * width))).any()
        ):
            raise ValueError("token_sel_idx contains an out-of-range pixel index.")
        u = (token_sel_idx % int(width)).to(dtype=dtype)
        v = (token_sel_idx // int(width)).to(dtype=dtype)
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _gather_depth(
        depth_b1hw: torch.Tensor,
        token_sel_idx: torch.Tensor,
    ) -> torch.Tensor:
        if depth_b1hw.dim() == 3:
            depth_b1hw = depth_b1hw.unsqueeze(1)
        if depth_b1hw.dim() != 4 or depth_b1hw.shape[1] < 1:
            raise ValueError(
                "Depth map must be [B,H,W] or [B,1,H,W], got "
                f"{tuple(depth_b1hw.shape)}"
            )
        B = int(depth_b1hw.shape[0])
        flat = depth_b1hw[:, 0].reshape(B, -1)
        return torch.gather(flat, 1, token_sel_idx.long())

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
        (
            seed_features,
            seed_xyz_base,
            token_sel_idx,
            xyz_all_pred,
            uv_all,
            graspable_num_batch,
        ) = super()._select_graspable_seed_queries(
            feat_grid=feat_grid,
            depth_map=depth_map,
            camera_K=camera_K,
            graspable_mask=graspable_mask,
            valid_tok=valid_tok,
            grasp_score=grasp_score,
            end_points=end_points,
        )

        end_points["p1o_center_xyz_base"] = seed_xyz_base
        end_points["p1o_center_token_sel_idx"] = token_sel_idx
        end_points["D: P1O GT center active"] = seed_xyz_base.new_tensor(
            float(self.p1o_use_gt_center)
        ).reshape(())

        if not self.p1o_use_gt_center:
            return (
                seed_features,
                seed_xyz_base,
                token_sel_idx,
                xyz_all_pred,
                uv_all,
                graspable_num_batch,
            )

        gt = end_points.get("gt_depth_m", None)
        if not torch.is_tensor(gt):
            raise KeyError(
                "P1-O requires end_points['gt_depth_m']; the oracle pass must "
                "retain clean GT depth while the baseline pass must not use it."
            )
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        elif gt.dim() == 4:
            gt = gt[:, :1]
        else:
            raise ValueError(
                f"gt_depth_m must be [B,H,W] or [B,1,H,W], got {tuple(gt.shape)}"
            )
        if depth_map.dim() != 4 or depth_map.shape[1] < 1:
            raise ValueError(
                f"Predicted depth map must be [B,1,H,W], got {tuple(depth_map.shape)}"
            )
        if tuple(gt.shape[-2:]) != tuple(depth_map.shape[-2:]):
            raise RuntimeError(
                "P1-O refuses to resample GT depth because same-pixel boundary "
                "semantics would change: gt_hw="
                f"{tuple(gt.shape[-2:])}, pred_hw={tuple(depth_map.shape[-2:])}."
            )

        _, _, C = seed_xyz_base.shape
        if C != 3:
            raise ValueError(
                f"Base sparse centers must be [B,M,3], got {tuple(seed_xyz_base.shape)}"
            )
        H, W = int(depth_map.shape[-2]), int(depth_map.shape[-1])
        uv = self._token_uv(
            token_sel_idx,
            H,
            W,
            dtype=seed_xyz_base.dtype,
        ).to(seed_xyz_base.device)

        base_z = seed_xyz_base[..., 2]
        gt_z = self._gather_depth(
            gt.to(device=base_z.device, dtype=base_z.dtype),
            token_sel_idx,
        )
        valid = (
            torch.isfinite(gt_z)
            & (gt_z >= float(self.min_depth))
            & (gt_z <= float(self.max_depth))
            & torch.isfinite(base_z)
        )
        oracle_z = torch.where(valid, gt_z, base_z)
        oracle_xyz = self._backproject_uvz(
            uv,
            oracle_z.unsqueeze(-1),
            camera_K.to(device=uv.device, dtype=uv.dtype),
        )
        gt_xyz = self._backproject_uvz(
            uv,
            gt_z.unsqueeze(-1),
            camera_K.to(device=uv.device, dtype=uv.dtype),
        )

        end_points["p1o_center_xyz_oracle"] = oracle_xyz
        end_points["p1o_center_xyz_gt"] = gt_xyz
        end_points["p1o_center_depth_base"] = base_z
        end_points["p1o_center_depth_gt"] = gt_z
        end_points["p1o_center_depth_oracle"] = oracle_z
        end_points["p1o_center_gt_valid_mask"] = valid

        with torch.no_grad():
            zero = base_z.new_zeros(())
            end_points["D: P1O GT center valid ratio"] = valid.float().mean()
            if bool(valid.any()):
                end_points["D: P1O base center z MAE"] = (
                    (base_z - gt_z).abs()[valid].mean()
                )
                end_points["D: P1O oracle center z MAE"] = (
                    (oracle_z - gt_z).abs()[valid].mean()
                )
                end_points["D: P1O base center xyz MAE"] = torch.linalg.norm(
                    seed_xyz_base - gt_xyz, dim=-1
                )[valid].mean()
                end_points["D: P1O oracle center xyz MAE"] = torch.linalg.norm(
                    oracle_xyz - gt_xyz, dim=-1
                )[valid].mean()
                end_points["D: P1O center displacement abs mean"] = (
                    oracle_z - base_z
                ).abs()[valid].mean()
            else:
                end_points["D: P1O base center z MAE"] = zero
                end_points["D: P1O oracle center z MAE"] = zero
                end_points["D: P1O base center xyz MAE"] = zero
                end_points["D: P1O oracle center xyz MAE"] = zero
                end_points["D: P1O center displacement abs mean"] = zero

        # The only model intervention: replace sparse center XYZ returned to
        # downstream ViewNet/CVA. Dense xyz_all_pred and depth_map remain native.
        return (
            seed_features,
            oracle_xyz,
            token_sel_idx,
            xyz_all_pred,
            uv_all,
            graspable_num_batch,
        )


def _fresh_input(
    pristine: Mapping[str, Any],
    *,
    include_gt_depth: bool,
) -> Dict[str, Any]:
    data = dict(pristine)
    if not include_gt_depth:
        data.pop("gt_depth_m", None)
    data["cva_compute_diagnostics"] = False
    data["geometry_compute_diagnostics"] = False
    data["cva_export_angle_feature"] = False
    data["cva_force_process_grasp_labels"] = False
    return data


def _save_json(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _require_tensor(end_points: Mapping[str, Any], key: str) -> torch.Tensor:
    value = end_points.get(key, None)
    if not torch.is_tensor(value):
        raise KeyError(f"Missing tensor endpoint {key!r}.")
    return value


def _assert_same_query(
    baseline: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key in ("kview_base_token_sel_idx", "token_sel_idx"):
        a = _require_tensor(baseline, key).detach().long()
        b = _require_tensor(oracle, key).detach().long()
        if a.shape != b.shape:
            raise RuntimeError(
                f"P1-O exact-query shape mismatch for {key}: "
                f"{tuple(a.shape)} vs {tuple(b.shape)}"
            )
        mismatch = float((a != b).float().mean().item())
        metrics[f"{key}_mismatch_ratio"] = mismatch
        if mismatch != 0.0:
            raise RuntimeError(
                f"P1-O requires identical ordered query pixels, but {key} "
                f"differs at {100.0 * mismatch:.6f}% positions."
            )

    bview = _require_tensor(baseline, "grasp_top_view_inds").detach().long()
    oview = _require_tensor(oracle, "grasp_top_view_inds").detach().long()
    if bview.shape != oview.shape:
        raise RuntimeError("Baseline/oracle view-index shapes differ.")
    metrics["view_index_change_ratio"] = float(
        (bview != oview).float().mean().item()
    )

    bxyz = _require_tensor(baseline, "grasp_top_view_xyz").detach().float()
    oxyz = _require_tensor(oracle, "grasp_top_view_xyz").detach().float()
    if bxyz.shape != oxyz.shape or bxyz.shape[-1] != 3:
        raise RuntimeError("Baseline/oracle view-vector shapes differ.")
    bxyz = torch.nn.functional.normalize(bxyz, dim=-1)
    oxyz = torch.nn.functional.normalize(oxyz, dim=-1)
    cosine = (bxyz * oxyz).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    metrics["view_angle_change_deg_mean"] = float(angle.mean().item())
    return metrics


def _accumulate_oracle_metrics(
    sums: Dict[str, float],
    counts: Dict[str, int],
    oracle: Mapping[str, Any],
) -> None:
    valid = _require_tensor(oracle, "p1o_center_gt_valid_mask").detach().bool()
    base_z = _require_tensor(oracle, "p1o_center_depth_base").detach()
    gt_z = _require_tensor(oracle, "p1o_center_depth_gt").detach()
    oracle_z = _require_tensor(oracle, "p1o_center_depth_oracle").detach()
    base_xyz = _require_tensor(oracle, "p1o_center_xyz_base").detach()
    gt_xyz = _require_tensor(oracle, "p1o_center_xyz_gt").detach()
    oracle_xyz = _require_tensor(oracle, "p1o_center_xyz_oracle").detach()

    total = int(valid.numel())
    nvalid = int(valid.sum().item())
    sums["query_total"] = sums.get("query_total", 0.0) + float(total)
    sums["query_valid"] = sums.get("query_valid", 0.0) + float(nvalid)
    counts["query_batches"] = counts.get("query_batches", 0) + 1
    if nvalid <= 0:
        return

    def add(name: str, values: torch.Tensor) -> None:
        selected = values[valid].double()
        sums[name] = sums.get(name, 0.0) + float(selected.sum().item())
        counts[name] = counts.get(name, 0) + int(selected.numel())

    add("base_z_abs_error_m", (base_z - gt_z).abs())
    add("oracle_z_abs_error_m", (oracle_z - gt_z).abs())
    add("center_displacement_abs_m", (oracle_z - base_z).abs())
    add("base_xyz_error_m", torch.linalg.norm(base_xyz - gt_xyz, dim=-1))
    add("oracle_xyz_error_m", torch.linalg.norm(oracle_xyz - gt_xyz, dim=-1))

    max_oracle_z_err = float((oracle_z - gt_z).abs()[valid].max().item())
    atol = float(P1O_ARGS.p1o_assert_atol)
    if max_oracle_z_err > atol:
        raise RuntimeError(
            "P1-O oracle center does not equal same-pixel GT depth on valid "
            f"queries: max_abs_error={max_oracle_z_err:.3e} > atol={atol:.3e}."
        )


def _accumulate_pair_metrics(
    sums: Dict[str, float],
    counts: Dict[str, int],
    metrics: Mapping[str, float],
) -> None:
    for key, value in metrics.items():
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1


def _mean_metric(
    sums: Mapping[str, float],
    counts: Mapping[str, int],
    key: str,
) -> float:
    count = int(counts.get(key, 0))
    if count <= 0:
        return float("nan")
    return float(sums.get(key, 0.0)) / float(count)


def _save_grasp_group(
    gg: GraspGroup,
    *,
    root: str,
    scene_name: str,
    camera: str,
    frame_id: int,
) -> None:
    out_dir = os.path.join(root, scene_name, camera)
    os.makedirs(out_dir, exist_ok=True)
    gg.save_npy(os.path.join(out_dir, f"{int(frame_id):04d}.npy"))


def inference() -> None:
    if not bool(getattr(cfgs, "multi_modal", False)):
        raise RuntimeError("P1-O CVA inference requires --multi_modal.")
    if not bool(getattr(cfgs, "use_cdf", False)):
        raise RuntimeError("P1-O supports CVA-CDF only; add --use_cdf.")
    if bool(getattr(cfgs, "use_obs_depth", False)):
        raise RuntimeError("P1-O is not an observed-depth experiment; remove --use_obs_depth.")
    if bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError(
            "Keep the legacy dataset --use_gt_depth switch disabled. P1-O reads "
            "the separately returned clean gt_depth_m only at selected pixels."
        )
    if bool(getattr(cfgs, "kview_use_collision", False)):
        raise RuntimeError("P1-O CVA-CDF has no learned collision head.")
    if bool(getattr(cfgs, "use_top4_view_infer", False)):
        raise RuntimeError(
            "P1-O mechanism diagnostic is defined under controlled Top-1 view "
            "inference; remove --use_top4_view_infer."
        )
    if not cfgs.save_dir:
        raise ValueError("--save_dir is required.")
    if not cfgs.test_mode:
        raise ValueError("--test_mode is required.")

    checkpoint, state = base._read_checkpoint(cfgs.checkpoint_path)
    stage = base._resolve_distill_stage(checkpoint)
    if stage != 1:
        raise RuntimeError(
            f"P1-O requires a controlled Stage-1 RGB checkpoint, got stage={stage}."
        )
    geometry_source, pose_depth_mode, checkpoint_fuse = (
        base._validate_checkpoint_contract(checkpoint, stage)
    )
    if geometry_source != "pred":
        raise RuntimeError("P1-O Stage-1 dense geometry must remain predicted-depth.")
    if bool(checkpoint.get("p1_center_recenter", False)):
        raise RuntimeError(
            "P1-O should start from the unmodified Stage-1 baseline checkpoint, "
            "not a learned P1 recenter checkpoint."
        )

    requested_fuse = bool(getattr(cfgs, "use_fuse_depth", False))
    if requested_fuse != bool(checkpoint_fuse):
        raise RuntimeError(
            "--use_fuse_depth must match checkpoint metadata: checkpoint="
            f"{int(bool(checkpoint_fuse))}, requested={int(requested_fuse)}."
        )
    requested_pose = str(getattr(cfgs, "pose_depth_mode", "none") or "none")
    if requested_pose != str(pose_depth_mode):
        raise RuntimeError(
            "--pose_depth_mode must match checkpoint metadata: checkpoint="
            f"{pose_depth_mode!r}, requested={requested_pose!r}."
        )

    os.makedirs(cfgs.save_dir, exist_ok=True)
    baseline_root = os.path.join(cfgs.save_dir, "baseline")
    oracle_root = os.path.join(cfgs.save_dir, "gt_center")
    os.makedirs(baseline_root, exist_ok=True)
    os.makedirs(oracle_root, exist_ok=True)

    full_dataset = base.GraspNetMultiDataset(
        cfgs.dataset_root,
        split=cfgs.test_mode,
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        use_fuse_depth=requested_fuse,
        graspness_mode=cfgs.graspness_mode,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
    )
    eval_dataset, sampled_indices = base._build_subset(
        full_dataset,
        float(getattr(cfgs, "sample_interval", 1.0)),
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=cfgs.num_workers,
        worker_init_fn=base._worker_init,
        collate_fn=base.collate_fn,
        pin_memory=False,
        persistent_workers=(cfgs.num_workers > 0),
    )
    scene_list = full_dataset.scene_list()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SameQueryGTCenterOracleStudent(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_cdf=True,
        use_obs_depth=False,
        pose_depth_mode=pose_depth_mode,
        camera_pose_key=str(checkpoint.get("camera_pose_key", "camera_pose_vec")),
        camera_gravity_key=str(
            checkpoint.get("camera_gravity_key", "camera_gravity_vec")
        ),
        pose_hidden_dim=int(checkpoint.get("pose_hidden_dim", 64)),
        ray_gravity_hidden_dim=int(
            checkpoint.get("ray_gravity_hidden_dim", 64)
        ),
        ray_gravity_mid_dim=int(checkpoint.get("ray_gravity_mid_dim", 32)),
        vis_dir=getattr(cfgs, "vis_dir", None),
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    ).to(device)
    base._load_checkpoint_strict(model, state)
    model.eval()

    print(
        "[P1-O] same-query GT-center oracle: "
        f"split={cfgs.test_mode} total={len(full_dataset)} "
        f"selected={len(eval_dataset)} batch={cfgs.batch_size} "
        f"sample_interval={float(getattr(cfgs, 'sample_interval', 1.0)):g}",
        flush=True,
    )
    print(
        "[P1-O] intervention=selected sparse center z only; "
        "seed_pixel=student image-FPS; dense_geometry=predicted; "
        "downstream ViewNet/CVA/CDF=recomputed; GT view/pose labels=none; "
        f"collision_thresh={float(cfgs.collision_thresh):g}",
        flush=True,
    )

    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    processed = 0
    processed_batches = 0
    role_checked = False
    start = time.perf_counter()

    for batch_idx, batch in enumerate(dataloader):
        if int(P1O_ARGS.p1o_max_batches) > 0 and batch_idx >= int(
            P1O_ARGS.p1o_max_batches
        ):
            break

        pristine = base._move_fixed_inputs(batch, device)
        if "gt_depth_m" not in pristine:
            raise RuntimeError(
                "Dataset did not return gt_depth_m required by P1-O."
            )

        with torch.inference_mode():
            model.p1o_use_gt_center = False
            baseline_end = model(
                _fresh_input(pristine, include_gt_depth=False)
            )
            model.p1o_use_gt_center = True
            oracle_end = model(
                _fresh_input(pristine, include_gt_depth=True)
            )
            model.p1o_use_gt_center = False

            if not role_checked:
                # Both branches must retain the Stage-1 *dense* predicted-depth
                # geometry role. GT is privileged only at sparse selected centers.
                base._assert_inference_geometry_role(baseline_end, "pred")
                base._assert_inference_geometry_role(oracle_end, "pred")
                active = float(
                    _require_tensor(oracle_end, "D: P1O GT center active").item()
                )
                if abs(active - 1.0) > float(P1O_ARGS.p1o_assert_atol):
                    raise RuntimeError("P1-O oracle switch did not activate.")
                role_checked = True

            pair_metrics = _assert_same_query(baseline_end, oracle_end)
            _accumulate_pair_metrics(metric_sums, metric_counts, pair_metrics)
            _accumulate_oracle_metrics(metric_sums, metric_counts, oracle_end)

            baseline_preds = pred_decode_center_view_angle(
                baseline_end, use_cdf=True
            )
            oracle_preds = pred_decode_center_view_angle(
                oracle_end, use_cdf=True
            )

        if len(baseline_preds) != len(oracle_preds):
            raise RuntimeError("Baseline/oracle decoded batch sizes differ.")

        for sample_i, (base_pred, oracle_pred) in enumerate(
            zip(baseline_preds, oracle_preds)
        ):
            subset_idx = batch_idx * cfgs.batch_size + sample_i
            if subset_idx >= len(sampled_indices):
                raise IndexError(
                    f"Subset index {subset_idx} exceeds {len(sampled_indices)}."
                )
            data_idx = int(sampled_indices[subset_idx])
            scene_name = scene_list[data_idx]
            frame_id = data_idx % 256

            base_gg = GraspGroup(base_pred.detach().cpu().numpy())
            oracle_gg = GraspGroup(oracle_pred.detach().cpu().numpy())

            if cfgs.save_nocollision:
                _save_grasp_group(
                    base_gg,
                    root=baseline_root + "_nocollision",
                    scene_name=scene_name,
                    camera=cfgs.camera,
                    frame_id=frame_id,
                )
                _save_grasp_group(
                    oracle_gg,
                    root=oracle_root + "_nocollision",
                    scene_name=scene_name,
                    camera=cfgs.camera,
                    frame_id=frame_id,
                )

            if cfgs.collision_thresh > 0:
                cloud, _ = full_dataset.get_data(
                    data_idx, return_raw_cloud=True
                )
                detector = base.ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=cfgs.collision_voxel_size,
                )
                base_collision = detector.detect(
                    base_gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                oracle_collision = detector.detect(
                    oracle_gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                base_gg = base_gg[~base_collision.detach().cpu().numpy()]
                oracle_gg = oracle_gg[
                    ~oracle_collision.detach().cpu().numpy()
                ]

            _save_grasp_group(
                base_gg,
                root=baseline_root,
                scene_name=scene_name,
                camera=cfgs.camera,
                frame_id=frame_id,
            )
            _save_grasp_group(
                oracle_gg,
                root=oracle_root,
                scene_name=scene_name,
                camera=cfgs.camera,
                frame_id=frame_id,
            )
            processed += 1

        processed_batches += 1
        if batch_idx % 20 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[P1-O] batch={batch_idx}/{len(dataloader)} "
                f"samples={processed}/{len(eval_dataset)} "
                f"sec_per_sample={elapsed / max(processed, 1):.3f}",
                flush=True,
            )

    q_total = float(metric_sums.get("query_total", 0.0))
    q_valid = float(metric_sums.get("query_valid", 0.0))
    diagnostics = {
        "query_gt_valid_ratio": (
            q_valid / q_total if q_total > 0 else float("nan")
        ),
        "base_center_z_mae_m": _mean_metric(
            metric_sums, metric_counts, "base_z_abs_error_m"
        ),
        "oracle_center_z_mae_m": _mean_metric(
            metric_sums, metric_counts, "oracle_z_abs_error_m"
        ),
        "base_center_xyz_mae_m": _mean_metric(
            metric_sums, metric_counts, "base_xyz_error_m"
        ),
        "oracle_center_xyz_mae_m": _mean_metric(
            metric_sums, metric_counts, "oracle_xyz_error_m"
        ),
        "center_displacement_abs_mean_m": _mean_metric(
            metric_sums, metric_counts, "center_displacement_abs_m"
        ),
        "base_seed_idx_mismatch_ratio": _mean_metric(
            metric_sums, metric_counts, "kview_base_token_sel_idx_mismatch_ratio"
        ),
        "query_pixel_idx_mismatch_ratio": _mean_metric(
            metric_sums, metric_counts, "token_sel_idx_mismatch_ratio"
        ),
        "view_index_change_ratio": _mean_metric(
            metric_sums, metric_counts, "view_index_change_ratio"
        ),
        "view_angle_change_deg_mean": _mean_metric(
            metric_sums, metric_counts, "view_angle_change_deg_mean"
        ),
    }

    expected_samples = len(eval_dataset)
    complete_subset = processed == expected_samples
    if int(P1O_ARGS.p1o_max_batches) == 0 and not complete_subset:
        raise RuntimeError(
            f"P1-O expected {expected_samples} samples but saved {processed}."
        )

    summary = {
        "experiment": "p1o_same_query_gt_center_v1",
        "checkpoint": os.path.abspath(str(cfgs.checkpoint_path)),
        "distill_stage": 1,
        "test_mode": str(cfgs.test_mode),
        "camera": str(cfgs.camera),
        "sample_interval": float(getattr(cfgs, "sample_interval", 1.0)),
        "selected_samples_expected": int(expected_samples),
        "selected_samples_processed": int(processed),
        "processed_batches": int(processed_batches),
        "complete_sampled_split": bool(complete_subset),
        "baseline_output_dir": os.path.abspath(baseline_root),
        "gt_center_output_dir": os.path.abspath(oracle_root),
        "protocol": {
            "seed_pixel": "native Stage-1 deterministic image-FPS",
            "query_pixel_exactly_shared": True,
            "dense_depth_geometry": "native RGB-predicted metric depth",
            "sparse_center_intervention": "same-pixel z_pred -> clean z_gt",
            "invalid_gt_fallback": "native predicted sparse center",
            "approach_view": "native Stage-1 ViewNet recomputed after center intervention",
            "cdf_width_tail": "native Stage-1 CVA-CDF",
            "gt_view_or_grasp_labels_injected": False,
            "legacy_dataset_use_gt_depth": False,
            "observed_depth_used_by_network": False,
            "dexnet_or_graspnet_evaluator_inside_inference": False,
            "top4": False,
            "collision_thresh": float(cfgs.collision_thresh),
            "collision_voxel_size": float(cfgs.collision_voxel_size),
            "use_fuse_depth": bool(requested_fuse),
            "pose_depth_mode": str(pose_depth_mode),
        },
        "diagnostics": diagnostics,
    }
    _save_json(
        summary,
        os.path.join(cfgs.save_dir, P1O_ARGS.p1o_summary_filename),
    )

    print(
        "[P1-O] done: valid_gt={:.4f} base_z_mae={:.3f}mm "
        "oracle_z_mae={:.6f}mm view_change={:.4f} view_delta={:.2f}deg".format(
            diagnostics["query_gt_valid_ratio"],
            1000.0 * diagnostics["base_center_z_mae_m"],
            1000.0 * diagnostics["oracle_center_z_mae_m"],
            diagnostics["view_index_change_ratio"],
            diagnostics["view_angle_change_deg_mean"],
        ),
        flush=True,
    )
    print(
        f"[P1-O] evaluate paired dirs externally: baseline={baseline_root} "
        f"gt_center={oracle_root}",
        flush=True,
    )


if __name__ == "__main__":
    inference()
