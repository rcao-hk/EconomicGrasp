"""Strict Stage-0/1/2 inference for privileged-depth distillation.

Stage 0 is a diagnostic upper bound: it consumes RGB plus clean synthetic
``gt_depth_m`` and bypasses the DPT metric-depth decoder.  Stages 1 and 2 are
the deployable RGB-only students and execute the predicted-depth path.  All
roles use the same deterministic image-space FPS selector.
"""

import argparse
import os
import sys
import time
from typing import List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup


def _parse_distillation_inference_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--distill_stage",
        type=str,
        default="auto",
        choices=("auto", "0", "1", "2"),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


DISTILL_INFER_ARGS = _parse_distillation_inference_args()

from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import (
    DISTILL_CONTRACT_VERSION,
    economicgrasp_dpt_student,
    economicgrasp_dpt_teacher,
)


def _worker_init(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _build_subset(
    dataset,
    sample_interval: float,
    annos_per_scene: int = 256,
) -> Tuple[torch.utils.data.Dataset, List[int]]:
    if sample_interval <= 0:
        raise ValueError(
            f"sample_interval must be positive, got {sample_interval}."
        )
    total = len(dataset)
    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices

    stride = max(1, int(round(1.0 / sample_interval)))
    indices = []
    for start in range(0, total, annos_per_scene):
        end = min(start + annos_per_scene, total)
        indices.extend(range(start, end, stride))
    return Subset(dataset, indices), indices


def _move_fixed_inputs(batch, device):
    # The network path is point-cloud-free.  A captured cloud is read only by
    # the optional legacy collision post-processor when collision_thresh > 0.
    for key in (
        "point_clouds",
        "cloud_colors",
        "coordinates_for_voxel",
    ):
        batch.pop(key, None)
    for key, value in batch.items():
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Inference received unexpected list-valued key '{key}'. "
                "The test dataset must be constructed with load_label=False."
            )
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    return batch


def _read_checkpoint(checkpoint_path: str):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"CVA model checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "This inference script requires a full Stage-0/1/2 checkpoint with "
            "privileged-depth metadata; a plain or legacy state dict is unsafe."
        )
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(
            f"Checkpoint does not contain a model state dict: {checkpoint_path}"
        )
    return checkpoint, state


def _resolve_distill_stage(checkpoint) -> int:
    requested = str(DISTILL_INFER_ARGS.distill_stage)
    if "distill_stage" not in checkpoint:
        raise RuntimeError(
            "Checkpoint has no distill_stage metadata.  Retrain/export it with "
            "the corrected privileged-depth Stage-0--2 implementation."
        )
    metadata_stage = int(checkpoint["distill_stage"])
    if metadata_stage not in (0, 1, 2):
        raise RuntimeError(
            f"Invalid checkpoint distill_stage={metadata_stage}."
        )

    if requested == "auto":
        return metadata_stage

    stage = int(requested)
    if stage != metadata_stage:
        raise RuntimeError(
            f"Requested distill_stage={stage}, but checkpoint metadata says "
            f"distill_stage={metadata_stage}."
        )
    return stage


def _validate_checkpoint_contract(checkpoint, distill_stage: int):
    version = int(checkpoint.get("distill_contract_version", -1))
    if version != DISTILL_CONTRACT_VERSION:
        raise RuntimeError(
            "Checkpoint predates the clean-depth teacher correction. Expected "
            f"distill_contract_version={DISTILL_CONTRACT_VERSION}, got {version}."
        )
    if str(checkpoint.get("seed_selection_mode", "")) != "image_fps":
        raise RuntimeError(
            "Inference requires seed_selection_mode='image_fps'; got "
            f"{checkpoint.get('seed_selection_mode')!r}."
        )

    expected_source = "gt" if distill_stage == 0 else "pred"
    saved_source = str(checkpoint.get("geometry_depth_source", ""))
    if saved_source != expected_source:
        raise RuntimeError(
            f"Stage {distill_stage} requires geometry_depth_source="
            f"{expected_source!r}, got {saved_source!r}."
        )

    expected_head = expected_source == "pred"
    if bool(checkpoint.get("depth_head_executed", not expected_head)) != expected_head:
        raise RuntimeError(
            f"Stage {distill_stage} has inconsistent depth_head_executed metadata."
        )

    saved_pose_mode = str(checkpoint.get("pose_depth_mode", ""))
    if distill_stage == 0 and saved_pose_mode != "none":
        raise RuntimeError(
            "Stage-0 privileged teacher must use pose_depth_mode='none'; got "
            f"{saved_pose_mode!r}."
        )
    if distill_stage == 2 and str(
        checkpoint.get("teacher_geometry_depth_source", "")
    ) != "gt":
        raise RuntimeError(
            "Stage-2 checkpoint does not record a clean-depth teacher."
        )
    if bool(checkpoint.get("legacy_dataset_use_gt_depth", True)):
        raise RuntimeError(
            "Checkpoint used the legacy dataset --use_gt_depth switch, which "
            "changes crops/labels and violates the controlled experiment."
        )

    if "use_fuse_depth" not in checkpoint:
        raise RuntimeError("Checkpoint has no use_fuse_depth metadata.")
    return expected_source, saved_pose_mode, bool(checkpoint["use_fuse_depth"])


def _load_checkpoint_strict(model, state) -> None:
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
    if missing or unexpected:
        raise RuntimeError(
            "Strict CVA checkpoint loading produced missing/unexpected keys: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _assert_inference_geometry_role(end_points, expected_source: str) -> None:
    required = (
        "D: Geometry depth source GT",
        "D: Depth head executed",
        "depth_map_used_for_geometry",
    )
    missing = [key for key in required if key not in end_points]
    if missing:
        raise RuntimeError(
            f"Inference geometry contract is missing endpoint(s): {missing}."
        )

    source_is_gt = bool(round(float(
        end_points["D: Geometry depth source GT"].detach().item()
    )))
    head_executed = bool(round(float(
        end_points["D: Depth head executed"].detach().item()
    )))
    expect_gt = expected_source == "gt"
    if source_is_gt != expect_gt or head_executed == expect_gt:
        raise RuntimeError(
            "Inference executed the wrong geometry-depth path: "
            f"expected={expected_source}, source_is_gt={source_is_gt}, "
            f"head_executed={head_executed}."
        )

    used = end_points["depth_map_used_for_geometry"]
    if expect_gt:
        if "depth_net_pred" in end_points or "depth_head_raw_pred" in end_points:
            raise RuntimeError(
                "Stage-0 inference unexpectedly executed/exported the DPT depth head."
            )
        gt = end_points.get("gt_depth_m", None)
        if gt is None:
            raise RuntimeError("Stage-0 inference has no gt_depth_m.")
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        else:
            gt = gt[:, :1]
        gt = torch.nan_to_num(gt.to(used), nan=0.0, posinf=0.0, neginf=0.0)
        if gt.shape != used.shape or float((gt - used).abs().max().item()) > 1e-6:
            raise RuntimeError(
                "Stage-0 geometry is not exactly the supplied clean gt_depth_m."
            )
    else:
        pred = end_points.get("depth_net_pred", None)
        if pred is None:
            raise RuntimeError("RGB-only student did not execute its DPT depth head.")
        if pred.shape != used.shape or float((pred - used).abs().max().item()) > 1e-6:
            raise RuntimeError(
                "RGB-only student geometry is not its predicted metric depth."
            )


def inference() -> None:
    if not cfgs.multi_modal:
        raise RuntimeError("CVA inference requires --multi_modal.")
    if bool(getattr(cfgs, "kview_use_collision", False)):
        raise RuntimeError(
            "This CVA configuration has no learned collision head. Remove "
            "--kview_use_collision."
        )
    if bool(getattr(cfgs, "use_obs_depth", False)):
        raise RuntimeError(
            "Stage-0--2 inference never consumes captured depth. Remove "
            "--use_obs_depth."
        )
    if bool(getattr(cfgs, "use_gt_depth", False)):
        raise RuntimeError(
            "Keep the legacy dataset --use_gt_depth switch disabled. Stage 0 "
            "reads gt_depth_m internally without changing RGB crops or labels."
        )
    if not cfgs.save_dir:
        raise ValueError("--save_dir is required for inference.")
    if not cfgs.test_mode:
        raise ValueError("--test_mode is required for inference.")

    use_cdf = bool(getattr(cfgs, "use_cdf", False))
    if not use_cdf:
        raise RuntimeError(
            "The distillation implementation supports only CVA-CDF. Add --use_cdf."
        )

    checkpoint, state = _read_checkpoint(cfgs.checkpoint_path)
    distill_stage = _resolve_distill_stage(checkpoint)
    (
        geometry_depth_source,
        checkpoint_pose_depth_mode,
        checkpoint_use_fuse_depth,
    ) = _validate_checkpoint_contract(checkpoint, distill_stage)

    requested_use_fuse_depth = bool(getattr(cfgs, "use_fuse_depth", False))
    if requested_use_fuse_depth != checkpoint_use_fuse_depth:
        raise RuntimeError(
            "--use_fuse_depth must match checkpoint metadata so the Stage-0 "
            "teacher receives the same clean-depth construction used in training: "
            f"checkpoint={checkpoint_use_fuse_depth}, requested="
            f"{requested_use_fuse_depth}."
        )

    os.makedirs(cfgs.save_dir, exist_ok=True)
    full_dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        split=cfgs.test_mode,
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        use_fuse_depth=requested_use_fuse_depth,
        graspness_mode=cfgs.graspness_mode,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
    )
    eval_dataset, sampled_indices = _build_subset(
        full_dataset,
        float(getattr(cfgs, "sample_interval", 1.0)),
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=cfgs.num_workers,
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(cfgs.num_workers > 0),
    )
    scene_list = full_dataset.scene_list()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_kwargs = dict(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_cdf=True,
        vis_dir=getattr(cfgs, "vis_dir", None),
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    )
    if distill_stage == 0:
        model = economicgrasp_dpt_teacher(**model_kwargs)
        model_role = "privileged_gt_depth_teacher"
        effective_pose_depth_mode = "none"
    else:
        model = economicgrasp_dpt_student(
            **model_kwargs,
            use_obs_depth=False,
            pose_depth_mode=checkpoint_pose_depth_mode,
            camera_pose_key=str(
                checkpoint.get("camera_pose_key", "camera_pose_vec")
            ),
            camera_gravity_key=str(
                checkpoint.get("camera_gravity_key", "camera_gravity_vec")
            ),
            pose_hidden_dim=int(checkpoint.get("pose_hidden_dim", 64)),
            ray_gravity_hidden_dim=int(
                checkpoint.get("ray_gravity_hidden_dim", 64)
            ),
            ray_gravity_mid_dim=int(
                checkpoint.get("ray_gravity_mid_dim", 32)
            ),
        )
        model_role = "rgb_pred_depth_student"
        effective_pose_depth_mode = checkpoint_pose_depth_mode

    model = model.to(device)
    _load_checkpoint_strict(model, state)
    model.eval()

    print(f"[INFER] total={len(full_dataset)} selected={len(eval_dataset)}")
    print(
        f"[INFER] distill_stage={distill_stage} model={model_role} "
        f"geometry_depth={geometry_depth_source} depth_head="
        f"{int(geometry_depth_source == 'pred')} pose_depth_mode="
        f"{effective_pose_depth_mode} use_fuse_depth="
        f"{int(requested_use_fuse_depth)} seed_selection=image_fps "
        f"top4={bool(getattr(cfgs, 'use_top4_view_infer', False))} "
        f"batch={cfgs.batch_size}",
        flush=True,
    )
    if distill_stage == 0:
        print(
            "[INFER] NOTE: Stage 0 consumes test-time clean synthetic depth and "
            "is a privileged teacher upper bound, not an RGB-only result.",
            flush=True,
        )

    start = time.perf_counter()
    processed = 0
    role_checked = False
    for batch_idx, batch in enumerate(dataloader):
        batch = _move_fixed_inputs(batch, device)
        batch["cva_export_angle_feature"] = False
        with torch.inference_mode():
            end_points = model(batch)
            if not role_checked:
                _assert_inference_geometry_role(
                    end_points,
                    geometry_depth_source,
                )
                role_checked = True
            grasp_preds = pred_decode_center_view_angle(
                end_points,
                use_cdf=True,
            )

        for sample_i, pred in enumerate(grasp_preds):
            subset_idx = batch_idx * cfgs.batch_size + sample_i
            if subset_idx >= len(sampled_indices):
                raise IndexError(
                    f"Subset index {subset_idx} exceeds {len(sampled_indices)}."
                )
            data_idx = sampled_indices[subset_idx]
            gg = GraspGroup(pred.detach().cpu().numpy())

            if cfgs.save_nocollision:
                out_dir = os.path.join(
                    cfgs.save_dir + "_nocollision",
                    scene_list[data_idx],
                    cfgs.camera,
                )
                os.makedirs(out_dir, exist_ok=True)
                gg.save_npy(
                    os.path.join(out_dir, f"{data_idx % 256:04d}.npy")
                )

            if cfgs.collision_thresh > 0:
                cloud, _ = full_dataset.get_data(
                    data_idx,
                    return_raw_cloud=True,
                )
                detector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=cfgs.collision_voxel_size,
                )
                collision = detector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                gg = gg[~collision.detach().cpu().numpy()]

            out_dir = os.path.join(
                cfgs.save_dir,
                scene_list[data_idx],
                cfgs.camera,
            )
            os.makedirs(out_dir, exist_ok=True)
            gg.save_npy(
                os.path.join(out_dir, f"{data_idx % 256:04d}.npy")
            )
            processed += 1

        if batch_idx % 20 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[INFER] batch={batch_idx}/{len(dataloader)} "
                f"samples={processed}/{len(eval_dataset)} "
                f"sec_per_sample={elapsed / max(processed, 1):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    inference()
