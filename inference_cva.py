"""Strict inference for the CDF-only first-generation CVA Transformer."""

import os
import time
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup

from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from models.economicgrasp_bip3d import (
    economicgrasp_dpt,
    pred_decode_center_view_angle_cdf,
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
    for key, value in batch.items():
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Inference received unexpected list-valued key '{key}'. "
                "The test dataset must be constructed with load_label=False."
            )
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=False)
    return batch


def _load_checkpoint_strict(model, checkpoint_path: str) -> None:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"CDF model checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Strict CDF checkpoint loading produced missing/unexpected keys: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def inference() -> None:
    if not cfgs.multi_modal:
        raise RuntimeError("CDF CVA inference requires --multi_modal.")
    if bool(getattr(cfgs, "kview_use_collision", False)):
        raise RuntimeError(
            "The cleaned CDF model has no learned collision head. Remove "
            "--kview_use_collision."
        )
    if not cfgs.save_dir:
        raise ValueError("--save_dir is required for inference.")
    if not cfgs.test_mode:
        raise ValueError("--test_mode is required for inference.")

    os.makedirs(cfgs.save_dir, exist_ok=True)
    full_dataset = GraspNetMultiDataset(
        cfgs.dataset_root,
        split=cfgs.test_mode,
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
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
    model = economicgrasp_dpt(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_obs_depth=bool(getattr(cfgs, "use_obs_depth", False)),
        vis_dir=getattr(cfgs, "vis_dir", None),
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    ).to(device)
    _load_checkpoint_strict(model, cfgs.checkpoint_path)
    model.eval()

    print(f"[INFER] total={len(full_dataset)} selected={len(eval_dataset)}")
    print(
        f"[INFER] top4={bool(getattr(cfgs, 'use_top4_view_infer', False))} "
        f"batch={cfgs.batch_size} observed_depth={bool(cfgs.use_obs_depth)}"
    )

    start = time.perf_counter()
    processed = 0
    for batch_idx, batch in enumerate(dataloader):
        batch = _move_fixed_inputs(batch, device)
        batch["cva_export_angle_feature"] = False
        with torch.inference_mode():
            end_points = model(batch)
            grasp_preds = pred_decode_center_view_angle_cdf(end_points)

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
