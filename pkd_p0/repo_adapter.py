"""Runtime adapter for source-free PKD P0 experiments.

Only the dedicated constructors in ``models.economicgrasp_dpt_p0`` are used.
The current EconomicGrasp model sources and checkpoint state dictionaries remain
unchanged.
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import os
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .common import (
    CheckpointContract,
    ContractError,
    clone_batch,
    filtered_kwargs,
    load_current_checkpoint,
    move_tensors,
    resolve_tensor,
    seed_everything,
    temporary_attributes,
)


DEPTH_USED_ALIASES = (
    "depth_map_used_for_geometry",
    "geometry_depth",
    "depth_for_geometry",
)
DEPTH_PRED_ALIASES = (
    "depth_net_pred",
    "metric_depth_pred",
    "depth_pred_metric",
    "pred_depth",
)
CDF_LOGIT_ALIASES = (
    "grasp_cdf_pred_angle_depth",
    "grasp_cdf_logits_angle_depth",
    "grasp_cdf_logits",
)
WIDTH_ALIASES = (
    "grasp_width_pred_angle_depth",
    "grasp_width_pred",
)
CENTER_ALIASES = ("xyz_graspable", "grasp_center_xyz")
VIEW_XYZ_ALIASES = ("grasp_top_view_xyz", "top_view_xyz")
VIEW_INDEX_ALIASES = (
    "grasp_top_view_inds",
    "top_view_inds",
    "top_view_indices",
)
TOKEN_INDEX_ALIASES = ("token_sel_idx", "seed_indices", "seed_idx")


class RepoImports:
    """Lazy holder; P0 entry scripts clear argv before constructing it."""

    def __init__(self) -> None:
        dataset_mod = importlib.import_module("dataset.graspnet_dataset")
        cdf_adapter_mod = importlib.import_module("dataset.cdf_label_adapter")
        p0_model_mod = importlib.import_module("models.economicgrasp_dpt_p0")
        args_mod = importlib.import_module("utils.arguments")
        decode_mod = importlib.import_module("models.economicgrasp_bip3d")
        collision_mod = importlib.import_module("utils.collision_detector")
        graspnet_mod = importlib.import_module("graspnetAPI")

        self.p0_model_module = "models.economicgrasp_dpt_p0"
        self.GraspNetMultiDataset = getattr(
            dataset_mod, "GraspNetMultiDataset"
        )
        self.CVAExtendedLabelAdapter = getattr(
            cdf_adapter_mod, "CVAExtendedLabelAdapter"
        )
        self.collate_fn = getattr(dataset_mod, "collate_fn")
        self.economicgrasp_dpt_student = getattr(
            p0_model_mod, "economicgrasp_dpt_p0_student"
        )
        self.economicgrasp_dpt_teacher = getattr(
            p0_model_mod, "economicgrasp_dpt_p0_teacher"
        )
        self.load_checkpoint_state = getattr(
            p0_model_mod, "load_checkpoint_state", None
        )
        self.enable_p0_exact_query_runtime = getattr(
            p0_model_mod, "enable_p0_exact_query_runtime"
        )
        self.build_p0_exact_query_input = getattr(
            p0_model_mod, "build_p0_exact_query_input"
        )
        self.assert_p0_exact_query_output = getattr(
            p0_model_mod, "assert_p0_exact_query_output"
        )
        self.forward_with_p0_geometry_override = getattr(
            p0_model_mod, "forward_with_p0_geometry_override"
        )
        self.extract_p0_query_contract = getattr(
            p0_model_mod, "extract_p0_query_contract"
        )
        self.p0_geometry_marker_key = getattr(
            p0_model_mod, "P0_GEOMETRY_MARKER"
        )
        self.p0_query_marker_key = getattr(
            p0_model_mod, "P0_QUERY_MARKER"
        )
        self.p0_runtime_contract = getattr(
            p0_model_mod, "runtime_contract"
        )()
        self.cfgs = getattr(args_mod, "cfgs")
        self.pred_decode_center_view_angle = getattr(
            decode_mod, "pred_decode_center_view_angle"
        )
        self.ModelFreeCollisionDetectorTorch = getattr(
            collision_mod, "ModelFreeCollisionDetectorTorch"
        )
        self.GraspGroup = getattr(graspnet_mod, "GraspGroup")


# The current CDF matcher intentionally keeps the variable-size object payload
# on CPU and copies only query-referenced rows to the active CUDA device.  This
# mirrors train_cva_distill_ddp.py and prevents DDP/DataLoader-style recursive
# transfers from materializing the full extended cache on the GPU.
CVA_COMMON_CPU_LABEL_KEYS = {
    "object_poses_list",
    "grasp_points_list",
    "view_graspness_list",
    "top_view_index_list",
}
CVA_CDF_CPU_LABEL_KEYS = {
    "grasp_cdf_bins_list",
    "grasp_widths_depth_list",
    "grasp_width_valids_depth_list",
}
CVA_CPU_RESIDENT_LABEL_LIST_KEYS = (
    CVA_COMMON_CPU_LABEL_KEYS | CVA_CDF_CPU_LABEL_KEYS
)
CVA_REQUIRED_CDF_BATCH_KEYS = (
    CVA_CPU_RESIDENT_LABEL_LIST_KEYS | {"cdf_thresholds"}
)


def validate_cdf_batch_label_contract(batch: Mapping[str, Any]) -> None:
    """Fail before model forward when the current extended-CDF payload is absent.

    The bare GraspNetMultiDataset does not attach the compact CDF/depth-wise
    width payload used by process_grasp_labels_cdf_width.  The batch must come
    from CVAExtendedLabelAdapter(use_cdf=True).
    """
    missing = sorted(key for key in CVA_REQUIRED_CDF_BATCH_KEYS if key not in batch)
    if missing:
        raise KeyError(
            "P0 paired-query collection requires the current extended CDF "
            "label adapter; batch is missing: " + ", ".join(missing)
        )

    poses = batch["object_poses_list"]
    if not isinstance(poses, (list, tuple)) or len(poses) <= 0:
        raise TypeError(
            "object_poses_list must be a non-empty batch list produced by the "
            "repository collate_fn."
        )
    batch_size = len(poses)
    for key in sorted(CVA_CPU_RESIDENT_LABEL_LIST_KEYS):
        value = batch[key]
        if not isinstance(value, (list, tuple)) or len(value) != batch_size:
            raise TypeError(
                f"{key} must be a batch list of length {batch_size}, got "
                f"{type(value).__name__}."
            )
        for batch_i, sample in enumerate(value):
            if not isinstance(sample, (list, tuple)):
                raise TypeError(
                    f"{key}[{batch_i}] must be an object list, got "
                    f"{type(sample).__name__}."
                )
            if len(sample) != len(poses[batch_i]):
                raise RuntimeError(
                    f"{key}[{batch_i}] contains {len(sample)} objects, but "
                    f"object_poses_list contains {len(poses[batch_i])}."
                )
            for obj_i, tensor in enumerate(sample):
                if not torch.is_tensor(tensor):
                    raise TypeError(
                        f"{key}[{batch_i}][{obj_i}] must be a tensor."
                    )
                if tensor.device.type != "cpu":
                    raise RuntimeError(
                        f"{key}[{batch_i}][{obj_i}] must remain CPU-resident "
                        f"before CDF label matching; got {tensor.device}."
                    )

    thresholds = batch["cdf_thresholds"]
    if not torch.is_tensor(thresholds):
        raise TypeError("cdf_thresholds must be a collated torch.Tensor.")
    if thresholds.shape[-1] != 6:
        raise RuntimeError(
            f"cdf_thresholds must end in six thresholds, got {tuple(thresholds.shape)}."
        )


def _move_batch_preserving_cdf_labels(
    batch: MutableMapping[str, Any],
    device: torch.device,
) -> MutableMapping[str, Any]:
    """Move fixed-size inputs while preserving variable-size CDF lists on CPU."""
    resident = {
        key: batch.pop(key)
        for key in list(CVA_CPU_RESIDENT_LABEL_LIST_KEYS)
        if key in batch
    }
    move_tensors(batch, device)
    batch.update(resident)
    return batch


class DeterministicSubset(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, indices: Sequence[int], seed: int) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> Any:
        index = self.indices[int(local_index)]
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            seed_everything(self.seed + index * 1_000_003)
            return self.dataset[index]
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)


def dataset_index_records(dataset: Any) -> List[Tuple[int, int, int]]:
    """Return ``(dataset_index, absolute_scene_id, annotation_id)``."""
    if not hasattr(dataset, "scene_list"):
        raise ContractError(
            "GraspNetMultiDataset no longer exposes scene_list()."
        )
    scene_names = dataset.scene_list()
    records: List[Tuple[int, int, int]] = []
    for index, scene_name in enumerate(scene_names):
        scene_id = int(str(scene_name).split("_")[-1])
        records.append((int(index), scene_id, int(index % 256)))
    return records


def build_dataset(
    repo: RepoImports,
    *,
    dataset_root: str,
    split: str,
    camera: str,
    num_point: int,
    min_depth: float,
    max_depth: float,
    bin_num: int,
    use_fuse_depth: bool,
    graspness_mode: str,
    load_label: bool = False,
    use_gt_depth: bool = False,
    use_cdf: bool = False,
    cva_label_folder: str = "",
    num_angle: Optional[int] = None,
    num_depth: Optional[int] = None,
) -> Any:
    """Construct the same current-CDF dataset contract as the trainer.

    For CDF supervision, the base dataset owns image/depth inputs and object
    poses only. CVAExtendedLabelAdapter is the sole owner of the common
    extended-angle cache. This is the exact routing used by
    train_cva_distill_ddp.py.
    """
    if bool(use_gt_depth):
        raise ContractError(
            "P0 uses the corrected contract: the legacy dataset "
            "use_gt_depth switch must remain disabled."
        )

    cdf_mode = bool(load_label and use_cdf)
    kwargs = {
        "root": dataset_root,
        "dataset_root": dataset_root,
        "split": split,
        "camera": camera,
        "num_points": int(num_point),
        "num_point": int(num_point),
        "remove_outlier": True,
        "augment": False,
        "load_label": bool(load_label),
        "use_gt_depth": False,
        "use_fuse_depth": bool(use_fuse_depth),
        "graspness_mode": graspness_mode,
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "bin_num": int(bin_num),
    }
    if cdf_mode:
        # Match the current Stage-0/1/2 trainer exactly. The base dataset must
        # expose per-frame object poses while avoiding a second object-cache
        # read; the adapter attaches the compact CDF and depth-wise widths.
        kwargs.update(
            {
                "depth_strides": 1,
                "extend_angle": True,
                "load_grasp_payload": False,
            }
        )

    signature = inspect.signature(repo.GraspNetMultiDataset)
    if cdf_mode:
        required = ("extend_angle", "load_grasp_payload", "depth_strides")
        missing_signature = [name for name in required if name not in signature.parameters]
        if missing_signature:
            raise ContractError(
                "The current P0 collector requires the cleaned CVA dataset "
                f"contract, but GraspNetMultiDataset lacks {missing_signature}."
            )

    if "root" in signature.parameters:
        base_dataset = repo.GraspNetMultiDataset(
            **filtered_kwargs(repo.GraspNetMultiDataset, kwargs)
        )
    else:
        selected = filtered_kwargs(repo.GraspNetMultiDataset, kwargs)
        selected.pop("dataset_root", None)
        base_dataset = repo.GraspNetMultiDataset(dataset_root, **selected)

    if not cdf_mode:
        return base_dataset

    folder = str(cva_label_folder or "").strip()
    if not folder:
        folder = str(getattr(repo.cfgs, "cva_label_folder", "") or "").strip()
    if not folder:
        folder = str(
            os.environ.get(
                "CVA_LABEL_FOLDER",
                os.environ.get(
                    "CDF_LABEL_FOLDER",
                    "economic_grasp_label_300views_extend_angle_cdf_depth",
                ),
            )
        ).strip()
    if not folder:
        raise ContractError("The current extended CDF label folder is empty.")

    angle_count = int(
        getattr(repo.cfgs, "num_angle", 12)
        if num_angle is None
        else num_angle
    )
    depth_count = int(
        getattr(repo.cfgs, "num_depth", 4)
        if num_depth is None
        else num_depth
    )
    dataset = repo.CVAExtendedLabelAdapter(
        base_dataset,
        dataset_root=dataset_root,
        use_cdf=True,
        label_folder=folder,
        num_angle=angle_count,
        num_depth=depth_count,
    )
    # Explicit markers used by the collector's provenance record.
    dataset.pkd_p0_cdf_label_folder = folder
    dataset.pkd_p0_label_adapter = "CVAExtendedLabelAdapter"
    return dataset


def build_current_model(
    repo: RepoImports,
    *,
    checkpoint_path: str,
    device: torch.device,
    min_depth: float,
    max_depth: float,
    bin_num: int,
    is_training: bool,
) -> Tuple[torch.nn.Module, Mapping[str, Any], CheckpointContract]:
    checkpoint, contract = load_current_checkpoint(checkpoint_path)
    constructor = (
        repo.economicgrasp_dpt_teacher
        if contract.distill_stage == 0
        else repo.economicgrasp_dpt_student
    )
    kwargs = {
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "bin_num": int(bin_num),
        "is_training": bool(is_training),
        "use_cdf": True,
        "use_obs_depth": False,
        "pose_depth_mode": contract.pose_depth_mode,
        "camera_pose_key": str(
            checkpoint.get("camera_pose_key", "camera_pose_vec")
        ),
        "camera_gravity_key": str(
            checkpoint.get("camera_gravity_key", "camera_gravity_vec")
        ),
        "pose_hidden_dim": int(checkpoint.get("pose_hidden_dim", 64)),
        "ray_gravity_hidden_dim": int(
            checkpoint.get("ray_gravity_hidden_dim", 64)
        ),
        "ray_gravity_mid_dim": int(
            checkpoint.get("ray_gravity_mid_dim", 32)
        ),
        "vis_dir": None,
    }

    try:
        model = constructor(**filtered_kwargs(constructor, kwargs)).to(device)
        if repo.load_checkpoint_state is not None:
            loader_kwargs = {
                "model": model,
                "checkpoint_path": contract.path,
                "path": contract.path,
                "strict": True,
                "checkpoint_data": checkpoint,
            }
            signature = inspect.signature(repo.load_checkpoint_state)
            selected = filtered_kwargs(
                repo.load_checkpoint_state, loader_kwargs
            )
            positional: List[Any] = []
            if "model" not in signature.parameters:
                positional.append(model)
            if (
                "checkpoint_path" not in signature.parameters
                and "path" not in signature.parameters
            ):
                positional.append(contract.path)
            repo.load_checkpoint_state(*positional, **selected)
        else:
            model.load_state_dict(
                checkpoint["model_state_dict"], strict=True
            )
        repo.enable_p0_exact_query_runtime(model)
        model.train(bool(is_training))
        if not is_training:
            model.requires_grad_(False)
        return model, checkpoint, contract
    except Exception as exc:
        raise ContractError(
            f"Dedicated P0 constructor {constructor.__name__} could not "
            f"strictly load {contract.path}: {exc!r}"
        ) from exc


@contextlib.contextmanager
def model_stage_context(
    repo: RepoImports,
    contract: CheckpointContract,
) -> Iterator[None]:
    updates = {
        "use_cdf": True,
        "use_obs_depth": False,
        # gt_depth_m remains separately available to the Stage-0 teacher.
        "use_gt_depth": False,
        "use_fuse_depth": bool(contract.use_fuse_depth),
        "pose_depth_mode": contract.pose_depth_mode,
    }
    with temporary_attributes(repo.cfgs, **updates):
        yield


def forward_model(
    repo: RepoImports,
    model: torch.nn.Module,
    contract: CheckpointContract,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int,
    geometry_override: Optional[torch.Tensor] = None,
    forced_query: Optional[Mapping[str, Any]] = None,
    require_override_marker: bool = False,
    force_process_grasp_labels: Optional[bool] = None,
    compute_cdf_diagnostics: bool = False,
    compute_geometry_diagnostics: bool = False,
) -> MutableMapping[str, Any]:
    """Run a current checkpoint through the source-free P0 runtime."""
    local = clone_batch(batch)
    _move_batch_preserving_cdf_labels(local, device)

    if forced_query is not None:
        local = repo.build_p0_exact_query_input(
            local,
            forced_query,
            force_process_grasp_labels=(
                True
                if force_process_grasp_labels is None
                else bool(force_process_grasp_labels)
            ),
        )
    elif force_process_grasp_labels is not None:
        local["cva_force_process_grasp_labels"] = bool(
            force_process_grasp_labels
        )

    local["cva_compute_diagnostics"] = bool(compute_cdf_diagnostics)
    local["geometry_compute_diagnostics"] = bool(
        compute_geometry_diagnostics
    )
    local["cva_export_angle_feature"] = False

    seed_everything(seed)
    with model_stage_context(repo, contract), torch.set_grad_enabled(
        model.training
    ):
        output = repo.forward_with_p0_geometry_override(
            model,
            local,
            geometry_override,
        )
    if not isinstance(output, MutableMapping):
        raise TypeError(
            "Current model forward must return a mapping, got "
            f"{type(output).__name__}."
        )

    if geometry_override is not None:
        resolve_tensor([output], DEPTH_USED_ALIASES, required=True)
        if (
            require_override_marker
            and repo.p0_geometry_marker_key not in output
        ):
            raise ContractError(
                "Dedicated P0 geometry runtime marker is absent."
            )

    if forced_query is not None:
        output.update(
            repo.assert_p0_exact_query_output(forced_query, output)
        )
        if require_override_marker and repo.p0_query_marker_key not in output:
            raise ContractError(
                "Dedicated P0 exact-query runtime marker is absent."
            )
    return output


def extract_query_override(output: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract exact image pixels/views; never copy student 3D centers."""
    return repo_extract_p0_query_contract(output)


def repo_extract_p0_query_contract(
    output: Mapping[str, Any],
) -> Dict[str, Any]:
    # Local import avoids triggering EconomicGrasp's CLI parser when this module
    # is imported only for utility functions.
    module = importlib.import_module("models.economicgrasp_dpt_p0")
    return getattr(module, "extract_p0_query_contract")(output)

def extract_core_outputs(output: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for canonical, aliases in {
        "cdf_logits": CDF_LOGIT_ALIASES,
        "width": WIDTH_ALIASES,
        "centers": CENTER_ALIASES,
        "view_xyz": VIEW_XYZ_ALIASES,
        "view_indices": VIEW_INDEX_ALIASES,
        "token_indices": TOKEN_INDEX_ALIASES,
        "geometry_depth": DEPTH_USED_ALIASES,
        "predicted_depth": DEPTH_PRED_ALIASES,
    }.items():
        _, tensor = resolve_tensor([output], aliases, required=canonical not in {"token_indices", "predicted_depth"})
        if tensor is not None:
            result[canonical] = tensor
    return result


def decode_current(repo: RepoImports, output: Mapping[str, Any]) -> Sequence[torch.Tensor]:
    kwargs = {
        "end_points": output,
        "use_cdf": True,
        "batch_viewpoint_params_to_matrix_fn": None,
    }
    signature = inspect.signature(repo.pred_decode_center_view_angle)
    if "end_points" in signature.parameters:
        decoded = repo.pred_decode_center_view_angle(**filtered_kwargs(repo.pred_decode_center_view_angle, kwargs))
    else:
        selected = filtered_kwargs(repo.pred_decode_center_view_angle, kwargs)
        selected.pop("end_points", None)
        decoded = repo.pred_decode_center_view_angle(output, **selected)
    if torch.is_tensor(decoded):
        return [row for row in decoded]
    return decoded


def find_point_cloud(batch: Mapping[str, Any], batch_index: int = 0) -> np.ndarray:
    aliases = ("point_clouds", "point_cloud", "cloud", "cloud_xyz", "raw_point_cloud")
    for key in aliases:
        value = batch.get(key)
        if torch.is_tensor(value):
            array = value[batch_index] if value.ndim == 3 else value
            return array.detach().cpu().numpy().astype(np.float32)
        if isinstance(value, list) and value and torch.is_tensor(value[batch_index]):
            return value[batch_index].detach().cpu().numpy().astype(np.float32)
        if isinstance(value, np.ndarray):
            return (value[batch_index] if value.ndim == 3 else value).astype(np.float32)
    raise KeyError(f"No point-cloud key found in batch; available keys={sorted(batch)}")


def postprocess_grasps(
    repo: RepoImports,
    grasp_rows: np.ndarray,
    *,
    point_cloud: Optional[np.ndarray],
    collision_thresh: float,
    collision_voxel_size: float,
    approach_dist: float = 0.05,
    apply_nms: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    rows = np.asarray(grasp_rows, dtype=np.float32)
    rows = rows[np.isfinite(rows).all(axis=1)]
    before = len(rows)
    group = repo.GraspGroup(rows)
    collision_removed = 0
    if collision_thresh > 0 and len(group) > 0:
        if point_cloud is None:
            raise ValueError("point_cloud is required when collision_thresh > 0")

        # Despite its name, the repository's ModelFreeCollisionDetectorTorch
        # constructor expects an ordinary CPU NumPy array.  It first passes the
        # scene points to Open3D for voxel downsampling and only then converts
        # the downsampled cloud to a CUDA tensor internally.  Passing a CUDA or
        # CPU torch.Tensor here makes Open3D's Vector3dVector conversion fail.
        if torch.is_tensor(point_cloud):
            scene_points = point_cloud.detach().cpu().numpy()
        else:
            scene_points = np.asarray(point_cloud)
        if scene_points.ndim != 2 or scene_points.shape[1] != 3:
            raise ValueError(
                "point_cloud must have shape [N,3] before model-free collision "
                f"detection, got {tuple(scene_points.shape)}"
            )
        finite = np.isfinite(scene_points).all(axis=1)
        scene_points = np.ascontiguousarray(
            scene_points[finite], dtype=np.float32
        )
        if scene_points.shape[0] == 0:
            raise ValueError(
                "point_cloud contains no finite scene points for collision "
                "detection"
            )

        detector = repo.ModelFreeCollisionDetectorTorch(
            scene_points,
            voxel_size=float(collision_voxel_size),
        )
        mask = detector.detect(
            group,
            approach_dist=float(approach_dist),
            collision_thresh=float(collision_thresh),
        )
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask, dtype=bool)
        collision_removed = int(mask.sum())
        group = group[~mask]
    after_collision = len(group)
    if apply_nms and len(group) > 0:
        nms_result = group.nms()
        if nms_result is not None:
            group = nms_result
    if hasattr(group, "sort_by_score") and len(group) > 0:
        sorted_result = group.sort_by_score()
        if sorted_result is not None:
            group = sorted_result
    array = np.asarray(group.grasp_group_array, dtype=np.float32)
    return array, {
        "before": int(before),
        "collision_removed": int(collision_removed),
        "after_collision": int(after_collision),
        "after_nms": int(len(array)),
    }
