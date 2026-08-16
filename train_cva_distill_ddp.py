# Basic Libraries
import argparse
import os
import math
import sys
import time
import random
from datetime import timedelta
import numpy as np

# PyTorch Libraries
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Distillation-only arguments
# ---------------------------------------------------------------------------
#
# ``utils.arguments`` parses ``sys.argv`` during import in the full repository.
# Keep the shared parser untouched: consume only the new stage-0/1/2 flags here
# and pass all remaining arguments to the existing parser. This keeps this
# experiment self-contained and avoids changing every other training/inference
# entry point in the repository.


def _parse_distillation_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--distill_stage",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help=(
            "0: privileged clean-depth image-FPS teacher; "
            "1: RGB/predicted-depth image-FPS student with GT losses; "
            "2: the same RGB student with frozen Stage-0 E1 output KD; the "
            "student selects image-FPS seeds autonomously and the teacher "
            "reuses those exact student seeds."
        ),
    )
    parser.add_argument("--teacher_checkpoint", type=str, default="")
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--kd_objectness_weight", type=float, default=0.0)
    parser.add_argument("--kd_graspness_weight", type=float, default=0.0)
    parser.add_argument("--kd_depth_weight", type=float, default=0.0)
    parser.add_argument("--kd_view_weight", type=float, default=1.0)
    parser.add_argument("--kd_cdf_weight", type=float, default=1.0)
    parser.add_argument("--kd_width_weight", type=float, default=0.1)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--kd_max_query_view_angle_deg", type=float, default=35.0)
    parser.add_argument("--kd_width_positive_threshold", type=float, default=0.5)
    parser.add_argument(
        "--kd_diag_interval_steps",
        type=int,
        default=5000,
        help=(
            "Run paired privileged-KD diagnostics every N optimizer steps in "
            "Stage 2; 0 disables the expensive diagnostics."
        ),
    )
    parser.add_argument(
        "--kd_diag_eval_batches",
        type=int,
        default=64,
        help=(
            "Number of validation batches per epoch used for paired teacher/"
            "student diagnostics in Stage 2; 0 means all batches."
        ),
    )
    parser.add_argument(
        "--kd_diag_grad_conflict",
        type=int,
        choices=(0, 1),
        default=1,
        help="Measure supervised-vs-KD output-gradient conflict at diag steps.",
    )
    parser.add_argument(
        "--diagnose_only",
        action="store_true",
        help=(
            "Load the requested Stage-2 student and Stage-0 teacher, run one "
            "validation pass with paired privileged-KD diagnostics, and exit."
        ),
    )
    parser.add_argument(
        "--diagnose_epoch",
        type=int,
        default=0,
        help="Epoch tag used only for diagnose-only logging.",
    )

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


DISTILL_ARGS = _parse_distillation_args()

# Config shared with the original CVA trainer.
from utils.arguments import cfgs

# Local libraries.
from models.economicgrasp_dpt_distill import (
    DISTILL_CONTRACT_VERSION,
    OutputDistillationConfig,
    compute_output_distillation_loss,
    economicgrasp_dpt_student,
    economicgrasp_dpt_teacher,
    extract_distillation_targets,
    load_checkpoint_state,
)
from models.privileged_kd_diagnostics import (
    compute_output_gradient_conflict,
    compute_privileged_kd_diagnostics,
)
from models.loss_economicgrasp_depth_kview_transformer import (
    get_loss as get_loss_economicgrasp,
)
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter

# ----------- GLOBAL CONFIG ------------
EPOCH_CNT = 0
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None else None


def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def sync_print(tag, t0):
    """CUDA-synchronized timing print for debugging only."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[rank{get_rank()}][time] {tag}: {time.time() - t0:.3f}s", flush=True)
    return time.time()


def setup_distributed():
    """Initialize single-node multi-GPU distributed training via torchrun.

    Launch:
      torchrun --nproc_per_node=NUM_GPUS train_ddp.py

    Also supports plain python execution as a single-process fallback.
    """
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1

    if distributed:
        if 'RANK' not in os.environ or 'LOCAL_RANK' not in os.environ:
            raise RuntimeError(
                'Distributed mode requires RANK and LOCAL_RANK. '
                'Please launch with torchrun.'
            )
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        timeout_sec = int(os.environ.get('DDP_TIMEOUT_SEC', '3600'))
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(seconds=timeout_sec),
        )
        dist.barrier()
    else:
        rank = 0
        local_rank = 0 if torch.cuda.is_available() else -1
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        # Do not call barrier here. If one rank exits due to an error,
        # a cleanup barrier can create a second hang and hide the original issue.
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def my_worker_init_fn(worker_id):
    base_seed = np.random.get_state()[1][0]
    np.random.seed(base_seed + worker_id)


def _sync(distributed: bool):
    """Explicit global sync; do not use inside the normal training loop."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if distributed and dist.is_initialized():
        dist.barrier()


CVA_COMMON_CPU_LABEL_KEYS = {
    "object_poses_list",
    "grasp_points_list",
    "view_graspness_list",
    "top_view_index_list",
}
CVA_LEGACY_CPU_LABEL_KEYS = {
    "grasp_rotations_list",
    "grasp_depth_list",
    "grasp_scores_list",
    "grasp_widths_list",
    "grasp_collision_list",
}
CVA_CDF_CPU_LABEL_KEYS = {
    "grasp_cdf_bins_list",
    "grasp_widths_depth_list",
    "grasp_width_valids_depth_list",
}
CVA_CPU_RESIDENT_LABEL_LIST_KEYS = (
    CVA_COMMON_CPU_LABEL_KEYS
    | CVA_LEGACY_CPU_LABEL_KEYS
    | CVA_CDF_CPU_LABEL_KEYS
)

# Stage 0 obtains geometry from the separately returned clean ``gt_depth_m``;
# stages 1/2 obtain it from the RGB depth decoder. None of them reads the
# captured/sampled input point cloud. Drop these tensors before CUDA transfer so
# the experiment cannot silently regain a point-cloud dependency.
UNUSED_POINT_INPUT_KEYS = {
    "point_clouds",
    "cloud_colors",
    "coordinates_for_voxel",
}


def drop_unused_point_inputs(batch):
    for key in UNUSED_POINT_INPUT_KEYS:
        batch.pop(key, None)
    return batch


def assert_geometry_depth_contract(
    end_points,
    *,
    expected_source: str,
    context: str,
):
    """Fail fast if a stage executes the wrong geometry-depth path.

    This is intentionally called only on the first batch of an epoch/evaluation
    loop, so the scalar checks do not add a synchronization point to every
    optimization step.
    """
    expected_source = str(expected_source).strip().lower()
    if expected_source not in {"gt", "pred"}:
        raise ValueError(f"Unknown expected geometry source: {expected_source!r}")

    required = (
        "D: Geometry depth source GT",
        "D: Depth head executed",
        "depth_map_used_for_geometry",
        "depth_map_pred",
    )
    missing = [key for key in required if key not in end_points]
    if missing:
        raise RuntimeError(
            f"{context}: geometry-depth contract is missing endpoint(s): {missing}"
        )

    source_is_gt = bool(
        round(float(end_points["D: Geometry depth source GT"].detach().item()))
    )
    head_executed = bool(
        round(float(end_points["D: Depth head executed"].detach().item()))
    )
    expect_gt = expected_source == "gt"
    if source_is_gt != expect_gt:
        raise RuntimeError(
            f"{context}: expected geometry_depth_source={expected_source!r}, "
            f"but the forward endpoint reports source_is_gt={source_is_gt}."
        )
    if head_executed == expect_gt:
        raise RuntimeError(
            f"{context}: depth-head execution mismatch for source "
            f"{expected_source!r}; head_executed={head_executed}."
        )

    used = end_points["depth_map_used_for_geometry"]
    if expected_source == "gt":
        if "depth_net_pred" in end_points or "depth_head_raw_pred" in end_points:
            raise RuntimeError(
                f"{context}: privileged teacher unexpectedly exported depth-head "
                "predictions, so the decoder was not fully bypassed."
            )
        gt = end_points.get("gt_depth_m", None)
        if gt is None:
            raise RuntimeError(f"{context}: privileged teacher has no gt_depth_m.")
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        elif gt.dim() == 4:
            gt = gt[:, :1]
        else:
            raise RuntimeError(
                f"{context}: unexpected gt_depth_m shape {tuple(gt.shape)}."
            )
        gt = torch.nan_to_num(gt.to(used), nan=0.0, posinf=0.0, neginf=0.0)
        if gt.shape != used.shape:
            raise RuntimeError(
                f"{context}: GT/geometry shapes differ: "
                f"gt={tuple(gt.shape)}, used={tuple(used.shape)}."
            )
        max_error = float((used - gt).abs().max().detach().item())
        if max_error > 1e-6:
            raise RuntimeError(
                f"{context}: teacher geometry does not equal gt_depth_m "
                f"(max_abs_error={max_error:.3e})."
            )
    else:
        if "depth_net_pred" not in end_points or "depth_head_raw_pred" not in end_points:
            raise RuntimeError(
                f"{context}: RGB-only student did not export depth-head outputs."
            )
        pred = end_points["depth_net_pred"]
        if pred.shape != used.shape:
            raise RuntimeError(
                f"{context}: predicted/geometry shapes differ: "
                f"pred={tuple(pred.shape)}, used={tuple(used.shape)}."
            )
        max_error = float((used - pred).abs().max().detach().item())
        if max_error > 1e-6:
            raise RuntimeError(
                f"{context}: student geometry is not its predicted depth "
                f"(max_abs_error={max_error:.3e})."
            )


def validate_batch_label_contract(batch_data_label, use_cdf: bool):
    common = CVA_COMMON_CPU_LABEL_KEYS
    mode_keys = (
        CVA_CDF_CPU_LABEL_KEYS | {"cdf_thresholds"}
        if bool(use_cdf)
        else CVA_LEGACY_CPU_LABEL_KEYS
    )
    forbidden = (
        CVA_LEGACY_CPU_LABEL_KEYS
        if bool(use_cdf)
        else CVA_CDF_CPU_LABEL_KEYS | {"cdf_thresholds"}
    )
    missing = sorted(
        key for key in common | mode_keys
        if key not in batch_data_label
    )
    if missing:
        raise KeyError(
            f"Extended CVA dataset mode="
            f"{'CDF' if use_cdf else 'legacy'} is missing keys: "
            f"{missing}"
        )
    incompatible = sorted(
        key for key in forbidden
        if key in batch_data_label
    )
    if incompatible:
        raise RuntimeError(
            f"Extended CVA dataset mode="
            f"{'CDF' if use_cdf else 'legacy'} contains incompatible "
            f"keys: {incompatible}"
        )

    # Every object-wise list must stay aligned with object_poses_list.
    batch_size = len(batch_data_label["object_poses_list"])
    list_keys = sorted(common | (mode_keys - {"cdf_thresholds"}))
    for key in list_keys:
        value = batch_data_label[key]
        if not isinstance(value, (list, tuple)) or len(value) != batch_size:
            raise TypeError(
                f"{key} must be a batch list of length {batch_size}."
            )
    for batch_i in range(batch_size):
        num_objects = len(
            batch_data_label["object_poses_list"][batch_i]
        )
        if num_objects <= 0:
            raise RuntimeError(
                f"Batch item {batch_i} has no CVA object labels."
            )
        for key in list_keys:
            if len(batch_data_label[key][batch_i]) != num_objects:
                raise RuntimeError(
                    f"{key}[{batch_i}] has "
                    f"{len(batch_data_label[key][batch_i])} objects, "
                    f"expected {num_objects}."
                )

    num_angle = int(cfgs.num_angle)
    num_depth = int(cfgs.num_depth)
    if bool(use_cdf):
        for batch_i, sample in enumerate(
            batch_data_label["grasp_cdf_bins_list"]
        ):
            for obj_i, tensor in enumerate(sample):
                if (
                    tensor.dim() != 4
                    or tensor.shape[-2:] != (
                        num_angle,
                        num_depth,
                    )
                ):
                    raise RuntimeError(
                        "grasp_cdf_bins_list must contain "
                        f"[P,K,A,D] with A/D={num_angle}/{num_depth}; "
                        f"got {tuple(tensor.shape)} at "
                        f"[{batch_i}][{obj_i}]."
                    )
                if tensor.dtype != torch.uint8:
                    raise TypeError(
                        "CDF bins must remain uint8; got "
                        f"{tensor.dtype}."
                    )
        for key, expected_dtype in (
            ("grasp_widths_depth_list", torch.uint16),
            ("grasp_width_valids_depth_list", torch.uint8),
        ):
            for sample in batch_data_label[key]:
                for tensor in sample:
                    if (
                        tensor.dim() != 4
                        or tensor.shape[-2:] != (
                            num_angle,
                            num_depth,
                        )
                    ):
                        raise RuntimeError(
                            f"{key} must contain [P,K,A,D] labels; "
                            f"got {tuple(tensor.shape)}."
                        )
                    if tensor.dtype != expected_dtype:
                        raise TypeError(
                            f"{key} must remain {expected_dtype}; "
                            f"got {tensor.dtype}."
                        )
    else:
        for key in (
            "grasp_rotations_list",
            "grasp_depth_list",
            "grasp_scores_list",
            "grasp_widths_list",
            "grasp_collision_list",
        ):
            for sample in batch_data_label[key]:
                for tensor in sample:
                    if (
                        tensor.dim() != 3
                        or tensor.shape[-1] != num_angle
                    ):
                        raise RuntimeError(
                            f"{key} must contain extended [P,K,A] "
                            f"labels with A={num_angle}; got "
                            f"{tuple(tensor.shape)}."
                        )


def _recursive_to_device(value, device, non_blocking=False):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, list):
        return [
            _recursive_to_device(v, device, non_blocking)
            for v in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _recursive_to_device(v, device, non_blocking)
            for v in value
        )
    if isinstance(value, dict):
        return {
            k: _recursive_to_device(v, device, non_blocking)
            for k, v in value.items()
        }
    return value


def move_batch_to_device(
    batch_data_label,
    device,
    use_cdf: bool,
    non_blocking=False,
):
    """Keep all extended object payloads on CPU until label matching."""
    del use_cdf  # The CPU boundary is shared by both CVA modes.
    for key, value in list(batch_data_label.items()):
        if key in CVA_CPU_RESIDENT_LABEL_LIST_KEYS:
            continue
        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Unexpected list-valued batch key '{key}'. Add it to the "
                "explicit CVA CPU-resident contract or collate it to a tensor."
            )
        batch_data_label[key] = _recursive_to_device(
            value,
            device=device,
            non_blocking=non_blocking,
        )
    return batch_data_label


def assert_cpu_resident_label_lists(
    batch_data_label,
    use_cdf: bool,
):
    del use_cdf
    for key in CVA_CPU_RESIDENT_LABEL_LIST_KEYS:
        if key not in batch_data_label:
            continue
        value = batch_data_label[key]
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"{key} must be a nested list/tuple, got "
                f"{type(value).__name__}."
            )
        for batch_i, sample in enumerate(value):
            if not isinstance(sample, (list, tuple)):
                raise TypeError(
                    f"{key}[{batch_i}] must be a list/tuple."
                )
            for obj_i, tensor in enumerate(sample):
                if not torch.is_tensor(tensor):
                    raise TypeError(
                        f"{key}[{batch_i}][{obj_i}] must be a tensor."
                    )
                if tensor.device.type != "cpu":
                    raise RuntimeError(
                        f"{key}[{batch_i}][{obj_i}] must remain on CPU "
                        f"before label matching; got {tensor.device}."
                    )


def reduce_scalar(value, device, distributed: bool, average: bool = True):
    """Reduce a fixed scalar. Every rank must call this in the same order."""
    if not torch.is_tensor(value):
        value = torch.tensor(float(value), device=device, dtype=torch.float32)
    else:
        value = value.detach().to(device=device, dtype=torch.float32)
    if distributed and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if average:
            value = value / dist.get_world_size()
    return value


def reduce_metric_dict(metrics: dict, device, distributed: bool, average: bool = True):
    """Safely reduce a dynamic scalar-metric dict across DDP ranks.

    DO NOT iterate over each rank's local metrics and call all_reduce per key:
    label/loss debug keys can be data-dependent, so ranks can have different
    key sets and enter different collective sequences.

    This function first gathers the union of keys, then reduces one fixed-shape
    tensor [num_keys, 2] = [sum, count]. Every rank executes the same collectives.
    """
    if (not distributed) or (not dist.is_available()) or (not dist.is_initialized()):
        return dict(metrics)

    local_keys = sorted(metrics.keys())
    gathered_keys = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_keys, local_keys)

    all_keys = sorted(set(k for ks in gathered_keys for k in ks))
    if len(all_keys) == 0:
        return {}

    buf = torch.zeros((len(all_keys), 2), device=device, dtype=torch.float64)
    for j, key in enumerate(all_keys):
        if key in metrics:
            val = float(metrics[key])
            if math.isfinite(val):
                buf[j, 0] = val
                buf[j, 1] = 1.0

    dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    reduced = {}
    for j, key in enumerate(all_keys):
        count = float(buf[j, 1].item())
        if count > 0:
            reduced[key] = float((buf[j, 0] / count).item()) if average else float(buf[j, 0].item())
    return reduced


def reduce_sum_and_count(local_sum: float, local_count: int, device, distributed: bool):
    buf = torch.tensor([float(local_sum), float(local_count)], device=device, dtype=torch.float64)
    if distributed and dist.is_initialized():
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return float(buf[0].item()), int(buf[1].item())


def reduce_metric_sums_counts(stat_sums: dict, stat_counts: dict, device, distributed: bool):
    """Reduce epoch-level metric sums/counts safely when key sets differ by rank."""
    if (not distributed) or (not dist.is_available()) or (not dist.is_initialized()):
        return {
            k: float(stat_sums[k]) / max(int(stat_counts.get(k, 0)), 1)
            for k in sorted(stat_sums.keys())
        }

    local_keys = sorted(stat_sums.keys())
    gathered_keys = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_keys, local_keys)

    all_keys = sorted(set(k for ks in gathered_keys for k in ks))
    if len(all_keys) == 0:
        return {}

    buf = torch.zeros((len(all_keys), 2), device=device, dtype=torch.float64)
    for j, key in enumerate(all_keys):
        if key in stat_sums:
            buf[j, 0] = float(stat_sums[key])
            buf[j, 1] = float(stat_counts.get(key, 0))

    dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    reduced = {}
    for j, key in enumerate(all_keys):
        count = float(buf[j, 1].item())
        if count > 0:
            reduced[key] = float((buf[j, 0] / count).item())
    return reduced


class MetricAverager:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update_scalar(self, key: str, value: float, n: int = 1):
        self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
        self.counts[key] = self.counts.get(key, 0) + n

    def get_local_avg(self, key: str):
        c = self.counts.get(key, 0)
        return self.sums.get(key, 0.0) / max(c, 1)

    def keys(self):
        return list(self.sums.keys())


class Trainer:
    def __init__(self):
        self.distributed, self.rank, self.local_rank, self.world_size, self.device = setup_distributed()
        self.main = is_main_process(self.rank)
        seed_everything(getattr(cfgs, 'seed', 0), self.rank)
        self.distill_stage = int(DISTILL_ARGS.distill_stage)
        self.train_geometry_depth_source = (
            "gt" if self.distill_stage == 0 else "pred"
        )
        self.teacher_geometry_depth_source = (
            "gt" if self.distill_stage == 2 else None
        )
        # E2 protocol: the deployed RGB student owns proposal selection. During
        # Stage 2 training it therefore runs image-FPS autonomously first; the
        # frozen clean-depth teacher is evaluated at those exact student seeds.
        self.stage2_seed_source = "student" if self.distill_stage == 2 else None
        self.train_pose_depth_mode = (
            "none"
            if self.distill_stage == 0
            else str(getattr(cfgs, "pose_depth_mode", "none") or "none")
        )
        self.use_cdf = bool(
            getattr(cfgs, "use_cdf", False)
        )
        if not self.use_cdf:
            raise RuntimeError(
                "The minimal distillation implementation intentionally supports "
                "only the current CVA-CDF model. Add --use_cdf."
            )
        if self.distill_stage == 2:
            teacher_checkpoint = str(DISTILL_ARGS.teacher_checkpoint).strip()
            if not teacher_checkpoint:
                raise RuntimeError(
                    "--distill_stage 2 requires --teacher_checkpoint."
                )
            if not os.path.isfile(teacher_checkpoint):
                raise FileNotFoundError(
                    f"Teacher checkpoint does not exist: {teacher_checkpoint}"
                )
        elif str(DISTILL_ARGS.teacher_checkpoint).strip() and self.main:
            print(
                "[WARN] --teacher_checkpoint is ignored unless "
                "--distill_stage 2.",
                flush=True,
            )

        self.distill_config = OutputDistillationConfig(
            overall_weight=float(DISTILL_ARGS.distill_weight),
            objectness_weight=float(DISTILL_ARGS.kd_objectness_weight),
            graspness_weight=float(DISTILL_ARGS.kd_graspness_weight),
            depth_weight=float(DISTILL_ARGS.kd_depth_weight),
            view_weight=float(DISTILL_ARGS.kd_view_weight),
            cdf_weight=float(DISTILL_ARGS.kd_cdf_weight),
            width_weight=float(DISTILL_ARGS.kd_width_weight),
            temperature=float(DISTILL_ARGS.kd_temperature),
            max_query_view_angle_deg=float(
                DISTILL_ARGS.kd_max_query_view_angle_deg
            ),
            width_positive_threshold=float(
                DISTILL_ARGS.kd_width_positive_threshold
            ),
            min_depth=float(cfgs.min_depth),
            max_depth=float(cfgs.max_depth),
        )
        if self.distill_config.temperature <= 0:
            raise ValueError("--kd_temperature must be positive.")
        for name in (
            "overall_weight",
            "objectness_weight",
            "graspness_weight",
            "depth_weight",
            "view_weight",
            "cdf_weight",
            "width_weight",
        ):
            if float(getattr(self.distill_config, name)) < 0:
                raise ValueError(f"KD weight {name} must be non-negative.")
        if not (0.0 <= self.distill_config.max_query_view_angle_deg <= 180.0):
            raise ValueError(
                "--kd_max_query_view_angle_deg must lie in [0, 180]."
            )
        if not (0.0 <= self.distill_config.width_positive_threshold <= 1.0):
            raise ValueError(
                "--kd_width_positive_threshold must lie in [0, 1]."
            )
        self.kd_diag_interval_steps = max(
            0, int(DISTILL_ARGS.kd_diag_interval_steps)
        )
        self.kd_diag_eval_batches = max(
            0, int(DISTILL_ARGS.kd_diag_eval_batches)
        )
        self.kd_diag_grad_conflict = bool(
            int(DISTILL_ARGS.kd_diag_grad_conflict)
        )
        self.visualization_enabled = bool(
            getattr(cfgs, "vis_dir", None)
        )
        if not bool(getattr(cfgs, 'multi_modal', False)):
            raise RuntimeError("CVA training requires --multi_modal.")
        if bool(getattr(cfgs, "use_obs_depth", False)):
            raise RuntimeError(
                "Stage 0--2 is defined as privileged clean-depth teacher -> "
                "RGB-only student. Remove --use_obs_depth."
            )
        if bool(getattr(cfgs, "use_gt_depth", False)):
            raise RuntimeError(
                "Do not pass --use_gt_depth to the dataset. The teacher reads "
                "the always-returned gt_depth_m tensor internally, while all "
                "stages keep identical sensor-based RGB crops and token labels."
            )
        if self.distill_stage == 0:
            configured_pose_mode = str(
                getattr(cfgs, "pose_depth_mode", "none") or "none"
            )
            if configured_pose_mode != "none":
                raise RuntimeError(
                    "Stage 0 bypasses the metric-depth decoder; set "
                    "--pose_depth_mode none."
                )
            if abs(float(getattr(cfgs, "depth_prob_loss_weight", 0.0))) > 1e-12:
                raise RuntimeError(
                    "Stage 0 has no predicted depth; set "
                    "--depth_prob_loss_weight 0."
                )
        if bool(getattr(cfgs, "use_depth_comp", False)):
            raise RuntimeError(
                "This switchable CVA trainer uses explicit per-angle labels "
                "for both legacy and CDF modes. No extended-angle depth-"
                "compensation matcher is implemented; disable --use_depth_comp."
            )
        if bool(getattr(cfgs, 'kview_use_collision', False)):
            raise RuntimeError(
                "This switchable CVA model has no collision prediction head; "
                "remove --kview_use_collision."
            )
        if bool(getattr(cfgs, "pin_memory", False)):
            raise RuntimeError(
                "Do not use --pin_memory in the CVA trainer: the "
                "extended variable-size object labels must remain pageable "
                "CPU tensors until label matching."
            )
        self.cdf_diag_interval = (
            20
            if self.use_cdf
            else 0
        )
        self.cdf_eval_diag_interval = (
            50
            if self.use_cdf
            else 0
        )
        self.geometry_diag_interval = (
            20
            if self.visualization_enabled and self.main
            else 0
        )
        self.geometry_eval_diag_interval = (
            50
            if self.visualization_enabled and self.main
            else 0
        )

        os.makedirs(cfgs.log_dir, exist_ok=True)
        self.log_path = os.path.join(cfgs.log_dir, 'log_train.txt')
        self.LOG_FOUT = open(self.log_path, 'a') if self.main else None
        if self.main:
            self.LOG_FOUT.write(str(cfgs) + '\n')
            self.LOG_FOUT.write(
                "distillation_args=" + str(vars(DISTILL_ARGS)) + "\n"
            )
            self.LOG_FOUT.write(
                "distillation_config="
                + str(self.distill_config.to_dict())
                + "\n"
            )
            self.LOG_FOUT.flush()

        self.log_writer = SummaryWriter(os.path.join(cfgs.log_dir)) if self.main else None

        # Every Center-View-Angle variant requires the extended-angle
        # cache. The base dataset supplies frame inputs and object poses only;
        # CVAExtendedLabelAdapter owns one read of the common superset cache.
        train_base_dataset = GraspNetMultiDataset(
            cfgs.dataset_root,
            camera=cfgs.camera,
            split="train",
            voxel_size=cfgs.voxel_size,
            num_points=cfgs.num_point,
            remove_outlier=True,
            augment=False,
            # Keep RGB crop/mask/point labels identical across stages.  The
            # privileged teacher consumes the separately returned gt_depth_m.
            use_gt_depth=False,
            use_fuse_depth=cfgs.use_fuse_depth,
            graspness_mode=cfgs.graspness_mode,
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            depth_strides=1,
            extend_angle=True,
            load_grasp_payload=False,
        )
        test_base_dataset = GraspNetMultiDataset(
            cfgs.dataset_root,
            camera=cfgs.camera,
            split="test_seen",
            num_points=cfgs.num_point,
            remove_outlier=True,
            augment=False,
            voxel_size=cfgs.voxel_size,
            use_gt_depth=False,
            use_fuse_depth=cfgs.use_fuse_depth,
            graspness_mode=cfgs.graspness_mode,
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            depth_strides=1,
            extend_angle=True,
            load_grasp_payload=False,
        )
        cva_label_folder = str(
            getattr(cfgs, "cva_label_folder", "")
            or os.environ.get(
                "CVA_LABEL_FOLDER",
                os.environ.get(
                    "CDF_LABEL_FOLDER",
                    "economic_grasp_label_300views_"
                    "extend_angle_cdf_depth",
                ),
            )
        )
        self.TRAIN_DATASET = CVAExtendedLabelAdapter(
            train_base_dataset,
            dataset_root=cfgs.dataset_root,
            use_cdf=self.use_cdf,
            label_folder=cva_label_folder,
            num_angle=cfgs.num_angle,
            num_depth=cfgs.num_depth,
        )
        self.TEST_DATASET = CVAExtendedLabelAdapter(
            test_base_dataset,
            dataset_root=cfgs.dataset_root,
            use_cdf=self.use_cdf,
            label_folder=cva_label_folder,
            num_angle=cfgs.num_angle,
            num_depth=cfgs.num_depth,
        )
        if self.main:
            self.log_string(
                "-> CVA label mode="
                + ("CDF+depth-wise-width" if self.use_cdf else "legacy explicit-angle")
                + f", common cache={cva_label_folder}"
            )

        self.train_sampler = DistributedSampler(
            self.TRAIN_DATASET,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=False,
        ) if self.distributed else None

        self.test_sampler = DistributedSampler(
            self.TEST_DATASET,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        ) if self.distributed else None

        self.TRAIN_DATALOADER = DataLoader(
            self.TRAIN_DATASET,
            batch_size=cfgs.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=cfgs.num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            persistent_workers=(cfgs.num_workers > 0),
        )
        eval_num_workers = max(int(getattr(cfgs, 'eval_num_workers', 1)), 0)
        self.TEST_DATALOADER = DataLoader(
            self.TEST_DATASET,
            batch_size=cfgs.batch_size,
            shuffle=False,
            sampler=self.test_sampler,
            num_workers=eval_num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )

        common_model_kwargs = dict(
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            use_depth_comp=False,
            use_cdf=self.use_cdf,
            vis_every=int(getattr(cfgs, 'vis_every', 1000)),
        )
        student_pose_depth_mode = self.train_pose_depth_mode

        if self.distill_stage == 0:
            # Stage 0: RGB proposal features + privileged clean GT geometry.
            # The teacher subclass forces geometry_depth_source='gt', disables
            # pose-conditioned depth, and freezes/bypasses the depth decoder.
            self.net = economicgrasp_dpt_teacher(
                **common_model_kwargs,
                is_training=True,
                vis_dir=getattr(cfgs, 'vis_dir', None) if self.main else None,
            )
            train_model_kind = "privileged_gt_depth_teacher"
        else:
            # Stage 1/2 trainable model: strict RGB-only student.
            self.net = economicgrasp_dpt_student(
                **common_model_kwargs,
                is_training=True,
                use_obs_depth=False,
                pose_depth_mode=student_pose_depth_mode,
                vis_dir=getattr(cfgs, 'vis_dir', None) if self.main else None,
            )
            train_model_kind = "rgb_pred_depth_student"

        self.net.to(self.device)

        self.teacher = None
        if self.distill_stage == 2:
            teacher_path = str(DISTILL_ARGS.teacher_checkpoint)
            teacher_checkpoint_data = torch.load(
                teacher_path, map_location="cpu"
            )
            if not isinstance(teacher_checkpoint_data, dict) or (
                "model_state_dict" not in teacher_checkpoint_data
            ):
                raise RuntimeError(
                    "Stage 2 requires a full Stage-0 checkpoint with metadata."
                )
            if int(teacher_checkpoint_data.get("distill_stage", -1)) != 0:
                raise RuntimeError(
                    "Stage 2 requires a Stage-0 privileged teacher checkpoint, "
                    f"but distill_stage={teacher_checkpoint_data.get('distill_stage')!r}."
                )
            if int(teacher_checkpoint_data.get("distill_contract_version", -1)) != DISTILL_CONTRACT_VERSION:
                raise RuntimeError(
                    "Stage 2 rejects pre-privileged-depth checkpoints. Expected "
                    f"distill_contract_version={DISTILL_CONTRACT_VERSION}, got "
                    f"{teacher_checkpoint_data.get('distill_contract_version')!r}."
                )
            if str(teacher_checkpoint_data.get("seed_selection_mode", "")) != "image_fps":
                raise RuntimeError(
                    "Stage 2 requires an image-FPS teacher checkpoint, but "
                    f"seed_selection_mode={teacher_checkpoint_data.get('seed_selection_mode')!r}."
                )
            if str(teacher_checkpoint_data.get("geometry_depth_source", "")) != "gt":
                raise RuntimeError(
                    "Stage 2 requires a newly trained clean-depth teacher. "
                    "The checkpoint must contain geometry_depth_source='gt'; "
                    f"got {teacher_checkpoint_data.get('geometry_depth_source')!r}."
                )
            if bool(teacher_checkpoint_data.get("depth_head_executed", True)):
                raise RuntimeError(
                    "Stage-0 teacher metadata says the depth head was executed; "
                    "train a corrected privileged clean-depth teacher first."
                )
            if str(teacher_checkpoint_data.get("pose_depth_mode", "")) != "none":
                raise RuntimeError(
                    "Stage-0 teacher must use pose_depth_mode='none'; got "
                    f"{teacher_checkpoint_data.get('pose_depth_mode')!r}."
                )
            if bool(
                teacher_checkpoint_data.get(
                    "legacy_dataset_use_gt_depth", True
                )
            ):
                raise RuntimeError(
                    "Stage-0 teacher used the legacy dataset --use_gt_depth "
                    "switch and violates the controlled crop/label protocol."
                )
            saved_fuse = teacher_checkpoint_data.get("use_fuse_depth", None)
            if saved_fuse is None or bool(saved_fuse) != bool(cfgs.use_fuse_depth):
                raise RuntimeError(
                    "Stage-2 data must use the same clean-depth construction as "
                    "the teacher checkpoint: "
                    f"checkpoint use_fuse_depth={saved_fuse!r}, current="
                    f"{bool(cfgs.use_fuse_depth)}."
                )

            self.teacher = economicgrasp_dpt_teacher(
                **common_model_kwargs,
                is_training=False,
                vis_dir=None,
            )
            load_checkpoint_state(
                self.teacher,
                teacher_path,
                strict=True,
                checkpoint_data=teacher_checkpoint_data,
            )
            del teacher_checkpoint_data
            self.teacher.to(self.device)
            self.teacher.eval()
            self.teacher.requires_grad_(False)

        if self.main:
            self.log_string(
                f"-> distillation stage={self.distill_stage}, "
                f"train_model={train_model_kind}, "
                f"train_geometry_depth={self.train_geometry_depth_source}, "
                f"pose_depth_mode={self.train_pose_depth_mode}, "
                f"use_fuse_depth={bool(cfgs.use_fuse_depth)}, "
                f"teacher={'enabled' if self.teacher is not None else 'disabled'}"
            )
            if self.teacher is not None:
                self.log_string(
                    f"-> frozen clean-depth teacher: "
                    f"{DISTILL_ARGS.teacher_checkpoint}"
                )
                self.log_string(
                    "-> Stage-2 protocol: E1 privileged-output KD "
                    "(view/CDF/width by default) + E2 student-driven image-FPS"
                )
                self.log_string(
                    "-> KD diagnostics: interval_steps="
                    f"{self.kd_diag_interval_steps}, eval_batches="
                    f"{self.kd_diag_eval_batches}, grad_conflict="
                    f"{int(self.kd_diag_grad_conflict)}"
                )

        if self.distributed:
            # IMPORTANT: keep device_ids=None.
            #
            # With device_ids=[local_rank], PyTorch DDP recursively applies
            # _to_kwargs() to every forward input. That silently moves the
            # full variable-length extended CVA label lists to CUDA,
            # even though move_batch_to_device() deliberately keeps them on
            # CPU. the label matcher would then violate its
            # CPU-residency check.
            #
            # The module has already been moved to this rank's CUDA device and
            # all fixed-size model inputs are transferred explicitly before
            # forward, so device_ids=None is the correct DDP mode here. It
            # disables DDP's recursive input transfer while preserving normal
            # gradient synchronization.
            self.net = DDP(
                self.net,
                device_ids=None,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
            if self.main:
                self.log_string(
                    '[DDP] device_ids=None: fixed-size inputs are moved '
                    'explicitly; full extended CVA object labels remain on CPU.'
                )

        self.optimizer = self.build_optimizer()

        self.start_epoch = 0
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            state_dict = (
                checkpoint['model_state_dict']
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
                else checkpoint
            )

            if bool(getattr(cfgs, 'resume', False)):
                # Resume is strict: it must be a checkpoint produced by the same
                # CDF/depth-wise-width architecture.
                if isinstance(checkpoint, dict) and 'distill_stage' in checkpoint:
                    saved_stage = int(checkpoint['distill_stage'])
                    if saved_stage != self.distill_stage:
                        raise RuntimeError(
                            '--resume stage mismatch: checkpoint was produced by '
                            f'distill_stage={saved_stage}, current stage='
                            f'{self.distill_stage}.'
                        )
                if isinstance(checkpoint, dict) and 'seed_selection_mode' in checkpoint:
                    saved_seed_mode = str(checkpoint['seed_selection_mode'])
                    if saved_seed_mode != 'image_fps':
                        raise RuntimeError(
                            '--resume seed selector mismatch: checkpoint uses '
                            f'{saved_seed_mode!r}, current selector is image_fps.'
                        )
                saved_depth_source = (
                    checkpoint.get('geometry_depth_source', None)
                    if isinstance(checkpoint, dict)
                    else None
                )
                if str(saved_depth_source) != self.train_geometry_depth_source:
                    raise RuntimeError(
                        '--resume geometry-depth mismatch: checkpoint uses '
                        f'{saved_depth_source!r}, current stage requires '
                        f'{self.train_geometry_depth_source!r}.'
                    )
                if int(checkpoint.get('distill_contract_version', -1)) != DISTILL_CONTRACT_VERSION:
                    raise RuntimeError(
                        '--resume requires the current privileged-depth contract '
                        f'version {DISTILL_CONTRACT_VERSION}.'
                    )
                if bool(checkpoint.get('use_fuse_depth', False)) != bool(cfgs.use_fuse_depth):
                    raise RuntimeError(
                        '--resume use_fuse_depth mismatch: checkpoint uses '
                        f"{checkpoint.get('use_fuse_depth')!r}, current="
                        f"{bool(cfgs.use_fuse_depth)}."
                    )
                if str(checkpoint.get('pose_depth_mode', '')) != self.train_pose_depth_mode:
                    raise RuntimeError(
                        '--resume pose-depth mismatch: checkpoint uses '
                        f"{checkpoint.get('pose_depth_mode')!r}, current="
                        f"{self.train_pose_depth_mode!r}."
                    )
                if self.distill_stage == 2:
                    saved_seed_source = str(
                        checkpoint.get('stage2_seed_source', '')
                    )
                    if saved_seed_source != 'student':
                        raise RuntimeError(
                            '--resume Stage-2 seed-source mismatch: E2 requires '
                            "stage2_seed_source='student'. Start a new E1+E2 "
                            'Stage-2 run from the Stage-1 checkpoint instead of '
                            'resuming an older teacher-driven Stage-2 checkpoint.'
                        )
                self.unwrap_model().load_state_dict(state_dict, strict=True)
                if not (
                    isinstance(checkpoint, dict)
                    and 'optimizer_state_dict' in checkpoint
                    and 'epoch' in checkpoint
                ):
                    raise RuntimeError(
                        '--resume requires a full training checkpoint with model, '
                        'optimizer, and epoch states.'
                    )
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = int(checkpoint['epoch'])
                self.log_string(
                    f'-> resumed checkpoint {CHECKPOINT_PATH} '
                    f'(epoch: {self.start_epoch})'
                )
            else:
                # Initialization from the previous CVA model: load every tensor
                # whose name and shape still match.  New CDF/width heads are
                # intentionally initialized from scratch.
                model = self.unwrap_model()
                current = model.state_dict()
                compatible = {}
                skipped_shape = []
                unexpected = []
                for key, value in state_dict.items():
                    if key not in current:
                        unexpected.append(key)
                    elif tuple(value.shape) != tuple(current[key].shape):
                        skipped_shape.append(
                            (key, tuple(value.shape), tuple(current[key].shape))
                        )
                    else:
                        compatible[key] = value
                load_result = model.load_state_dict(compatible, strict=False)
                self.log_string(
                    f'-> initialized from {CHECKPOINT_PATH}: '
                    f'loaded={len(compatible)}, shape_skipped={len(skipped_shape)}, '
                    f'unexpected={len(unexpected)}, missing={len(load_result.missing_keys)}'
                )
                for key, old_shape, new_shape in skipped_shape[:20]:
                    self.log_string(
                        f'   [INIT-SKIP] {key}: {old_shape} -> {new_shape}'
                    )

    def unwrap_model(self):
        return self.net.module if hasattr(self.net, 'module') else self.net

    def build_optimizer(self):
        depth_weight_decay = float(getattr(cfgs, 'depth_weight_decay', 0.0))

        if depth_weight_decay <= 0:
            trainable_params = [p for p in self.net.parameters() if p.requires_grad]
            return optim.AdamW(
                trainable_params,
                lr=cfgs.learning_rate,
                weight_decay=cfgs.weight_decay,
            )

        model = self.unwrap_model()
        depth_net = getattr(model, 'depth_net', None)
        if depth_net is None:
            self.log_string(
                '[WARN] cfgs.depth_weight_decay > 0, but model has no depth_net; '
                'fall back to cfgs.weight_decay for all trainable parameters.'
            )
            trainable_params = [p for p in self.net.parameters() if p.requires_grad]
            return optim.AdamW(
                trainable_params,
                lr=cfgs.learning_rate,
                weight_decay=cfgs.weight_decay,
            )

        depth_params = [p for p in depth_net.parameters() if p.requires_grad]
        depth_param_ids = {id(p) for p in depth_params}
        grasp_params = [
            p for p in self.net.parameters()
            if p.requires_grad and id(p) not in depth_param_ids
        ]

        param_groups = []
        if grasp_params:
            param_groups.append({
                'params': grasp_params,
                'weight_decay': cfgs.weight_decay,
            })
        if depth_params:
            param_groups.append({
                'params': depth_params,
                'weight_decay': depth_weight_decay,
            })

        if len(param_groups) == 0:
            raise RuntimeError('No trainable parameters found for optimizer.')

        self.log_string(
            f'-> optimizer weight_decay: grasp={cfgs.weight_decay}, '
            f'depth_net={depth_weight_decay}'
        )
        return optim.AdamW(
            param_groups,
            lr=cfgs.learning_rate,
            weight_decay=cfgs.weight_decay,
        )

    def log_string(self, out_str: str):
        if self.main and self.LOG_FOUT is not None:
            self.LOG_FOUT.write(out_str + '\n')
            self.LOG_FOUT.flush()
        if self.main:
            print(out_str)

    def get_current_lr(self, epoch):
        lr = cfgs.learning_rate
        lr = lr * (math.cos(epoch / cfgs.max_epoch * math.pi) + 1) * 0.5
        return lr

    def adjust_learning_rate(self, epoch):
        lr = self.get_current_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def maybe_log_scalars(self, prefix: str, scalars: dict, global_step: int):
        if self.main and self.log_writer is not None:
            for key, val in scalars.items():
                self.log_writer.add_scalar(prefix + key, val, global_step)

    def extract_scalar_metrics(self, end_points):
        metrics = {}
        for key, val in end_points.items():
            if ('loss' in key) or key.startswith(('A:', 'B:', 'C:', 'D:')):
                if torch.is_tensor(val):
                    if val.numel() == 1:
                        metrics[key] = float(val.detach().item())
                else:
                    try:
                        metrics[key] = float(val)
                    except Exception:
                        pass
        return metrics

    def train_one_epoch(self, epoch):
        self.adjust_learning_rate(epoch)
        self.net.train()
        if self.teacher is not None:
            # ``self.net.train()`` must never change the frozen teacher mode.
            self.teacher.eval()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        overall_loss_local = 0.0
        local_batches = 0
        num_batches_local = len(self.TRAIN_DATALOADER)
        batch_interval = 20

        stat_interval = MetricAverager()
        interval_data_time = 0.0
        interval_teacher_time = 0.0
        interval_model_time = 0.0
        interval_loss_time = 0.0
        interval_opt_time = 0.0
        interval_start_time = time.perf_counter()
        data_start_time = time.perf_counter()

        for batch_idx, batch_data_label in enumerate(self.TRAIN_DATALOADER):
            t = time.time()
            optimizer_step = epoch * num_batches_local + batch_idx + 1
            run_kd_diag = (
                self.teacher is not None
                and self.kd_diag_interval_steps > 0
                and (
                    optimizer_step == 1
                    or optimizer_step % self.kd_diag_interval_steps == 0
                )
            )

            validate_batch_label_contract(
                batch_data_label,
                use_cdf=self.use_cdf,
            )
            batch_data_label = drop_unused_point_inputs(batch_data_label)
            batch_data_label = move_batch_to_device(
                batch_data_label,
                self.device,
                use_cdf=self.use_cdf,
                non_blocking=False,
            )
            batch_data_label["cva_compute_diagnostics"] = (
                self.use_cdf
                and (
                    (
                        self.cdf_diag_interval > 0
                        and batch_idx % self.cdf_diag_interval == 0
                    )
                    or run_kd_diag
                )
            )
            batch_data_label[
                "geometry_compute_diagnostics"
            ] = (
                self.visualization_enabled
                and self.main
                and self.geometry_diag_interval > 0
                and batch_idx % self.geometry_diag_interval == 0
            )
            batch_data_label["cva_export_angle_feature"] = False

            data_end_time = time.perf_counter()
            interval_data_time += (data_end_time - data_start_time)

            assert_cpu_resident_label_lists(
                batch_data_label,
                use_cdf=self.use_cdf,
            )

            # E2: preserve a pristine teacher input before the trainable
            # student mutates the top-level endpoint dictionary. The student
            # must run first and select its own deployment-time image-FPS seeds.
            teacher_input = None
            diagnostic_teacher_input = None
            if self.teacher is not None:
                batch_data_label.pop("image_fps_seed_idx_override", None)
                batch_data_label.pop("oracle_view_inds_override", None)
                teacher_input = dict(batch_data_label)
                teacher_input["cva_compute_diagnostics"] = False
                teacher_input["geometry_compute_diagnostics"] = False
                teacher_input["cva_export_angle_feature"] = False
                teacher_input["cva_force_process_grasp_labels"] = False
                if run_kd_diag:
                    diagnostic_teacher_input = dict(teacher_input)
                    diagnostic_teacher_input["cva_compute_diagnostics"] = True
                    diagnostic_teacher_input[
                        "cva_force_process_grasp_labels"
                    ] = True

            model_start_time = time.perf_counter()
            end_points = self.net(batch_data_label)
            if batch_idx == 0:
                assert_geometry_depth_contract(
                    end_points,
                    expected_source=self.train_geometry_depth_source,
                    context=f"stage{self.distill_stage} train epoch={epoch}",
                )
            model_end_time = time.perf_counter()
            interval_model_time += (model_end_time - model_start_time)

            teacher_targets = None
            if self.teacher is not None:
                # Student-driven shared image-FPS: the RGB student determines
                # the exact ordered [B,M] seed indices used at deployment. The
                # privileged teacher then evaluates clean geometry at those same
                # image locations, avoiding the former teacher-seed train/test
                # mismatch while retaining exact query-center correspondence in
                # image space.
                student_idx = end_points["kview_base_token_sel_idx"].detach().long()
                teacher_input["image_fps_seed_idx_override"] = student_idx

                teacher_start_time = time.perf_counter()
                # Use no_grad rather than inference_mode: teacher targets are
                # consumed by student losses whose backward kernels may save
                # the target tensor. PyTorch inference tensors cannot be saved
                # for backward without an extra clone.
                with torch.no_grad():
                    teacher_end_points = self.teacher(teacher_input)
                    if batch_idx == 0:
                        assert_geometry_depth_contract(
                            teacher_end_points,
                            expected_source="gt",
                            context=f"stage2 teacher epoch={epoch}",
                        )
                    teacher_targets = extract_distillation_targets(
                        teacher_end_points
                    )
                teacher_end_time = time.perf_counter()
                interval_teacher_time += (
                    teacher_end_time - teacher_start_time
                )

                shared_idx = teacher_targets["kview_base_token_sel_idx"].to(
                    device=student_idx.device,
                    dtype=torch.long,
                )
                if not torch.equal(student_idx, shared_idx):
                    raise RuntimeError(
                        "E2 requires the teacher to reuse the student's exact "
                        "ordered image-FPS indices, but the indices differ."
                    )
                end_points["D: Shared image-FPS exact ratio"] = (
                    (student_idx == shared_idx).float().mean().reshape(())
                )
                end_points["D: Stage2 student autonomous image-FPS"] = (
                    end_points["D: Shared image-FPS exact ratio"].new_tensor(1.0)
                )
                del teacher_end_points, teacher_input


            end_points['epoch'] = epoch

            loss_start_time = time.perf_counter()
            supervised_loss, end_points = get_loss_economicgrasp(
                end_points,
                use_cdf=self.use_cdf,
            )
            end_points['A: Supervised Loss'] = supervised_loss
            if teacher_targets is not None:
                distill_loss, end_points = compute_output_distillation_loss(
                    end_points,
                    teacher_targets,
                    self.distill_config,
                )
                loss = supervised_loss + distill_loss
            else:
                distill_loss = supervised_loss.detach() * 0.0
                end_points['A: Distill Loss'] = distill_loss
                loss = supervised_loss
            end_points['A: Overall Loss'] = loss

            if run_kd_diag and diagnostic_teacher_input is not None:
                # Paired counterfactual teacher: same student-selected image
                # seeds and exact same selected view anchors.  This isolates
                # whether clean geometry produces outputs that are actually
                # closer to grasp GT, and whether depth-induced center drift
                # changes the matched CDF/width labels.
                diagnostic_teacher_input[
                    "image_fps_seed_idx_override"
                ] = end_points["kview_base_token_sel_idx"].detach().long()
                diagnostic_teacher_input[
                    "oracle_view_inds_override"
                ] = end_points["grasp_top_view_inds"].detach().long()
                diagnostic_teacher_input[
                    "geometry_compute_diagnostics"
                ] = False
                diagnostic_teacher_input[
                    "cva_export_angle_feature"
                ] = False
                with torch.no_grad():
                    diagnostic_teacher_end_points = self.teacher(
                        diagnostic_teacher_input
                    )
                    diagnostic_teacher_end_points["epoch"] = epoch
                    _, diagnostic_teacher_end_points = get_loss_economicgrasp(
                        diagnostic_teacher_end_points,
                        use_cdf=self.use_cdf,
                    )
                    end_points.update(
                        compute_privileged_kd_diagnostics(
                            end_points,
                            diagnostic_teacher_end_points,
                        )
                    )
                del diagnostic_teacher_end_points, diagnostic_teacher_input

                if self.kd_diag_grad_conflict:
                    end_points.update(
                        compute_output_gradient_conflict(end_points)
                    )
                end_points["D: KDDiag optimizer step"] = (
                    end_points["A: Overall Loss"].detach().new_tensor(
                        float(optimizer_step)
                    )
                )
                if self.main:
                    self.log_string(
                        f"[KDDIAG] optimizer_step={optimizer_step}, "
                        f"epoch={epoch}, batch={batch_idx + 1}"
                    )

            del teacher_targets
            loss_end_time = time.perf_counter()
            interval_loss_time += (loss_end_time - loss_start_time)

            self.optimizer.zero_grad(set_to_none=True)
            bwdopt_start_time = time.perf_counter()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()

            bwdopt_end_time = time.perf_counter()
            interval_opt_time += (bwdopt_end_time - bwdopt_start_time)

            # IMPORTANT: dynamic metric keys must be reduced with a fixed collective pattern.
            metrics = self.extract_scalar_metrics(end_points)
            metrics = reduce_metric_dict(metrics, self.device, self.distributed, average=True)
            for key, val in metrics.items():
                stat_interval.update_scalar(key, val)

            # Use the safely reduced metric if present; otherwise reduce this fixed scalar once.
            if 'A: Overall Loss' in metrics:
                grasp_loss_val = metrics['A: Overall Loss']
            else:
                grasp_loss_val = reduce_scalar(
                    end_points['A: Overall Loss'],
                    self.device,
                    self.distributed,
                    average=True,
                ).item()

            overall_loss_local += grasp_loss_val
            local_batches += 1

            if (batch_idx + 1) % batch_interval == 0:
                remain_batches_local = (cfgs.max_epoch - epoch) * num_batches_local - batch_idx - 1
                interval_time = time.perf_counter() - interval_start_time
                avg_batch_time = interval_time / batch_interval
                remain_time_h = remain_batches_local * avg_batch_time / 3600.0

                if self.main:
                    self.log_string(f' ---- epoch: {epoch}, batch: {batch_idx + 1} ----')

                time_dict = {
                    'C: Data Time': interval_data_time / batch_interval,
                    'C: Teacher Time': interval_teacher_time / batch_interval,
                    'C: Model Time': interval_model_time / batch_interval,
                    'C: Loss Time': interval_loss_time / batch_interval,
                    'C: Bwd+Opt Time': interval_opt_time / batch_interval,
                    'C: Remain Time (h)': remain_time_h,
                }
                global_step = (epoch * len(self.TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size * self.world_size

                if self.main:
                    scalar_log = {}
                    for key in sorted(stat_interval.keys()):
                        val = stat_interval.get_local_avg(key)
                        scalar_log['train_' + key] = val
                        self.log_string(f'{key:<20}: {val:.4f}')
                    for key in sorted(time_dict.keys()):
                        scalar_log['train_' + key] = time_dict[key]
                        self.log_string(f'{key:<20}: {time_dict[key]:.4f}')
                    self.maybe_log_scalars('', scalar_log, global_step)

                stat_interval = MetricAverager()
                interval_data_time = 0.0
                interval_teacher_time = 0.0
                interval_model_time = 0.0
                interval_loss_time = 0.0
                interval_opt_time = 0.0
                interval_start_time = time.perf_counter()

            data_start_time = time.perf_counter()

        overall_sum, overall_count = reduce_sum_and_count(
            overall_loss_local,
            local_batches,
            self.device,
            self.distributed,
        )
        overall_loss = overall_sum / max(overall_count, 1)
        self.log_string(
            f'overall training loss per batch: {overall_loss}, '
            f'batch num:{overall_count}'
        )
        return overall_loss

    def evaluate_one_epoch(self, epoch):
        stat_sums_local = {}
        stat_counts_local = {}
        self.net.eval()
        overall_loss_local = 0.0
        local_batches = 0

        if self.test_sampler is not None:
            self.test_sampler.set_epoch(epoch)

        for batch_idx, batch_data_label in enumerate(self.TEST_DATALOADER):
            run_eval_kd_diag = (
                self.teacher is not None
                and (
                    self.kd_diag_eval_batches == 0
                    or batch_idx < self.kd_diag_eval_batches
                )
            )
            if batch_idx % 50 == 0:
                print(
                    f"[rank{self.rank}] Eval batch "
                    f"{batch_idx}/{len(self.TEST_DATALOADER)}",
                    flush=True,
                )

            validate_batch_label_contract(
                batch_data_label,
                use_cdf=self.use_cdf,
            )
            batch_data_label = drop_unused_point_inputs(batch_data_label)
            batch_data_label = move_batch_to_device(
                batch_data_label,
                self.device,
                use_cdf=self.use_cdf,
                non_blocking=False,
            )
            batch_data_label["cva_compute_diagnostics"] = (
                self.use_cdf
                and (
                    (
                        self.cdf_eval_diag_interval > 0
                        and batch_idx % self.cdf_eval_diag_interval == 0
                    )
                    or run_eval_kd_diag
                )
            )
            batch_data_label[
                "geometry_compute_diagnostics"
            ] = (
                self.visualization_enabled
                and self.main
                and self.geometry_eval_diag_interval > 0
                and batch_idx % self.geometry_eval_diag_interval == 0
            )
            batch_data_label["cva_export_angle_feature"] = False

            diagnostic_teacher_input = None
            if run_eval_kd_diag:
                batch_data_label.pop("image_fps_seed_idx_override", None)
                batch_data_label.pop("oracle_view_inds_override", None)
                diagnostic_teacher_input = dict(batch_data_label)
                diagnostic_teacher_input["cva_compute_diagnostics"] = True
                diagnostic_teacher_input["geometry_compute_diagnostics"] = False
                diagnostic_teacher_input["cva_export_angle_feature"] = False
                diagnostic_teacher_input[
                    "cva_force_process_grasp_labels"
                ] = True

            with torch.no_grad():
                assert_cpu_resident_label_lists(
                    batch_data_label,
                    use_cdf=self.use_cdf,
                )
                end_points = self.net(batch_data_label)
                if batch_idx == 0:
                    assert_geometry_depth_contract(
                        end_points,
                        expected_source=self.train_geometry_depth_source,
                        context=f"stage{self.distill_stage} eval epoch={epoch}",
                    )
                loss, end_points = get_loss_economicgrasp(
                    end_points,
                    use_cdf=self.use_cdf,
                )
                end_points['A: Supervised Loss'] = loss
                end_points['A: Distill Loss'] = loss.detach() * 0.0
                end_points['A: Overall Loss'] = loss

                if diagnostic_teacher_input is not None:
                    diagnostic_teacher_input[
                        "image_fps_seed_idx_override"
                    ] = end_points[
                        "kview_base_token_sel_idx"
                    ].detach().long()
                    diagnostic_teacher_input[
                        "oracle_view_inds_override"
                    ] = end_points[
                        "grasp_top_view_inds"
                    ].detach().long()
                    diagnostic_teacher_end_points = self.teacher(
                        diagnostic_teacher_input
                    )
                    diagnostic_teacher_end_points["epoch"] = epoch
                    _, diagnostic_teacher_end_points = get_loss_economicgrasp(
                        diagnostic_teacher_end_points,
                        use_cdf=self.use_cdf,
                    )
                    end_points.update(
                        compute_privileged_kd_diagnostics(
                            end_points,
                            diagnostic_teacher_end_points,
                        )
                    )
                    end_points["D: KDDiag eval paired batch"] = (
                        loss.detach().new_tensor(1.0)
                    )
                    del diagnostic_teacher_end_points, diagnostic_teacher_input

            metrics = self.extract_scalar_metrics(end_points)
            for key, val in metrics.items():
                stat_sums_local[key] = stat_sums_local.get(key, 0.0) + float(val)
                stat_counts_local[key] = stat_counts_local.get(key, 0) + 1

            overall_loss_local += float(
                end_points['A: Overall Loss'].detach().item()
            )
            local_batches += 1

        reduced_metrics = reduce_metric_sums_counts(
            stat_sums_local,
            stat_counts_local,
            self.device,
            self.distributed,
        )

        overall_sum, overall_count = reduce_sum_and_count(
            overall_loss_local,
            local_batches,
            self.device,
            self.distributed,
        )
        overall_loss = overall_sum / max(overall_count, 1)

        if self.main:
            global_step = (epoch + 1) * len(self.TRAIN_DATALOADER) * cfgs.batch_size * self.world_size
            self.maybe_log_scalars('', {'test_' + k: v for k, v in reduced_metrics.items()}, global_step)
            for key in sorted(reduced_metrics.keys()):
                self.log_string(f'eval mean {key}: {reduced_metrics[key]:.6f}')
            self.log_string(f'overall loss:{overall_loss}, batch num:{overall_count}')
        return overall_loss

    def save_best_state_dict(self, epoch, train_loss, eval_loss):
        if not self.main:
            return
        ckpt_name = f'epoch_{epoch}_train_{train_loss}_val_{eval_loss}'
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': self.unwrap_model().state_dict(),
                'distill_stage': self.distill_stage,
                'teacher_checkpoint': str(DISTILL_ARGS.teacher_checkpoint),
                'distillation_config': self.distill_config.to_dict(),
                'distill_contract_version': DISTILL_CONTRACT_VERSION,
                'seed_selection_mode': 'image_fps',
                'geometry_depth_source': self.train_geometry_depth_source,
                'teacher_geometry_depth_source': self.teacher_geometry_depth_source,
                'depth_head_executed': bool(self.train_geometry_depth_source == 'pred'),
                'pose_depth_mode': self.train_pose_depth_mode,
                'camera_pose_key': str(getattr(cfgs, 'camera_pose_key', 'camera_pose_vec')),
                'camera_gravity_key': str(getattr(cfgs, 'camera_gravity_key', 'camera_gravity_vec')),
                'pose_hidden_dim': int(getattr(cfgs, 'pose_hidden_dim', 64)),
                'ray_gravity_hidden_dim': int(getattr(cfgs, 'ray_gravity_hidden_dim', 64)),
                'ray_gravity_mid_dim': int(getattr(cfgs, 'ray_gravity_mid_dim', 32)),
                'use_fuse_depth': bool(cfgs.use_fuse_depth),
                'legacy_dataset_use_gt_depth': False,
                'stage2_shared_teacher_image_fps': False,
                'stage2_seed_source': self.stage2_seed_source,
                'stage2_teacher_reuses_student_image_fps': bool(self.distill_stage == 2),
            },
            os.path.join(cfgs.log_dir, ckpt_name + '.tar'),
        )

    def save_checkpoint(self, epoch, save_interval=False):
        if not self.main:
            return
        save_dict = {
            'epoch': epoch + 1,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.unwrap_model().state_dict(),
            'distill_stage': self.distill_stage,
            'teacher_checkpoint': str(DISTILL_ARGS.teacher_checkpoint),
            'distillation_config': self.distill_config.to_dict(),
            'distill_contract_version': DISTILL_CONTRACT_VERSION,
            'seed_selection_mode': 'image_fps',
            'geometry_depth_source': self.train_geometry_depth_source,
            'teacher_geometry_depth_source': self.teacher_geometry_depth_source,
            'depth_head_executed': bool(self.train_geometry_depth_source == 'pred'),
            'pose_depth_mode': self.train_pose_depth_mode,
            'camera_pose_key': str(getattr(cfgs, 'camera_pose_key', 'camera_pose_vec')),
            'camera_gravity_key': str(getattr(cfgs, 'camera_gravity_key', 'camera_gravity_vec')),
            'pose_hidden_dim': int(getattr(cfgs, 'pose_hidden_dim', 64)),
            'ray_gravity_hidden_dim': int(getattr(cfgs, 'ray_gravity_hidden_dim', 64)),
            'ray_gravity_mid_dim': int(getattr(cfgs, 'ray_gravity_mid_dim', 32)),
            'use_fuse_depth': bool(cfgs.use_fuse_depth),
            'legacy_dataset_use_gt_depth': False,
            'stage2_shared_teacher_image_fps': False,
            'stage2_seed_source': self.stage2_seed_source,
            'stage2_teacher_reuses_student_image_fps': bool(self.distill_stage == 2),
        }
        if save_interval:
            torch.save(save_dict, os.path.join(cfgs.log_dir, f'checkpoint_{epoch}.tar'))
        torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))

    def train(self, start_epoch):
        global EPOCH_CNT
        min_loss = np.inf
        best_epoch = 0
        for epoch in range(start_epoch, cfgs.max_epoch):
            EPOCH_CNT = epoch
            self.log_string(f'**** EPOCH {epoch:<3} ****')
            self.log_string('Current learning rate: %f' % self.get_current_lr(epoch))

            np.random.seed()
            train_loss = self.train_one_epoch(epoch)

            if bool(getattr(cfgs, 'enable_eval', False)) and epoch >= cfgs.eval_start_epoch:
                eval_loss = self.evaluate_one_epoch(epoch)
                if eval_loss < min_loss:
                    min_loss = eval_loss
                    best_epoch = epoch
                    self.save_best_state_dict(epoch, train_loss, eval_loss)
                    self.save_checkpoint(epoch)
                self.log_string(f'best_epoch:{best_epoch}')
            save_interval_flag = EPOCH_CNT % cfgs.ckpt_save_interval == 0
            self.save_checkpoint(epoch, save_interval_flag)

    def close(self):
        if self.log_writer is not None:
            self.log_writer.close()
        if self.LOG_FOUT is not None:
            self.LOG_FOUT.close()
        cleanup_distributed(self.distributed)


def main():
    trainer = Trainer()
    try:
        if bool(DISTILL_ARGS.diagnose_only):
            if trainer.distill_stage != 2 or trainer.teacher is None:
                raise RuntimeError(
                    "--diagnose_only requires --distill_stage 2 and a valid "
                    "Stage-0 --teacher_checkpoint."
                )
            trainer.log_string(
                "[KDDIAG] diagnose-only validation pass: "
                f"epoch_tag={int(DISTILL_ARGS.diagnose_epoch)}, "
                f"paired_batches={trainer.kd_diag_eval_batches}"
            )
            trainer.evaluate_one_epoch(int(DISTILL_ARGS.diagnose_epoch))
        else:
            trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == '__main__':
    main()