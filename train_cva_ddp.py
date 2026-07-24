# Basic Libraries
import os
import math
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

# Config
from utils.arguments import cfgs

# Local libraries: this training entry point is intentionally CDF-only.
from models.economicgrasp_bip3d import economicgrasp_dpt
from models.loss_economicgrasp_depth_kview_transformer import (
    get_loss as get_loss_economicgrasp,
)
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from dataset.cdf_label_adapter import CDFLabelAdapter

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


CPU_RESIDENT_LABEL_LIST_KEYS = {
    "object_poses_list",
    "grasp_points_list",
    "view_graspness_list",
    "top_view_index_list",
    "grasp_cdf_bins_list",
    "grasp_widths_depth_list",
    "grasp_width_valids_depth_list",
}


def move_batch_to_device(batch_data_label, device, non_blocking=False):
    """Move fixed-shape model inputs to GPU; keep dense object labels on CPU."""
    for key, value in batch_data_label.items():
        if key in CPU_RESIDENT_LABEL_LIST_KEYS:
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"{key} must remain a nested CPU list, got "
                    f"{type(value).__name__}."
                )
            for sample in value:
                for tensor in sample:
                    if not torch.is_tensor(tensor) or tensor.device.type != "cpu":
                        raise RuntimeError(
                            f"{key} must contain CPU tensors before label matching."
                        )
            continue

        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Unexpected list-valued batch key '{key}'. Add it explicitly "
                "to CPU_RESIDENT_LABEL_LIST_KEYS or collate it to a tensor."
            )
        if torch.is_tensor(value):
            batch_data_label[key] = value.to(
                device,
                non_blocking=non_blocking,
            )
    return batch_data_label


def assert_cpu_resident_label_lists(batch_data_label):
    """Validate the CPU/GPU boundary immediately before model forward."""
    for key in CPU_RESIDENT_LABEL_LIST_KEYS:
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
                    f"{key}[{batch_i}] must be a list/tuple, got "
                    f"{type(sample).__name__}."
                )
            for obj_i, tensor in enumerate(sample):
                if not torch.is_tensor(tensor):
                    raise TypeError(
                        f"{key}[{batch_i}][{obj_i}] must be a tensor, got "
                        f"{type(tensor).__name__}."
                    )
                if tensor.device.type != "cpu":
                    raise RuntimeError(
                        f"{key}[{batch_i}][{obj_i}] must remain CPU-resident "
                        f"before DDP forward; got device={tensor.device}."
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
        if not bool(getattr(cfgs, 'multi_modal', False)):
            raise RuntimeError("CDF CVA training requires --multi_modal.")
        if bool(getattr(cfgs, 'kview_use_collision', False)):
            raise RuntimeError(
                "The cleaned CDF model has no collision head; remove "
                "--kview_use_collision."
            )
        if bool(getattr(cfgs, 'pin_memory', False)):
            raise RuntimeError(
                "Do not use --pin_memory with the CDF-depth cache: PyTorch "
                "would recursively pin the complete variable-size object labels. "
                "Keep them pageable and transfer only matched rows per object."
            )
        self.cdf_diag_interval = max(
            int(getattr(cfgs, 'cdf_diag_interval', 20)), 0
        )
        self.cdf_eval_diag_interval = max(
            int(getattr(cfgs, 'cdf_eval_diag_interval', 50)), 0
        )

        os.makedirs(cfgs.log_dir, exist_ok=True)
        self.log_path = os.path.join(cfgs.log_dir, 'log_train.txt')
        self.LOG_FOUT = open(self.log_path, 'a') if self.main else None
        if self.main:
            self.LOG_FOUT.write(str(cfgs) + '\n')
            self.LOG_FOUT.flush()

        self.log_writer = SummaryWriter(os.path.join(cfgs.log_dir)) if self.main else None

        # Keep the shared GraspNet dataset model-agnostic. It loads the
        # standard scene/image/point labels; the CDF-specific object payload is
        # attached by a separate adapter below.
        train_base_dataset = GraspNetMultiDataset(
            cfgs.dataset_root,
            camera=cfgs.camera,
            split='train',
            voxel_size=cfgs.voxel_size,
            num_points=cfgs.num_point,
            remove_outlier=True,
            augment=False,
            use_gt_depth=cfgs.use_gt_depth,
            use_fuse_depth=cfgs.use_fuse_depth,
            graspness_mode=cfgs.graspness_mode,
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            depth_strides=1,
            extend_angle=cfgs.extend_angle,
        )
        test_base_dataset = GraspNetMultiDataset(
            cfgs.dataset_root,
            camera=cfgs.camera,
            split='test_seen',
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
            extend_angle=cfgs.extend_angle,
        )
        cdf_label_folder = os.environ.get(
            "CDF_LABEL_FOLDER",
            "economic_grasp_label_300views_extend_angle_cdf_depth",
        )
        self.TRAIN_DATASET = CDFLabelAdapter(
            train_base_dataset,
            dataset_root=cfgs.dataset_root,
            label_folder=cdf_label_folder,
        )
        self.TEST_DATASET = CDFLabelAdapter(
            test_base_dataset,
            dataset_root=cfgs.dataset_root,
            label_folder=cdf_label_folder,
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

        self.net = economicgrasp_dpt(
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            is_training=True,
            use_obs_depth=bool(getattr(cfgs, 'use_obs_depth', False)),
            use_depth_comp=bool(getattr(cfgs, 'use_depth_comp', False)),
            vis_dir=getattr(cfgs, 'vis_dir', None) if self.main else None,
            vis_every=int(getattr(cfgs, 'vis_every', 1000)),
        )
        self.net.to(self.device)

        if self.distributed:
            # IMPORTANT: keep device_ids=None.
            #
            # With device_ids=[local_rank], PyTorch DDP recursively applies
            # _to_kwargs() to every forward input. That silently moves the
            # full variable-length object-level CDF/width label lists to CUDA,
            # even though move_batch_to_device() deliberately keeps them on
            # CPU. process_grasp_labels_cdf_width() then correctly fails its
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
                    'explicitly; full object-level CDF labels remain on CPU.'
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
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        overall_loss_local = 0.0
        local_batches = 0
        num_batches_local = len(self.TRAIN_DATALOADER)
        batch_interval = 20

        stat_interval = MetricAverager()
        interval_data_time = 0.0
        interval_model_time = 0.0
        interval_loss_time = 0.0
        interval_opt_time = 0.0
        interval_start_time = time.perf_counter()
        data_start_time = time.perf_counter()

        for batch_idx, batch_data_label in enumerate(self.TRAIN_DATALOADER):
            t = time.time()

            batch_data_label = move_batch_to_device(
                batch_data_label,
                self.device,
                non_blocking=False,
            )
            batch_data_label["cva_compute_diagnostics"] = (
                self.cdf_diag_interval > 0
                and batch_idx % self.cdf_diag_interval == 0
            )
            batch_data_label["cva_export_angle_feature"] = False

            data_end_time = time.perf_counter()
            interval_data_time += (data_end_time - data_start_time)

            model_start_time = time.perf_counter()
            assert_cpu_resident_label_lists(batch_data_label)
            end_points = self.net(batch_data_label)
            model_end_time = time.perf_counter()
            interval_model_time += (model_end_time - model_start_time)

            end_points['epoch'] = epoch

            loss_start_time = time.perf_counter()
            loss, end_points = get_loss_economicgrasp(end_points)
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
            if 'A: Grasp Loss' in metrics:
                grasp_loss_val = metrics['A: Grasp Loss']
            else:
                grasp_loss_val = reduce_scalar(
                    end_points['A: Grasp Loss'],
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
        self.log_string(f'overall grasp loss per batch: {overall_loss}, batch num:{overall_count}')
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
            if batch_idx % 50 == 0:
                print(
                    f"[rank{self.rank}] Eval batch "
                    f"{batch_idx}/{len(self.TEST_DATALOADER)}",
                    flush=True,
                )

            batch_data_label = move_batch_to_device(
                batch_data_label,
                self.device,
                non_blocking=False,
            )
            batch_data_label["cva_compute_diagnostics"] = (
                self.cdf_eval_diag_interval > 0
                and batch_idx % self.cdf_eval_diag_interval == 0
            )
            batch_data_label["cva_export_angle_feature"] = False

            with torch.no_grad():
                assert_cpu_resident_label_lists(batch_data_label)
                end_points = self.net(batch_data_label)
                loss, end_points = get_loss_economicgrasp(end_points)

            metrics = self.extract_scalar_metrics(end_points)
            for key, val in metrics.items():
                stat_sums_local[key] = stat_sums_local.get(key, 0.0) + float(val)
                stat_counts_local[key] = stat_counts_local.get(key, 0) + 1

            overall_loss_local += float(end_points['A: Grasp Loss'].detach().item())
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
        torch.save(self.unwrap_model().state_dict(), os.path.join(cfgs.log_dir, ckpt_name + '.tar'))

    def save_checkpoint(self, epoch, save_interval=False):
        if not self.main:
            return
        save_dict = {
            'epoch': epoch + 1,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.unwrap_model().state_dict(),
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
        trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
