# Basic Libraries
import os
import math
import time
import random
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

# Local Libraries
from models.economicgrasp_bip3d import economicgrasp_bip3d, economicgrasp_dpt, economicgrasp_dpt_direct
from models.loss_economicgrasp_depth_c1 import get_loss_c2_1 as get_loss_economicgrasp

# from models.economicgrasp_query import economicgrasp_query
# from models.loss_economicgrasp_query import get_loss_query as get_loss_economicgrasp

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn

# ----------- GLOBAL CONFIG ------------
EPOCH_CNT = 0
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None


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
        dist.init_process_group(backend='nccl', init_method='env://')
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
        dist.barrier()
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if distributed and dist.is_initialized():
        dist.barrier()


def move_batch_to_device(batch_data_label, device, non_blocking=False):
    for key in batch_data_label:
        if 'list' in key:
            for i in range(len(batch_data_label[key])):
                for j in range(len(batch_data_label[key][i])):
                    batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device, non_blocking=non_blocking)
        else:
            if torch.is_tensor(batch_data_label[key]):
                batch_data_label[key] = batch_data_label[key].to(device, non_blocking=non_blocking)
    return batch_data_label


def reduce_scalar(value, device, distributed: bool, average: bool = True):
    if not torch.is_tensor(value):
        value = torch.tensor(float(value), device=device, dtype=torch.float32)
    else:
        value = value.detach().to(device=device, dtype=torch.float32)
    if distributed and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if average:
            value = value / dist.get_world_size()
    return value


def reduce_sum_and_count(local_sum: float, local_count: int, device, distributed: bool):
    buf = torch.tensor([float(local_sum), float(local_count)], device=device, dtype=torch.float64)
    if distributed and dist.is_initialized():
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return float(buf[0].item()), int(buf[1].item())


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

        os.makedirs(cfgs.log_dir, exist_ok=True)
        self.log_path = os.path.join(cfgs.log_dir, 'log_train.txt')
        self.LOG_FOUT = open(self.log_path, 'a') if self.main else None
        if self.main:
            self.LOG_FOUT.write(str(cfgs) + '\n')
            self.LOG_FOUT.flush()

        self.log_writer = SummaryWriter(os.path.join(cfgs.log_dir)) if self.main else None

        self.TRAIN_DATASET = GraspNetMultiDataset(
            cfgs.dataset_root,
            camera=cfgs.camera,
            split='train',
            voxel_size=cfgs.voxel_size,
            num_points=cfgs.num_point,
            remove_outlier=True,
            augment=False,
            use_gt_depth=cfgs.use_gt_depth,
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            depth_strides=1,
        )
        self.TEST_DATASET = GraspNetMultiDataset(
            cfgs.dataset_root,
            camera=cfgs.camera,
            split='test_seen',
            num_points=cfgs.num_point,
            remove_outlier=True,
            augment=False,
            voxel_size=cfgs.voxel_size,
            use_gt_depth=False,
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            depth_strides=1,
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
            pin_memory=cfgs.pin_memory,
            drop_last=False,
            persistent_workers=(cfgs.num_workers > 0),
        )
        self.TEST_DATALOADER = DataLoader(
            self.TEST_DATASET,
            batch_size=cfgs.batch_size,
            shuffle=False,
            sampler=self.test_sampler,
            num_workers=cfgs.num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=cfgs.pin_memory,
            drop_last=False,
            persistent_workers=(cfgs.num_workers > 0),
        )

        # self.net = economicgrasp_bip3d(
        #     min_depth=cfgs.min_depth,
        #     max_depth=cfgs.max_depth,
        #     bin_num=cfgs.bin_num,
        #     is_training=True,
        #     vis_dir=os.path.join('vis', 'bip3d') if self.main else None,
        #     vis_every=1000,
        # )
        self.net = economicgrasp_dpt(
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            is_training=True,
            vis_dir=os.path.join('vis', 'dpt_enhancer') if self.main else None,
            vis_every=1000,
        )
        # self.net = economicgrasp_dpt_direct(
        #     min_depth=cfgs.min_depth,
        #     max_depth=cfgs.max_depth,
        #     bin_num=cfgs.bin_num,
        #     is_training=True,
        #     vis_dir=os.path.join('vis', 'dpt_view_attn_direct') if self.main else None,
        #     vis_every=1000,
        # )
        
        # self.net = economicgrasp_query(
        #     min_depth=cfgs.min_depth,
        #     max_depth=cfgs.max_depth,
        #     is_training=True,
        #     vis_dir=os.path.join('vis', 'query') if self.main else None,
        #     vis_every=500,
        # )
        self.net.to(self.device)

        if self.distributed:
            self.net = DDP(
                self.net,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=cfgs.learning_rate,
            weight_decay=cfgs.weight_decay,
        )

        self.start_epoch = 0
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            self.unwrap_model().load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.log_string(f'-> loaded checkpoint {CHECKPOINT_PATH} (epoch: {self.start_epoch})')

    def unwrap_model(self):
        return self.net.module if hasattr(self.net, 'module') else self.net

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
            batch_data_label = move_batch_to_device(batch_data_label, self.device, non_blocking=cfgs.pin_memory)

            _sync(self.distributed)
            data_end_time = time.perf_counter()
            interval_data_time += (data_end_time - data_start_time)

            _sync(self.distributed)
            model_start_time = time.perf_counter()
            end_points = self.net(batch_data_label)
            _sync(self.distributed)
            model_end_time = time.perf_counter()
            interval_model_time += (model_end_time - model_start_time)

            end_points['epoch'] = epoch

            _sync(self.distributed)
            loss_start_time = time.perf_counter()
            loss, end_points = get_loss_economicgrasp(end_points)
            _sync(self.distributed)
            loss_end_time = time.perf_counter()
            interval_loss_time += (loss_end_time - loss_start_time)

            self.optimizer.zero_grad(set_to_none=True)
            bwdopt_start_time = time.perf_counter()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            _sync(self.distributed)
            bwdopt_end_time = time.perf_counter()
            interval_opt_time += (bwdopt_end_time - bwdopt_start_time)

            metrics = self.extract_scalar_metrics(end_points)
            for key, val in metrics.items():
                reduced_val = reduce_scalar(val, self.device, self.distributed, average=True).item()
                stat_interval.update_scalar(key, reduced_val)

            grasp_loss_val = reduce_scalar(end_points['A: Grasp Loss'], self.device, self.distributed, average=True).item()
            overall_loss_local += grasp_loss_val
            local_batches += 1

            if (batch_idx + 1) % batch_interval == 0:
                remain_batches_local = (cfgs.max_epoch - epoch) * num_batches_local - batch_idx - 1
                _sync(self.distributed)
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

            _sync(self.distributed)
            data_start_time = time.perf_counter()

        overall_sum, overall_count = reduce_sum_and_count(overall_loss_local, local_batches, self.device, self.distributed)
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
            if self.main and batch_idx % 10 == 0:
                self.log_string(f'Eval batch: {batch_idx}')

            batch_data_label = move_batch_to_device(batch_data_label, self.device, non_blocking=cfgs.pin_memory)

            with torch.no_grad():
                end_points = self.net(batch_data_label)
                loss, end_points = get_loss_economicgrasp(end_points)

            metrics = self.extract_scalar_metrics(end_points)
            for key, val in metrics.items():
                stat_sums_local[key] = stat_sums_local.get(key, 0.0) + float(val)
                stat_counts_local[key] = stat_counts_local.get(key, 0) + 1

            overall_loss_local += float(end_points['A: Grasp Loss'].detach().item())
            local_batches += 1

        reduced_metrics = {}
        for key in sorted(stat_sums_local.keys()):
            total_sum, total_count = reduce_sum_and_count(
                stat_sums_local[key], stat_counts_local[key], self.device, self.distributed
            )
            reduced_metrics[key] = total_sum / max(total_count, 1)

        overall_sum, overall_count = reduce_sum_and_count(overall_loss_local, local_batches, self.device, self.distributed)
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

            if epoch >= cfgs.eval_start_epoch:
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
