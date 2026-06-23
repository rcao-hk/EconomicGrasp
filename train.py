# Basic Libraries
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import math
import time

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Config
from utils.arguments import cfgs

# Local Libraries
from models.economicgrasp import economicgrasp, economicgrasp_multi
# from models.economicgrasp_2d import EconomicGrasp_ImageCenter
# from models.loss_economicgrasp import get_loss as get_loss_economicgrasp
# from models.loss_economicgrasp_depth import get_loss as get_loss_economicgrasp
from models.economicgrasp_depth import economicgrasp_depth_baseline
from models.loss_economicgrasp_depth_c1 import get_loss as get_loss_economicgrasp
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn

# ----------- GLOBAL CONFIG ------------

# Epoch
EPOCH_CNT = 0

# Checkpoint path
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None

# Logging
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)
LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


# Create Dataset and Dataloader
# TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, camera=cfgs.camera, split='train',
#                                 voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, remove_outlier=True, augment=True)
if cfgs.multi_modal:
    TRAIN_DATASET = GraspNetMultiDataset(cfgs.dataset_root, camera=cfgs.camera, split='train', voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, remove_outlier=True, augment=False, use_gt_depth=cfgs.use_gt_depth, use_fuse_depth=cfgs.use_fuse_depth, min_depth=cfgs.min_depth, max_depth=cfgs.max_depth, bin_num=cfgs.bin_num, depth_strides=1)
    TEST_DATASET = GraspNetMultiDataset(cfgs.dataset_root, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, remove_outlier=True, augment=False, voxel_size=cfgs.voxel_size, use_gt_depth=False, use_fuse_depth=cfgs.use_fuse_depth, min_depth=cfgs.min_depth, max_depth=cfgs.max_depth, bin_num=cfgs.bin_num, depth_strides=1)
else:
    TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, camera=cfgs.camera, split='train',
                                    voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, remove_outlier=True, augment=True)
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, camera=cfgs.camera, split='test_seen',
                                    voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, remove_outlier=True, augment=False)
    
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                              num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
# Init the model
if cfgs.multi_modal:
    # net = economicgrasp(seed_feat_dim=512, is_training=True)
    # net = economicgrasp_multi(seed_feat_dim=512, fuse_type=cfgs.fuse_type, is_training=True, vis_dir=os.path.join('vis', 'eco_multi_{}'.format(cfgs.fuse_type)), vis_every=500)
    net = economicgrasp_depth_baseline(seed_feat_dim=512, depth_stride=1, min_depth=cfgs.min_depth, max_depth=cfgs.max_depth, is_training=True, use_obs_depth=cfgs.use_obs_depth, vis_dir=cfgs.vis_dir, vis_every=cfgs.vis_every)
else:
    net = economicgrasp(seed_feat_dim=512, is_training=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load the Adam optimizer
trainable_params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)

# Load checkpoint if there is any
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))


# cosine learning rate decay
def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (math.cos(epoch / cfgs.max_epoch * math.pi) + 1) * 0.5
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _scalar_to_float(x):
    if torch.is_tensor(x):
        if x.numel() != 1:
            return None
        return float(x.detach().item())
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return None


def _is_log_key(key: str):
    return key.startswith(("A:", "B:", "C:", "D:"))


def _accumulate_end_point_stats(stat_dict, end_points):
    for key, value in end_points.items():
        if not _is_log_key(key):
            continue
        v = _scalar_to_float(value)
        if v is None:
            continue
        stat_dict[key] = stat_dict.get(key, 0.0) + v


def _get_grasp_loss_value(end_points):
    if "A: Grasp Loss" not in end_points:
        raise KeyError(
            "end_points does not contain 'A: Grasp Loss'. "
            "Cannot use grasp loss for checkpoint selection."
        )
    v = _scalar_to_float(end_points["A: Grasp Loss"])
    if v is None:
        raise RuntimeError("'A: Grasp Loss' exists but is not a scalar.")
    return v


def _find_last_conv2d(m: nn.Module):
    last = None
    for mm in m.modules():
        if isinstance(mm, nn.Conv2d):
            last = mm
    return last

def dbg_depthhead_grads(model):
    head = model.depth_net.dpt.depth_head
    last_conv = _find_last_conv2d(head)
    print("[DBG][grad] last_conv =", last_conv)
    print("[DBG][grad] requires_grad =", last_conv.weight.requires_grad)
    g = last_conv.weight.grad
    print("[DBG][grad] grad is None?" , (g is None))
    if g is not None:
        print("[DBG][grad] grad abs mean =", float(g.abs().mean().item()))


def dbg_optimizer_has_depthhead(optimizer, model):
    head = model.depth_net.dpt.depth_head
    head_param_ids = set(id(p) for p in head.parameters())
    opt_param_ids = set()
    for g in optimizer.param_groups:
        for p in g["params"]:
            opt_param_ids.add(id(p))
    inter = head_param_ids & opt_param_ids
    print(f"[DBG][opt] depth_head params in optimizer: {len(inter)}/{len(head_param_ids)}")


# TensorBoard Visualizers
log_writer = SummaryWriter(os.path.join(cfgs.log_dir))
# ------TRAINING BEGIN  ------------
def train_one_epoch():
    stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)

    net.train()
    overall_grasp_loss = 0.0

    batch_interval = 20
    batch_start_time = time.time()
    data_start_time = time.time()
    num_batches = len(TRAIN_DATALOADER)

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].cuda(
                            non_blocking=cfgs.pin_memory
                        )
            else:
                batch_data_label[key] = batch_data_label[key].cuda(
                    non_blocking=cfgs.pin_memory
                )

        data_end_time = time.time()
        stat_dict['C: Data Time'] = stat_dict.get('C: Data Time', 0.0) + (
            data_end_time - data_start_time
        )

        model_start_time = time.time()
        end_points = net(batch_data_label)
        model_end_time = time.time()
        stat_dict['C: Model Time'] = stat_dict.get('C: Model Time', 0.0) + (
            model_end_time - model_start_time
        )

        end_points['epoch'] = EPOCH_CNT

        loss_start_time = time.time()

        optimizer.zero_grad(set_to_none=True)

        # loss can be total loss, e.g. grasp + depth.
        # checkpoint metric uses only A: Grasp Loss.
        loss, end_points = get_loss_economicgrasp(end_points)
        grasp_loss_value = _get_grasp_loss_value(end_points)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        loss_end_time = time.time()
        stat_dict['C: Loss Time'] = stat_dict.get('C: Loss Time', 0.0) + (
            loss_end_time - loss_start_time
        )

        overall_grasp_loss += grasp_loss_value

        # Accumulate scalar logs from end_points.
        _accumulate_end_point_stats(stat_dict, end_points)

        if (batch_idx + 1) % batch_interval == 0:
            interval_time = time.time() - batch_start_time
            batch_start_time = time.time()

            remain_batches = (cfgs.max_epoch - EPOCH_CNT) * num_batches - batch_idx - 1
            per_batch_time = interval_time / float(batch_interval)
            stat_dict['C: Remain Time (h)'] = remain_batches * per_batch_time / 3600.0

            log_string(f' ---- epoch: {EPOCH_CNT},  batch: {batch_idx + 1} ----')

            global_step = (
                EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx
            ) * cfgs.batch_size

            for key in sorted(stat_dict.keys()):
                # Time/metric logs are accumulated over batch_interval,
                # except remaining time, which is already an estimate.
                denom = 1.0 if key == 'C: Remain Time (h)' else float(batch_interval)
                value = stat_dict[key] / denom

                log_writer.add_scalar('train_' + key, value, global_step)
                log_string(f'{key:<24}: {round(value, 4):0<8}')

                stat_dict[key] = 0.0

        data_start_time = time.time()

    mean_grasp_loss = overall_grasp_loss / float(batch_idx + 1)
    log_string('overall grasp loss:{}, batch num:{}'.format(mean_grasp_loss, batch_idx + 1))

    return mean_grasp_loss


def evaluate_one_epoch():
    stat_dict = {}

    net.eval()
    overall_grasp_loss = 0.0

    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            log_string('Eval batch: %d' % batch_idx)

        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].cuda(
                            non_blocking=cfgs.pin_memory
                        )
            else:
                batch_data_label[key] = batch_data_label[key].cuda(
                    non_blocking=cfgs.pin_memory
                )

        with torch.no_grad():
            end_points = net(batch_data_label)
            end_points['epoch'] = EPOCH_CNT

            # loss may include depth loss, but checkpoint metric uses grasp loss only.
            loss, end_points = get_loss_economicgrasp(end_points)
            grasp_loss_value = _get_grasp_loss_value(end_points)

        overall_grasp_loss += grasp_loss_value

        _accumulate_end_point_stats(stat_dict, end_points)

    num_eval_batches = float(batch_idx + 1)

    global_step = (EPOCH_CNT + 1) * len(TRAIN_DATALOADER) * cfgs.batch_size

    for key in sorted(stat_dict.keys()):
        value = stat_dict[key] / num_eval_batches
        log_writer.add_scalar('test_' + key, value, global_step)
        log_string('eval mean %s: %f' % (key, value))

    mean_grasp_loss = overall_grasp_loss / num_eval_batches
    log_string('overall eval grasp loss:{}, batch num:{}'.format(mean_grasp_loss, batch_idx + 1))

    return mean_grasp_loss


def train(start_epoch):
    global EPOCH_CNT

    min_grasp_loss = np.inf
    best_epoch = 0

    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string(f'**** EPOCH {epoch:<3} ****')
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))

        np.random.seed()

        train_grasp_loss = train_one_epoch()

        save_dict = {
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
        }

        try:
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()

        if epoch >= cfgs.eval_start_epoch:
            eval_grasp_loss = evaluate_one_epoch()

            # Use grasp loss only for best-checkpoint selection.
            if eval_grasp_loss < min_grasp_loss:
                min_grasp_loss = eval_grasp_loss
                best_epoch = epoch

                ckpt_name = (
                    "epoch_" + str(best_epoch)
                    + "_train_grasp_" + str(train_grasp_loss)
                    + "_val_grasp_" + str(eval_grasp_loss)
                )

                # 保持你原来的保存方式：best 只保存 model_state_dict。
                torch.save(
                    save_dict['model_state_dict'],
                    os.path.join(cfgs.log_dir, ckpt_name + '.tar')
                )

            elif not EPOCH_CNT % cfgs.ckpt_save_interval:
                torch.save(
                    save_dict,
                    os.path.join(cfgs.log_dir, 'checkpoint_{}.tar'.format(EPOCH_CNT))
                )

            log_string("best_epoch:{}".format(best_epoch))
            log_string("best_eval_grasp_loss:{}".format(min_grasp_loss))

        torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))
        
        
if __name__ == '__main__':
    train(start_epoch)
