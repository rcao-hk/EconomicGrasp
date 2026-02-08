# Basic Libraries
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import math
import time

# PyTorch Libraries
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Config
from utils.arguments import cfgs

# Local Libraries
from models.economicgrasp_depth_c1 import economicgrasp_c1
# from models.loss_economicgrasp import get_loss as get_loss_economicgrasp
from models.loss_economicgrasp_depth import get_loss as get_loss_economicgrasp
from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn

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
TRAIN_DATASET = GraspNetMultiDataset(cfgs.dataset_root, camera=cfgs.camera, split='train', voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, remove_outlier=True, augment=False, use_gt_depth=cfgs.use_gt_depth, min_depth=cfgs.min_depth, max_depth=cfgs.max_depth, bin_num=cfgs.bin_num)
TEST_DATASET = GraspNetMultiDataset(cfgs.dataset_root, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, remove_outlier=True, augment=False, voxel_size=cfgs.voxel_size, use_gt_depth=False, min_depth=cfgs.min_depth, max_depth=cfgs.max_depth, bin_num=cfgs.bin_num)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                              num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
# Init the model
net = economicgrasp_c1(is_training=True)

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

import torch.nn as nn
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
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    # set model to training mode
    net.train()
    overall_loss = 0
    batch_start_time = time.time()
    data_start_time = time.time()
    num_batches = len(TRAIN_DATALOADER)
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].cuda(non_blocking=cfgs.pin_memory)
            else:
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=cfgs.pin_memory)
        data_end_time = time.time()
        stat_dict['C: Data Time'] = data_end_time - data_start_time

        model_start_time = time.time()
        end_points = net(batch_data_label)
        model_end_time = time.time()
        stat_dict['C: Model Time'] = model_end_time - model_start_time
        end_points['epoch'] = EPOCH_CNT

        loss_start_time = time.time()
        # Compute loss and gradients, update parameters.
        loss, end_points = get_loss_economicgrasp(end_points)

        loss.backward()
        # dbg_depthhead_grads(net)
        # dbg_optimizer_has_depthhead(optimizer, net)
        # if (batch_idx + 1) % 1 == 0:
        optimizer.step()
        optimizer.zero_grad()

        loss_end_time = time.time()
        stat_dict['C: Loss Time'] = loss_end_time - loss_start_time

        # Accumulate statistics and print out
        for key in end_points:
            if 'A' in key or 'B' in key or 'C' in key or 'D' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        overall_loss += stat_dict['A: Grasp Loss']
        batch_interval = 20

        if (batch_idx + 1) % batch_interval == 0:
            remain_batches = (cfgs.max_epoch - EPOCH_CNT) * num_batches - batch_idx - 1
            batch_time = time.time() - batch_start_time
            batch_start_time = time.time()
            stat_dict['C: Remain Time (h)'] = remain_batches * batch_time / 3600
            log_string(f' ---- epoch: {EPOCH_CNT},  batch: {batch_idx + 1} ----')
            for key in sorted(stat_dict.keys()):
                log_writer.add_scalar('train_' + key, stat_dict[key]/batch_interval, (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*cfgs.batch_size)
                log_string(f'{key:<20}: {round(stat_dict[key] / batch_interval, 4):0<8}')
                stat_dict[key] = 0

        data_start_time = time.time()
    overall_loss = overall_loss/float(cfgs.batch_size)
    log_string('overall loss:{}, batch num:{}'.format(overall_loss, batch_idx+1))
    mean_loss = overall_loss/float(batch_idx+1)
    return mean_loss


def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    overall_loss = 0
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            log_string('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].cuda(non_blocking=cfgs.pin_memory)
            else:
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=cfgs.pin_memory)
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data_label)

        # Compute loss
        loss, end_points = get_loss_economicgrasp(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'A' in key or 'B' in key or 'C' in key or 'D' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
    
        overall_loss += stat_dict['A: Grasp Loss']
    for key in sorted(stat_dict.keys()):
        log_writer.add_scalar('test_' + key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size)
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    overall_loss = overall_loss/float(cfgs.batch_size)
    log_string('overall loss:{}, batch num:{}'.format(overall_loss, batch_idx+1))
    mean_loss = overall_loss/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT
    min_loss = np.inf
    best_epoch = 0
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string(f'**** EPOCH {epoch:<3} ****')
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))

        np.random.seed()
        train_loss = train_one_epoch()

        # Save checkpoint
        save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                     'optimizer_state_dict': optimizer.state_dict(),
                     }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()

        if epoch >= cfgs.eval_start_epoch:
            eval_loss = evaluate_one_epoch()
            if eval_loss < min_loss:
                min_loss = eval_loss
                best_epoch = epoch
                ckpt_name = "epoch_" + str(best_epoch) \
                            + "_train_" + str(train_loss) \
                            + "_val_" + str(eval_loss)
                torch.save(save_dict['model_state_dict'], os.path.join(cfgs.log_dir, ckpt_name + '.tar'))
            elif not EPOCH_CNT % cfgs.ckpt_save_interval:
                torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint_{}.tar'.format(EPOCH_CNT)))
            log_string("best_epoch:{}".format(best_epoch))
        torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))

if __name__ == '__main__':
    train(start_epoch)
