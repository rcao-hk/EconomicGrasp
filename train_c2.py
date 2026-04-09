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
# from models.economicgrasp_depth_c1 import economicgrasp_c1, economicgrasp_c2_1, economicgrasp_c2_2, economicgrasp_c2_3, economicgrasp_c2_4
# from models.economicgrasp_c4 import economicgrasp_c4
# from models.economicgrasp_c5 import economicgrasp_c5
# from models.economicgrasp_c5 import economicgrasp_c5_1
# from models.economicgrasp_query import economicgrasp_query
from models.economicgrasp_bip3d import economicgrasp_bip3d
# from models.loss_economicgrasp_depth_c1 import get_loss as get_loss_economicgrasp
from models.loss_economicgrasp_depth_c1 import get_loss_c2_1 as get_loss_economicgrasp
# from models.loss_economicgrasp_depth_c1 import get_loss_c2_2 as get_loss_economicgrasp
# from models.loss_economicgrasp_c4 import get_loss as get_loss_economicgrasp
# from models.loss_economicgrasp_c5 import get_loss as get_loss_economicgrasp
# from models.loss_economicgrasp_c5 import get_loss_c5_1 as get_loss_economicgrasp
# from models.loss_economicgrasp_query import get_loss_query as get_loss_economicgrasp
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


def _sync():
    pass
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
        
# Create Dataset and Dataloader
TRAIN_DATASET = GraspNetMultiDataset(cfgs.dataset_root, camera=cfgs.camera, split='train', voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, remove_outlier=True, augment=False, use_gt_depth=cfgs.use_gt_depth, min_depth=cfgs.min_depth, max_depth=cfgs.max_depth, bin_num=cfgs.bin_num, depth_strides=1)
TEST_DATASET = GraspNetMultiDataset(cfgs.dataset_root, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, remove_outlier=True, augment=False, voxel_size=cfgs.voxel_size, use_gt_depth=False, min_depth=cfgs.min_depth, max_depth=cfgs.max_depth, bin_num=cfgs.bin_num, depth_strides=1)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                              num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
# Init the model
# net = economicgrasp_c1(depth_stride=2,     # <-- your expectation: 224x224 tokens
#                  min_depth=cfgs.min_depth,
#                  max_depth=cfgs.max_depth,
#                  is_training=True)
# net = economicgrasp_c2_1(depth_stride=2,     # <-- your expectation: 224x224 tokens
#                         min_depth=cfgs.min_depth,
#                         max_depth=cfgs.max_depth,
#                         bin_num=cfgs.bin_num,
#                         is_training=True)
# net = economicgrasp_c2_2(depth_stride=2,     # <-- your expectation: 224x224 tokens
#                         min_depth=cfgs.min_depth,
#                         max_depth=cfgs.max_depth,
#                         is_training=True)
# net = economicgrasp_c2_3(min_depth=cfgs.min_depth,
#                         max_depth=cfgs.max_depth,
#                         is_training=True,
#                         vis_dir=os.path.join('vis', 'c2.3_new'),
#                         vis_every=1000)
# net = economicgrasp_c2_4(min_depth=cfgs.min_depth,
#                         max_depth=cfgs.max_depth,
#                         is_training=True)
# net = economicgrasp_c4(min_depth=cfgs.min_depth,
#                         max_depth=cfgs.max_depth,
#                         is_training=True,
#                         vis_dir=os.path.join('vis', 'c4'),
#                         # vis_dir=None,
#                         vis_every=1000,
#                         )
# net = economicgrasp_c5_1(min_depth=cfgs.min_depth,
#                         max_depth=cfgs.max_depth,
#                         is_training=True,
#                         vis_dir=os.path.join('vis', 'c5_1_dbg'),
#                         # vis_dir=None,
#                         vis_every=50,
#                         )
# net = economicgrasp_query(min_depth=cfgs.min_depth,
#                         max_depth=cfgs.max_depth,
#                         is_training=True,
#                         vis_dir=os.path.join('vis', 'query'),
#                         # vis_dir=None,
#                         vis_every=500,
#                         )
net = economicgrasp_bip3d(min_depth=cfgs.min_depth,
                        max_depth=cfgs.max_depth,
                        is_training=True,
                        vis_dir=os.path.join('vis', 'bip3d'),
                        # vis_dir=None,
                        vis_every=500,
                        )
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

# def _find_last_conv2d(m: nn.Module):
#     last = None
#     for mm in m.modules():
#         if isinstance(mm, nn.Conv2d):
#             last = mm
#     return last

# def dbg_depthhead_grads(model):
#     head = model.depth_net.dpt.depth_head
#     last_conv = _find_last_conv2d(head)
#     print("[DBG][grad] last_conv =", last_conv)
#     print("[DBG][grad] requires_grad =", last_conv.weight.requires_grad)
#     g = last_conv.weight.grad
#     print("[DBG][grad] grad is None?" , (g is None))
#     if g is not None:
#         print("[DBG][grad] grad abs mean =", float(g.abs().mean().item()))


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

    overall_loss = 0.0
    num_batches = len(TRAIN_DATALOADER)
    batch_interval = 20

    # interval accumulators for timing
    interval_data_time = 0.0
    interval_model_time = 0.0
    interval_loss_time = 0.0
    interval_opt_time = 0.0
    interval_start_time = time.perf_counter()
    data_start_time = time.perf_counter()

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].cuda(non_blocking=cfgs.pin_memory)
            else:
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=cfgs.pin_memory)

        _sync()
        data_end_time = time.perf_counter()
        data_time = data_end_time - data_start_time
        interval_data_time += data_time

        _sync()
        model_start_time = time.perf_counter()
        end_points = net(batch_data_label)
        _sync()
        model_end_time = time.perf_counter()
        model_time = model_end_time - model_start_time
        interval_model_time += model_time

        end_points['epoch'] = EPOCH_CNT

        _sync()
        loss_start_time = time.perf_counter()
        loss, end_points = get_loss_economicgrasp(end_points)
        _sync()
        loss_end_time = time.perf_counter()
        loss_time = loss_end_time - loss_start_time
        interval_loss_time += loss_time

        bwdopt_start_time = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        _sync()
        bwdopt_end_time = time.perf_counter()
        opt_time = bwdopt_end_time - bwdopt_start_time
        interval_opt_time += opt_time

        for key, val in end_points.items():
            if ('loss' in key) or key.startswith(('A:', 'B:', 'C:', 'D:')):
                if torch.is_tensor(val):
                    if val.numel() == 1:
                        stat_dict.setdefault(key, 0.0)
                        stat_dict[key] += val.item()
                    else:
                        # 非标量调试量跳过
                        continue
                else:
                    try:
                        stat_dict.setdefault(key, 0.0)
                        stat_dict[key] += float(val)
                    except Exception:
                        continue

        overall_loss += end_points['A: Grasp Loss'].item()

        if (batch_idx + 1) % batch_interval == 0:
            remain_batches = (cfgs.max_epoch - EPOCH_CNT) * num_batches - batch_idx - 1

            _sync()
            interval_time = time.perf_counter() - interval_start_time
            avg_batch_time = interval_time / batch_interval
            remain_time_h = remain_batches * avg_batch_time / 3600.0

            log_string(f' ---- epoch: {EPOCH_CNT},  batch: {batch_idx + 1} ----')

            # time metrics: already averaged over interval
            time_dict = {
                'C: Data Time': interval_data_time / batch_interval,
                'C: Model Time': interval_model_time / batch_interval,
                'C: Loss Time': interval_loss_time / batch_interval,
                'C: Bwd+Opt Time': interval_opt_time / batch_interval,
                'C: Remain Time (h)': remain_time_h,
            }

            global_step = (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size

            for key in sorted(stat_dict.keys()):
                val = stat_dict[key] / batch_interval
                log_writer.add_scalar('train_' + key, val, global_step)
                log_string(f'{key:<20}: {val:.4f}')
                stat_dict[key] = 0.0

            for key in sorted(time_dict.keys()):
                val = time_dict[key]
                log_writer.add_scalar('train_' + key, val, global_step)
                log_string(f'{key:<20}: {val:.4f}')

            # reset interval timers
            interval_data_time = 0.0
            interval_model_time = 0.0
            interval_loss_time = 0.0
            interval_opt_time = 0.0
            interval_start_time = time.perf_counter()

        _sync()
        data_start_time = time.perf_counter()

    overall_loss = overall_loss / float(batch_idx + 1)
    log_string(f'overall grasp loss per batch: {overall_loss}, batch num:{batch_idx+1}')
    return overall_loss


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
        for key, val in end_points.items():
            if ('loss' in key) or key.startswith(('A:', 'B:', 'C:', 'D:')):
                if torch.is_tensor(val):
                    if val.numel() == 1:
                        stat_dict.setdefault(key, 0.0)
                        stat_dict[key] += val.item()
                    else:
                        # 非标量调试量跳过
                        continue
                else:
                    try:
                        stat_dict.setdefault(key, 0.0)
                        stat_dict[key] += float(val)
                    except Exception:
                        continue
    
        overall_loss += end_points['A: Grasp Loss'].item()
        
    for key in sorted(stat_dict.keys()):
        log_writer.add_scalar('test_' + key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size)
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    overall_loss = overall_loss/float(cfgs.batch_size)
    log_string('overall loss:{}, batch num:{}'.format(overall_loss, batch_idx+1))
    return overall_loss


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
