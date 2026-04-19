import os
import numpy as np
import time

import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup, GraspNetEval

from utils.collision_detector import ModelFreeCollisionDetector, ModelFreeCollisionDetectorTorch
from utils.arguments import cfgs

from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn

# ------------ GLOBAL CONFIG ------------
if not os.path.exists(cfgs.save_dir):
    os.mkdir(cfgs.save_dir)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def build_dataset(args):
    load_label = False   # 你当前测试脚本原本就是 False

    if args.multi_modal:
        dataset = GraspNetMultiDataset(
            args.dataset_root,
            split='{}'.format(args.test_mode),
            camera=args.camera,
            num_points=args.num_point,
            remove_outlier=True,
            augment=False,
            load_label=load_label
        )
    else:
        dataset = GraspNetDataset(
            args.dataset_root,
            split='{}'.format(args.test_mode),
            camera=args.camera,
            num_points=args.num_point,
            remove_outlier=True,
            augment=False,
            load_label=load_label
        )
    return dataset


def build_eval_subset(dataset, sample_interval, annos_per_scene=256):
    total = len(dataset)

    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")

    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices

    stride = max(1, int(round(1.0 / sample_interval)))
    indices = []

    num_scenes = (total + annos_per_scene - 1) // annos_per_scene
    for scene_id in range(num_scenes):
        start = scene_id * annos_per_scene
        end = min((scene_id + 1) * annos_per_scene, total)
        scene_len = end - start

        local_indices = list(range(0, scene_len, stride))
        indices.extend([start + idx for idx in local_indices])

    subset = Subset(dataset, indices)
    return subset, indices


def build_dataloader(args):
    full_dataset = build_dataset(args)
    sample_interval = getattr(args, "sample_interval", 1.0)
    eval_dataset, sampled_indices = build_eval_subset(full_dataset, sample_interval)

    test_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 2),
        worker_init_fn=my_worker_init_fn,
        collate_fn=collate_fn
    )
    return full_dataset, eval_dataset, test_dataloader, sampled_indices


# Create dataset and dataloader
FULL_TEST_DATASET, TEST_DATASET, TEST_DATALOADER, SAMPLED_INDICES = build_dataloader(cfgs)
SCENE_LIST = FULL_TEST_DATASET.scene_list()

print(f"Total test samples: {len(FULL_TEST_DATASET)}")
print(f"Evaluated samples:  {len(TEST_DATASET)}")
print(f"sample_interval:    {getattr(cfgs, 'sample_interval', 1.0)}")


# Init the model
if cfgs.multi_modal:
    # from models.economicgrasp import economicgrasp_multi, pred_decode
    # net = economicgrasp_multi(seed_feat_dim=512, fuse_type=cfgs.fuse_type, is_training=False, vis_dir=os.path.join('vis', 'eco_multi_{}_test'.format(cfgs.fuse_type)), vis_every=500)
    
    # from models.economicgrasp_depth import EconomicGrasp_RGBDepthProb, pred_decode
    # net = EconomicGrasp_RGBDepthProb(img_feat_dim=256,
    #              depth_stride=2,     # <-- your expectation: 224x224 tokens
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              bin_num=cfgs.bin_num, is_training=False)
    # from models.economicgrasp_depth_c1 import economicgrasp_c1, pred_decode
    # net = economicgrasp_c1(depth_stride=2,
    #                        min_depth=cfgs.min_depth,
    #                        max_depth=cfgs.max_depth,
    #                        is_training=False)
    # from models.economicgrasp_depth_c1 import economicgrasp_c2, pred_decode
    # net = economicgrasp_c2(depth_stride=2,     # <-- your expectation: 224x224 tokens
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              bin_num=cfgs.bin_num,
    #              is_training=False)
    # from models.economicgrasp_depth_c1 import economicgrasp_c2_1
    # from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode
    # net = economicgrasp_c2_1(depth_stride=2,     # <-- your expectation: 224x224 tokens
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              bin_num=cfgs.bin_num,
    #              is_training=False)
    # from models.economicgrasp_depth_c1 import economicgrasp_c2_2
    # from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode
    # net = economicgrasp_c2_2(depth_stride=2,     # <-- your expectation: 224x224 tokens
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              is_training=False)
    # from models.economicgrasp_depth_c1 import economicgrasp_c2_3
    # from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode
    # net = economicgrasp_c2_3(
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              is_training=False,
    #              vis_dir=os.path.join('vis', 'c2.3_test'),
    #              vis_every=1000)
    # from models.economicgrasp_depth_c1 import economicgrasp_c2_4, pred_decode
    # net = economicgrasp_c2_4(
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              is_training=False)
    # from models.economicgrasp_2d import economicgrasp_c3, pred_decode
    # net = economicgrasp_c3(
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              is_training=False,
    #              vis_every=100)
    # from models.economicgrasp_2d import economicgrasp_c3_1
    # from models.economicgrasp_2d import pred_decode_c3_1 as pred_decode
    # net = economicgrasp_c3_1(
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              is_training=False,
    #              vis_every=100)
    # from models.economicgrasp_2d import economicgrasp_c3_2
    # from models.economicgrasp_2d import pred_decode_c3_2 as pred_decode
    # net = economicgrasp_c3_2(
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              is_training=False,
    #              vis_dir=os.path.join('vis', 'c3.2_test'),
    #              vis_every=100)
    # from models.economicgrasp_c5 import economicgrasp_c5
    # from models.economicgrasp_c5 import pred_decode_c5 as pred_decode
    # net = economicgrasp_c5(
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              is_training=False,
    #              vis_dir=os.path.join('vis', 'c5_test'),
    #              vis_every=1000)
    # from models.economicgrasp_query import economicgrasp_query
    # from models.economicgrasp_query import pred_decode_query as pred_decode
    # net = economicgrasp_query(
    #              min_depth=cfgs.min_depth,
    #              max_depth=cfgs.max_depth,
    #              is_training=False,
    #              vis_dir=os.path.join('vis', 'query_test'),
    #              vis_every=500)
    # from models.economicgrasp_bip3d import economicgrasp_bip3d
    # from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode
    # net = economicgrasp_bip3d(
    #     min_depth=cfgs.min_depth,
    #     max_depth=cfgs.max_depth,
    #     bin_num=cfgs.bin_num,
    #     is_training=False,
    #     vis_dir=os.path.join('vis', 'bip3d_no_enhancer_test'),
    #     vis_every=500)
    from models.economicgrasp_bip3d import economicgrasp_dpt
    from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode
    net = economicgrasp_dpt(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        vis_dir=os.path.join('vis', 'dpt_test'),
        vis_every=500)
else:
    from models.economicgrasp import economicgrasp, pred_decode
    net = economicgrasp(seed_feat_dim=512, is_training=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
try:
    net.load_state_dict(checkpoint['model_state_dict'])
except:
    net.load_state_dict(checkpoint)

print("-> loaded checkpoint %s" % (cfgs.checkpoint_path))


# ------ Testing ------------
def inference():
    batch_interval = 20
    stat_dict = {}  # collect statistics
    net.eval()
    tic = time.time()

    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            elif 'graph' in key:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        cur_bs = len(grasp_preds)

        # Save results for evaluation
        for i in range(cur_bs):
            # 当前subset中的样本下标
            subset_data_idx = batch_idx * cfgs.batch_size + i
            # 映射回原始full dataset中的样本下标
            data_idx = SAMPLED_INDICES[subset_data_idx]

            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            if cfgs.save_nocollision:
                no_collision_dir = os.path.join(
                    cfgs.save_dir + '_nocollision',
                    SCENE_LIST[data_idx],
                    cfgs.camera
                )
                os.makedirs(no_collision_dir, exist_ok=True)
                no_collision_path = os.path.join(
                    no_collision_dir,
                    str(data_idx % 256).zfill(4) + '.npy'
                )
                gg.save_npy(no_collision_path)

            # collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = FULL_TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=cfgs.collision_voxel_size
                )
                collision_mask = mfcdetector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh
                )
                collision_mask = collision_mask.detach().cpu().numpy()
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.save_dir, SCENE_LIST[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            os.makedirs(save_dir, exist_ok=True)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc - tic) / batch_interval))
            tic = time.time()

if __name__ == '__main__':
    inference()