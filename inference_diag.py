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
    
    # from models.economicgrasp_depth import economicgrasp_depth_baseline, pred_decode
    # net = economicgrasp_depth_baseline(seed_feat_dim=512, 
    #                                    depth_stride=1, 
    #                                    min_depth=cfgs.min_depth, 
    #                                    max_depth=cfgs.max_depth, 
    #                                    is_training=False,
    #                                    use_obs_depth=cfgs.use_obs_depth, 
    #                                    vis_dir=cfgs.vis_dir, 
    #                                    vis_every=cfgs.vis_every)
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
    
    # from models.economicgrasp_bip3d import economicgrasp_dpt
    # from models.economicgrasp_bip3d import pred_decode_center_view_angle as pred_decode
    # from models.economicgrasp_bip3d import economicgrasp_dpt_rotnet as economicgrasp_dpt
    # from models.economicgrasp_bip3d import pred_decode_rotnet_cva as pred_decode
    
    from models.economicgrasp_bip3d import economicgrasp_dpt
    from models.economicgrasp_bip3d import pred_decode_center_view_angle_diag as pred_decode
    # # # from models.economicgrasp_bip3d import pred_decode_collision as pred_decode
    # # # from models.economicgrasp_bip3d import pred_decode_collision_filter as pred_decode
    # from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode
    # # from models.economicgrasp_depth_c1 import pred_decode
    
    # from models.economicgrasp_dpt_ray import economicgrasp_dpt_ray as economicgrasp_dpt
    # from models.economicgrasp_dpt_ray import pred_decode_ray as pred_decode
    net = economicgrasp_dpt(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_obs_depth=getattr(cfgs, 'use_obs_depth', False),
        vis_dir=getattr(cfgs, 'vis_dir', None),
        vis_every=getattr(cfgs, 'vis_every', 1000),)
    # from models.economicgrasp_bip3d import economicgrasp_dpt_direct
    # from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode
    # net = economicgrasp_dpt_direct(
    #     min_depth=cfgs.min_depth,
    #     max_depth=cfgs.max_depth,
    #     bin_num=cfgs.bin_num,
    #     is_training=False,
    #     vis_dir=os.path.join('vis', 'dpt_view_attn_direct_test'),
    #     vis_every=500)
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


import csv
import json


def _mode_save_root(base_dir: str, mode: str) -> str:
    mode = str(mode)
    if mode in ["base", "orig", "original"]:
        return base_dir
    return base_dir + "_" + mode


def _save_sample_manifest(root: str, sampled_indices, sample_interval, test_mode, camera):
    os.makedirs(root, exist_ok=True)
    out_path = os.path.join(root, "_sampled_indices.json")
    payload = {
        "test_mode": str(test_mode),
        "camera": str(camera),
        "sample_interval": float(sample_interval),
        "num_samples": int(len(sampled_indices)),
        "sampled_indices": [int(x) for x in sampled_indices],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_debug_csv(rows, out_path):
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def inference():
    batch_interval = 20
    net.eval()
    tic = time.time()

    all_debug_rows = []
    modes_seen = None

    sample_interval = getattr(cfgs, "sample_interval", 1.0)
    if sample_interval < 1.0:
        print(
            f"[WARN] sample_interval={sample_interval}. "
            f"Use a clean save_dir to avoid mixing stale full-eval npy files."
        )

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

        with torch.no_grad():
            end_points = net(batch_data)

            # New diagnostic decoder returns dict: mode -> list[tensor per batch item].
            try:
                grasp_preds_by_mode = pred_decode(end_points, return_dict=True)
            except TypeError:
                # Backward compatibility.
                grasp_preds = pred_decode(end_points)
                grasp_preds_by_mode = {"base": grasp_preds}

        modes = list(grasp_preds_by_mode.keys())
        if modes_seen is None:
            modes_seen = modes
            for mode in modes:
                root = _mode_save_root(cfgs.save_dir, mode)
                _save_sample_manifest(
                    root=root,
                    sampled_indices=SAMPLED_INDICES,
                    sample_interval=sample_interval,
                    test_mode=cfgs.test_mode,
                    camera=cfgs.camera,
                )
            print(f"[rerank modes] {modes}")

        cur_bs = len(grasp_preds_by_mode[modes[0]])

        # Optional per-sample debug rows.
        debug_rows = end_points.get("cva_rerank_debug_rows", None)

        for i in range(cur_bs):
            subset_data_idx = batch_idx * cfgs.batch_size + i
            data_idx = SAMPLED_INDICES[subset_data_idx]
            scene_name = SCENE_LIST[data_idx]
            anno_id = data_idx % 256

            # Build GraspGroup for each scoring mode.
            gg_by_mode = {}
            for mode in modes:
                preds = grasp_preds_by_mode[mode][i].detach().cpu().numpy()
                gg_by_mode[mode] = GraspGroup(preds)

            # Save no-collision raw predictions if needed.
            if cfgs.save_nocollision:
                for mode, gg in gg_by_mode.items():
                    root = _mode_save_root(cfgs.save_dir, mode) + "_nocollision"
                    no_collision_dir = os.path.join(root, scene_name, cfgs.camera)
                    os.makedirs(no_collision_dir, exist_ok=True)
                    no_collision_path = os.path.join(no_collision_dir, str(anno_id).zfill(4) + ".npy")
                    gg.save_npy(no_collision_path)

            # Collision detection is pose-based, so reuse the same mask for all modes.
            # This is valid because all diagnostic modes keep the same decoded poses and only change score.
            collision_mask = None
            if cfgs.collision_thresh > 0:
                cloud, _ = FULL_TEST_DATASET.get_data(data_idx, return_raw_cloud=True)

                ref_mode = modes[0]
                mfcdetector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=cfgs.collision_voxel_size
                )
                collision_mask = mfcdetector.detect(
                    gg_by_mode[ref_mode],
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh
                )
                collision_mask = collision_mask.detach().cpu().numpy()

            # Save mode-specific predictions.
            for mode, gg in gg_by_mode.items():
                if collision_mask is not None:
                    gg = gg[~collision_mask]

                root = _mode_save_root(cfgs.save_dir, mode)
                save_dir = os.path.join(root, scene_name, cfgs.camera)
                save_path = os.path.join(save_dir, str(anno_id).zfill(4) + ".npy")
                os.makedirs(save_dir, exist_ok=True)
                gg.save_npy(save_path)

            # Append debug row.
            if isinstance(debug_rows, list) and i < len(debug_rows):
                row = dict(debug_rows[i])
                row.update({
                    "split": str(cfgs.test_mode),
                    "scene": str(scene_name),
                    "data_idx": int(data_idx),
                    "anno_id": int(anno_id),
                })
                all_debug_rows.append(row)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            denom = max(batch_interval, 1)
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc - tic) / denom))
            tic = time.time()

    # Save debug CSV under the base save_dir.
    if bool(getattr(cfgs, "save_rerank_diag", True)):
        out_csv = os.path.join(
            cfgs.save_dir,
            f"_cva_rerank_diag_{cfgs.test_mode}_si{getattr(cfgs, 'sample_interval', 1.0)}.csv"
        )
        _save_debug_csv(all_debug_rows, out_csv)
        print(f"[diag] saved {out_csv}")

if __name__ == '__main__':
    inference()