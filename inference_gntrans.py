import os
import time
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup

from utils.collision_detector import ModelFreeCollisionDetector
from utils.arguments import cfgs
from dataset.graspnet_dataset import GraspNetTransDataset, collate_fn


# ------------ GLOBAL CONFIG ------------
os.makedirs(cfgs.save_dir, exist_ok=True)


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _build_subset(
    dataset,
    sample_interval: float,
    annos_per_scene: int = 256,
) -> Tuple[torch.utils.data.Dataset, List[int]]:
    """Uniformly subsample every scene while preserving original dataset ids.

    ``sample_interval`` follows the repository-wide inference convention:

      * 1.0 -> use all frames;
      * 0.2 -> every 5th frame in each scene;
      * 0.1 -> every 10th frame in each scene.

    Sampling is performed independently inside each 256-frame scene so the
    resulting frame ids stay aligned with ``eval.py --sample_interval K``.
    """
    sample_interval = float(sample_interval)
    if not 0.0 < sample_interval <= 1.0:
        raise ValueError(
            "--sample_interval must be in (0, 1], where 1.0 means all frames "
            f"and 0.1 means every 10th frame; got {sample_interval}."
        )

    total = len(dataset)
    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices

    stride = max(1, int(round(1.0 / sample_interval)))
    indices: List[int] = []
    for start in range(0, total, annos_per_scene):
        end = min(start + annos_per_scene, total)
        indices.extend(range(start, end, stride))

    return Subset(dataset, indices), indices


# Create full dataset first.  Keep it for scene/frame lookup and optional
# collision detection; the DataLoader may operate on a sampled Subset.
FULL_TEST_DATASET = GraspNetTransDataset(
    cfgs.dataset_root,
    '/data/robotarm/dataset/GN-Trans',
    split='{}'.format(cfgs.test_mode),
    camera=cfgs.camera,
    num_points=cfgs.num_point,
    remove_outlier=True,
    augment=False,
    load_label=False,
)

TEST_DATASET, SAMPLED_INDICES = _build_subset(
    FULL_TEST_DATASET,
    float(getattr(cfgs, 'sample_interval', 1.0)),
)
SCENE_LIST = FULL_TEST_DATASET.scene_list()

TEST_DATALOADER = DataLoader(
    TEST_DATASET,
    batch_size=cfgs.batch_size,
    shuffle=False,
    num_workers=cfgs.num_workers,
    worker_init_fn=my_worker_init_fn,
    collate_fn=collate_fn,
)

print(
    '[GNTRANS-INFER] '
    f'split={cfgs.test_mode} '
    f'sample_interval={float(getattr(cfgs, "sample_interval", 1.0)):.6g} '
    f'total={len(FULL_TEST_DATASET)} selected={len(TEST_DATASET)}',
    flush=True,
)


# Init the model
# from models.economicgrasp_depth import EconomicGrasp_RGBDepthProb, pred_decode
# net = EconomicGrasp_RGBDepthProb(img_feat_dim=256,
#              depth_stride=2,
#              min_depth=cfgs.min_depth,
#              max_depth=cfgs.max_depth,
#              bin_num=cfgs.bin_num, is_training=False)

from models.economicgrasp_depth_c1 import economicgrasp_c1, pred_decode
net = economicgrasp_c1(
    depth_stride=2,
    min_depth=cfgs.min_depth,
    max_depth=cfgs.max_depth,
    is_training=False,
)

# from models.economicgrasp_depth_c1 import economicgrasp_c2, pred_decode
# net = economicgrasp_c2(depth_stride=2,
#              min_depth=cfgs.min_depth,
#              max_depth=cfgs.max_depth,
#              bin_num=cfgs.bin_num,
#              is_training=False)
# from models.economicgrasp_depth_c1 import economicgrasp_c2_1
# from models.economicgrasp_depth_c1 import pred_decode_c2_1 as pred_decode
# net = economicgrasp_c2_1(depth_stride=2,
#                 min_depth=cfgs.min_depth,
#                 max_depth=cfgs.max_depth,
#                 bin_num=cfgs.bin_num,
#                 is_training=False)
# from models.economicgrasp import economicgrasp_multi, pred_decode
# net = economicgrasp_multi(seed_feat_dim=512, is_training=False)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path, map_location='cpu')
try:
    net.load_state_dict(checkpoint['model_state_dict'])
except Exception:
    net.load_state_dict(checkpoint)

print('-> loaded checkpoint %s' % (cfgs.checkpoint_path))


# ------ Testing ------------
def inference():
    batch_interval = 20
    net.eval()
    tic = time.time()
    processed = 0

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

        # Save results for evaluation.  Map subset position back to the original
        # full-dataset index so scene name and frame id remain correct.
        for i, pred in enumerate(grasp_preds):
            subset_idx = batch_idx * cfgs.batch_size + i
            if subset_idx >= len(SAMPLED_INDICES):
                raise IndexError(
                    f'Subset index {subset_idx} exceeds sampled size '
                    f'{len(SAMPLED_INDICES)}.'
                )
            data_idx = SAMPLED_INDICES[subset_idx]

            preds = pred.detach().cpu().numpy()
            gg = GraspGroup(preds)

            # collision detection must use the corresponding frame in the full
            # dataset, not the compact Subset index.
            if cfgs.collision_thresh > 0:
                cloud, _ = FULL_TEST_DATASET.get_data(
                    data_idx,
                    return_raw_cloud=True,
                )
                mfcdetector = ModelFreeCollisionDetector(
                    cloud,
                    voxel_size=cfgs.voxel_size,
                )
                collision_mask = mfcdetector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                gg = gg[~collision_mask]

            save_dir = os.path.join(
                cfgs.save_dir,
                SCENE_LIST[data_idx],
                cfgs.camera,
            )
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f'{data_idx % 256:04d}.npy',
            )
            gg.save_npy(save_path)
            processed += 1

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print(
                f'[GNTRANS-INFER] batch={batch_idx}/{len(TEST_DATALOADER)} '
                f'samples={processed}/{len(TEST_DATASET)} '
                f'time_since_last={toc - tic:.3f}s',
                flush=True,
            )
            tic = time.time()

    print(
        f'[GNTRANS-INFER] completed samples={processed}/{len(TEST_DATASET)}',
        flush=True,
    )


if __name__ == '__main__':
    inference()
