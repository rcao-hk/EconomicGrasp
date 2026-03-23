""" Testing for GraspNet baseline model. """

import os
import shutil
import numpy as np
import argparse

from graspnetAPI import GraspGroup, GraspNetEval

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet', help='Dataset root')
parser.add_argument('--dump_dir', default='ignet_v0.3.5.1', help='Dump dir to save outputs')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--split', default='test_seen', help='Test set split [test_seen/test_similar/test_novel]')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers used in evaluation [default: 30]')
parser.add_argument('--sample_interval', type=int, default=1,
                    help='Sample 1 frame every K frames in each scene (e.g., 10 means 0,10,20,...)')
parser.add_argument('--remove_dump', action='store_true', default=False,
                    help='If set, remove corresponding dump files after evaluation & saving AP npy.')
cfgs = parser.parse_args()
print(cfgs)


def _scene_range_for_split(split: str):
    # As requested:
    # seen:    scene_0100 - scene_0130
    # similar: scene_0130 - scene_0160
    # novel:   scene_0160 - scene_0190
    if split == 'test_seen':
        return range(100, 130)
    if split == 'test_similar':
        return range(130, 160)
    if split == 'test_novel':
        return range(160, 190)
    raise ValueError(f'Unknown split: {split}')


def _remove_dump_files(dump_dir: str, camera: str, split: str):
    scene_ids = _scene_range_for_split(split)
    removed, missing, failed = 0, 0, 0

    for sid in scene_ids:
        # In your inference script, saves to: dump_dir/scene_XXXX/<camera>/....
        cam_dir = os.path.join(dump_dir, f'scene_{sid:04d}', camera)
        if not os.path.exists(cam_dir):
            missing += 1
            continue
        try:
            shutil.rmtree(cam_dir)
            removed += 1
        except Exception as e:
            failed += 1
            print(f'[WARN] failed to remove: {cam_dir} | {e}')

    print(f'[CLEAN] split={split} camera={camera} removed={removed} missing={missing} failed={failed}')


def evaluate():
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split=cfgs.split)

    if cfgs.split == 'test_seen':
        res, ap = ge.eval_seen(os.path.join(cfgs.dump_dir), anno_sample_ratio=1/float(cfgs.sample_interval), proc=cfgs.num_workers)
    elif cfgs.split == 'test_similar':
        res, ap = ge.eval_similar(os.path.join(cfgs.dump_dir), anno_sample_ratio=1/float(cfgs.sample_interval), proc=cfgs.num_workers)
    else:
        res, ap = ge.eval_novel(os.path.join(cfgs.dump_dir), anno_sample_ratio=1/float(cfgs.sample_interval), proc=cfgs.num_workers)

    save_path = os.path.join(cfgs.dump_dir, f'ap_{cfgs.split}_{cfgs.camera}.npy')
    np.save(save_path, res)
    print(f'[SAVE] {save_path}')

    if cfgs.remove_dump:
        _remove_dump_files(cfgs.dump_dir, cfgs.camera, cfgs.split)


if __name__ == '__main__':
    evaluate()
