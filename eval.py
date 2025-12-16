""" Testing for GraspNet baseline model. """

import os
import sys
import numpy as np
import argparse
import time

from graspnetAPI import GraspGroup, GraspNetEval

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet', help='Dataset root')
parser.add_argument('--dump_dir', default='ignet_v0.3.5.1', help='Dump dir to save outputs')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--split', default='test_seen', help='Test set split [test_seen/test_similar/test_novel]')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()
print(cfgs)


def evaluate():
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split=cfgs.split)
    if cfgs.split == 'test_seen':
        res, ap = ge.eval_seen(os.path.join(cfgs.dump_dir), proc=cfgs.num_workers)
    elif cfgs.split == 'test_similar':
        res, ap = ge.eval_similar(os.path.join(cfgs.dump_dir), proc=cfgs.num_workers)
    else:
        res, ap = ge.eval_novel(os.path.join(cfgs.dump_dir), proc=cfgs.num_workers)
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}_{}.npy'.format(cfgs.split, cfgs.camera))
    np.save(save_dir, res)

if __name__=='__main__':
    evaluate()