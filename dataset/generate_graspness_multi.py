import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np

# np.int = np.int32
# np.float = np.float64
# np.bool = np.bool_

from PIL import Image
import scipy.io as scio
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
# from knn.knn_modules import knn
from pytorch3d.ops.knn import knn_points
import multiprocessing

import torch
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, transform_points
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


def generate_scene(scene_id, cfgs):
    dataset_root = cfgs.dataset_root   # set dataset root
    virtual_dataset_root = cfgs.virtual_dataset_root   # set virtual dataset root
    camera_type = cfgs.camera_type   # kinect / realsense
    if cfgs.depth_type == 'virtual':
        save_path_root = os.path.join(dataset_root, 'virtual_graspness')
    elif cfgs.depth_type == 'real':
        save_path_root = os.path.join(dataset_root, 'graspness')
    num_views, num_angles, num_depths = 300, 12, 4
    fric_coef_thresh = 0.8
    point_grasp_num = num_views * num_angles * num_depths
    for ann_id in range(256):
        # get scene point cloud
        print('generating scene: {} ann: {}'.format(scene_id, ann_id))
        if cfgs.depth_type == 'virtual':
            depth = np.array(Image.open(os.path.join(virtual_dataset_root, 'scene_' + str(scene_id).zfill(4),
                                                        camera_type, str(ann_id).zfill(4) + '_depth.png')))
        elif cfgs.depth_type == 'real':
            depth = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                        camera_type, 'depth', str(ann_id).zfill(4) + '.png')))
        seg = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'label', str(ann_id).zfill(4) + '.png')))
        meta = scio.loadmat(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'meta', str(ann_id).zfill(4) + '.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # remove outlier and get objectness label
        depth_mask = (depth > 0)
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'camera_poses.npy'))
        camera_pose = camera_poses[ann_id]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'cam0_wrt_table.npy'))
        trans = np.dot(align_mat, camera_pose)
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)
        cloud_masked = cloud[mask]
        objectness_label = seg[mask]

        # get scene object and grasp info
        scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'annotations', '%04d.xml' % ann_id))
        pose_vectors = scene_reader.getposevectorlist()
        obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)
        # print(obj_list)
        grasp_labels = {}
        for i in obj_list:
            file = np.load(os.path.join(dataset_root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
            grasp_labels[i] = (file['points'].astype(np.float32), file['offsets'].astype(np.float32),
                                file['scores'].astype(np.float32))

        labels = np.load(
            os.path.join(dataset_root, 'collision_label', 'scene_' + str(scene_id).zfill(4), 'collision_labels.npz'))
        collision_dump = []
        for j in range(len(labels)):
            collision_dump.append(labels['arr_{}'.format(j)])
        grasp_points = []
        grasp_points_graspness = []
        for i, (obj_idx, trans_) in enumerate(zip(obj_list, pose_list)):
            sampled_points, offsets, fric_coefs = grasp_labels[obj_idx]
            collision = collision_dump[i]  # Npoints * num_views * num_angles * num_depths
            num_points = sampled_points.shape[0]

            valid_grasp_mask = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision)
            valid_grasp_mask = valid_grasp_mask.reshape(num_points, -1)
            graspness = np.sum(valid_grasp_mask, axis=1) / point_grasp_num
            target_points = transform_points(sampled_points, trans_)
            target_points = transform_points(target_points, np.linalg.inv(camera_pose))  # fix bug
            grasp_points.append(target_points)
            grasp_points_graspness.append(graspness.reshape(num_points, 1))
        grasp_points = np.vstack(grasp_points)
        grasp_points_graspness = np.vstack(grasp_points_graspness)

        grasp_points = torch.from_numpy(grasp_points).cuda()
        grasp_points_graspness = torch.from_numpy(grasp_points_graspness).cuda()
        # grasp_points = grasp_points.transpose(0, 1).contiguous().unsqueeze(0)

        # masked_points_num = cloud_masked.shape[0]
        # cloud_masked_graspness = np.zeros((masked_points_num, 1))
        # part_num = int(masked_points_num / partition_num)
        # for i in range(1, part_num + 2):   # lack of cuda memory
        #     if i == part_num + 1:
        #         cloud_masked_partial = cloud_masked[partition_num * part_num:]
        #         if len(cloud_masked_partial) == 0:
        #             break
        #     else:
        #         cloud_masked_partial = cloud_masked[partition_num * (i - 1):(i * partition_num)]
        #     cloud_masked_partial = torch.from_numpy(cloud_masked_partial).cuda()
        #     # cloud_masked_partial = cloud_masked_partial.transpose(0, 1).contiguous().unsqueeze(0)
        #     # nn_inds = knn(grasp_points, cloud_masked_partial, k=1).squeeze() - 1
        #     cloud_masked_partial = cloud_masked_partial.contiguous().unsqueeze(0)
        #     _, nn_inds, _ = knn_points(cloud_masked_partial, grasp_points, K=1)
        #     nn_inds = nn_inds.squeeze(-1).squeeze(0)
    
        #     cloud_masked_graspness[partition_num * (i - 1):(i * partition_num)] = torch.index_select(
        #         grasp_points_graspness, 0, nn_inds).cpu().numpy()

        cloud_masked = torch.from_numpy(cloud_masked).cuda()
        grasp_points = grasp_points.contiguous().unsqueeze(0)
        cloud_masked = cloud_masked.contiguous().unsqueeze(0)
        _, nn_inds, _ = knn_points(cloud_masked, grasp_points, K=1)
        nn_inds = nn_inds.squeeze(-1).squeeze(0)
            
        cloud_masked_graspness = torch.index_select(grasp_points_graspness, 0, nn_inds).cpu().numpy()
        max_graspness = np.max(cloud_masked_graspness)
        min_graspness = np.min(cloud_masked_graspness)
        cloud_masked_graspness = (cloud_masked_graspness - min_graspness) / (max_graspness - min_graspness)
        save_path = os.path.join(save_path_root, 'scene_' + str(scene_id).zfill(4), camera_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cloud_masked_graspness_f16 = cloud_masked_graspness.astype(np.float16)
        np.save(os.path.join(save_path, str(ann_id).zfill(4) + '.npy'), cloud_masked_graspness_f16)


def parallel_generate(scene_ids, cfgs, proc = 2):
    # from multiprocessing import Pool
    ctx_in_main = multiprocessing.get_context('forkserver')
    p = ctx_in_main.Pool(processes = proc)
    for scene_id in scene_ids:
        p.apply_async(generate_scene, (scene_id, cfgs))
    p.close()
    p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/home/user/dataset/graspnet')
    parser.add_argument('--camera_type', default='realsense', help='Camera split [realsense/kinect]')
    parser.add_argument('--virtual_dataset_root', default='/home/user/dataset/graspnet/virtual_scenes')
    parser.add_argument('--depth_type', default='real', help='Depth type [virtual/real]')
    parser.add_argument('--proc', type=int, default=10, help='number of processes to use')
    cfgs = parser.parse_args()
    
    parallel_generate(list(range(9)), cfgs=cfgs, proc = cfgs.proc)