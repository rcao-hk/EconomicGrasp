import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import pdb
import cv2

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.arguments import cfgs
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, \
    get_workspace_mask, remove_invisible_grasp_points

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class GraspNetDataset(Dataset):
    def __init__(self, root, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=False, remove_invisible=True,
                 augment=False, load_label=True):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))

        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []

        self.grasp_labels = {}

        for x in tqdm(self.sceneIds, desc='Loading the scene data and its labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                self.scenename.append(x.strip())  # strip is for removing the space at the beginning and the end
                self.frameid.append(img_num)

            if self.load_label:
                self.grasp_labels[x.strip()] = os.path.join(self.root, 'economic_grasp_label_300views', x + '_labels.npz')

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))

        # camera in
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:  # they are not the outliers, just the points far away from the objects
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['coordinates_for_voxel'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['seg'] = seg_sampled.astype(np.float32)

        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]

        graspness = np.load(self.graspnesspath[index])  # already remove outliers
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        # depth is in millimeters (mm), the transformed cloud is in meters (m).
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:  # they are not the outliers, just the points far away from the objects
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        graspness_sampled = graspness[idxs]
        objectness_label = seg_sampled.copy()
        segmentation_label = objectness_label.copy()
        objectness_label[objectness_label > 1] = 1

        object_poses_list = []
        grasp_points_list = []
        grasp_rotations_list = []
        grasp_depth_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        view_graspness_list = []
        top_view_index_list = []

        # load labels
        grasp_labels = np.load(self.grasp_labels[scene])

        points = grasp_labels['points']
        rotations = grasp_labels['rotations'].astype(np.int32)
        depth = grasp_labels['depth'].astype(np.int32)
        scores = grasp_labels['scores'].astype(np.float32) / 10.
        widths = grasp_labels['widths'].astype(np.float32) / 1000.
        topview = grasp_labels['topview'].astype(np.int32)
        view_graspness = grasp_labels['vgraspness'].astype(np.float32)
        pointid = grasp_labels['pointid']
        for i, obj_idx in enumerate(obj_idxs):
            object_poses_list.append(poses[:, :, i])
            grasp_points_list.append(points[pointid == i])
            grasp_rotations_list.append(rotations[pointid == i])
            grasp_depth_list.append(depth[pointid == i])
            grasp_scores_list.append(scores[pointid == i])
            grasp_widths_list.append(widths[pointid == i])
            view_graspness_list.append(view_graspness[pointid == i])
            top_view_index_list.append(topview[pointid == i])

        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        # [scene_points, 3 (coords)]
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        # [scene_points, 3 (rgb)]
        ret_dict['coordinates_for_voxel'] = cloud_sampled.astype(np.float32) / self.voxel_size
        # [scene_points, 3 (coords)]
        ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
        # [scene_points, 1 (graspness)]
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        # [scene_points, 1 (objectness)]
        ret_dict['segmentation_label'] = segmentation_label.astype(np.int64)
        # [scene_points, 1 (objectness)]
        ret_dict['object_poses_list'] = object_poses_list
        # list has a length of objects amount, each has size [3, 4] (pose matrix)
        ret_dict['grasp_points_list'] = grasp_points_list
        # list has a length of objects amount, each has size [object_points, 3 (coordinate)]
        ret_dict['grasp_rotations_list'] = grasp_rotations_list
        # list has a length of objects amount, each has size [object_points, 60 (view)]
        ret_dict['grasp_depth_list'] = grasp_depth_list
        # list has a length of objects amount, each has size [object_points, 60 (view)]
        ret_dict['grasp_widths_list'] = grasp_widths_list
        # list has a length of objects amount, each has size [object_points, 60 (view)]
        ret_dict['grasp_scores_list'] = grasp_scores_list
        # list has a length of objects amount, each has size [object_points, 60 (view)]
        ret_dict['view_graspness_list'] = view_graspness_list
        # list has a length of objects amount, each has size [object_points, 300 (view graspness)]
        ret_dict['top_view_index_list'] = top_view_index_list
        # list has a length of objects amount, each has size [object_points, top views index]

        return ret_dict


from torchvision import transforms
class GraspNetMultiDataset(Dataset):
    def __init__(self, root, camera='kinect', split='train', num_points=20000, voxel_size=0.005, remove_outlier=False, remove_invisible=True,
                 augment=False, load_label=True, use_gt_depth=False,
                 min_depth=0.2, max_depth=1.0, bin_num=256):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.use_gt_depth = use_gt_depth
        self.collision_labels = {}

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))

        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        # ---- multi-modal (same setting as your GraspNetMultiDataset) ----
        self.resize_shape = (448, 448)  # (H, W)
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.depth_prob_min = min_depth
        self.depth_prob_max = max_depth
        self.depth_prob_bins = bin_num
        self.depth_prob_strides = [2]   # 需要更省就改成 [4]
        self.depth_prob_valid_threshold = -1       # <0 表示不做阈值裁剪；loss 里用 weight 即可
        self.gt_factor_depth = None                # None -> 默认用 meta['factor_depth']；也可以强制 1000.0
        
        self.colorpath = []
        self.depthpath = []
        self.gtdepthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        self.gtgraspnesspath = []
        self.grasp_labels = {}

        for x in tqdm(self.sceneIds, desc='Loading the scene data and its labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.gtdepthpath.append(os.path.join(root, 'virtual_scenes', x, camera, str(img_num).zfill(4) + '_depth.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                self.gtgraspnesspath.append(os.path.join(root, 'virtual_graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

            if self.load_label:
                self.grasp_labels[x.strip()] = os.path.join(self.root, 'economic_grasp_label_300views', x + '_labels.npz')

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [ 0, 1, 0],
                                 [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_resized_idxs(self, flat_idxs, orig_hw):
        """flat_idxs in original (H*W) -> flat idx in resized (Hr*Wr)."""
        orig_h, orig_w = orig_hw
        Hr, Wr = self.resize_shape

        flat_idxs = np.asarray(flat_idxs, dtype=np.int64)
        flat_idxs = np.clip(flat_idxs, 0, orig_h * orig_w - 1)
        ys, xs = np.unravel_index(flat_idxs, (orig_h, orig_w))

        if orig_h > 1:
            scale_y = (Hr - 1) / float(orig_h - 1)
            new_y = np.rint(ys * scale_y).astype(np.int64)
        else:
            new_y = np.zeros_like(ys, dtype=np.int64)

        if orig_w > 1:
            scale_x = (Wr - 1) / float(orig_w - 1)
            new_x = np.rint(xs * scale_x).astype(np.int64)
        else:
            new_x = np.zeros_like(xs, dtype=np.int64)

        new_y = np.clip(new_y, 0, Hr - 1)
        new_x = np.clip(new_x, 0, Wr - 1)
        return (new_y * Wr + new_x).astype(np.int64)

    def resize_intrinsics(self, K, orig_hw):
        """Scale intrinsics K from orig_hw to self.resize_shape (Hr,Wr)."""
        orig_h, orig_w = orig_hw
        Hr, Wr = self.resize_shape

        sx = Wr / float(orig_w)
        sy = Hr / float(orig_h)

        K = K.astype(np.float32).copy()
        K[0, 0] *= sx  # fx
        K[1, 1] *= sy  # fy
        K[0, 2] *= sx  # cx
        K[1, 2] *= sy  # cy
        return K

    # def build_depth_prob_gt(self, depth_m_resized: np.ndarray):
    #     """
    #     depth_m_resized: (Hr,Wr) float32 meters, aligned with resized RGB (448x448)

    #     Returns:
    #       depth_prob_gt: (1, Nfeat, D) float32
    #       depth_prob_w:  (1, Nfeat) float32   (valid_ratio per patch, in [0,1])
    #     """
    #     Hr, Wr = self.resize_shape
    #     D = int(self.depth_prob_bins)
    #     min_d = float(self.depth_prob_min)
    #     max_d = float(self.depth_prob_max)

    #     depth = depth_m_resized.astype(np.float32)
    #     if depth.shape != (Hr, Wr):
    #         raise ValueError(f"depth_m_resized shape {depth.shape} != resize_shape {(Hr,Wr)}")

    #     # valid mask: >0 (and optionally < max_d)
    #     valid = depth > 0
    #     # 若你希望把 >max_d 的也当 invalid，可以打开：
    #     # valid = np.logical_and(valid, depth < max_d)

    #     # clip to [min_d, max_d] for interpolation (same spirit as BIP3D)
    #     depth_clip = np.clip(depth, min_d, max_d)

    #     # map depth to continuous bin coordinate t in [0, D-1]
    #     denom = (max_d - min_d) if (max_d > min_d) else 1.0
    #     t = (depth_clip - min_d) / denom * (D - 1)

    #     i0 = np.floor(t).astype(np.int64)
    #     i0 = np.clip(i0, 0, D - 1)
    #     i1 = np.clip(i0 + 1, 0, D - 1)

    #     w1 = (t - i0.astype(np.float32)).astype(np.float32)
    #     w0 = (1.0 - w1).astype(np.float32)

    #     # invalid -> no supervision (weight=0), avoid scatter OOB
    #     inv = ~valid
    #     i0[inv] = 0
    #     i1[inv] = 0
    #     w0[inv] = 0.0
    #     w1[inv] = 0.0

    #     ys, xs = np.indices((Hr, Wr))
    #     pid_flat_cache = {}  # stride -> pid_flat

    #     flat_i0 = i0.reshape(-1)
    #     flat_i1 = i1.reshape(-1)
    #     flat_w0 = w0.reshape(-1)
    #     flat_w1 = w1.reshape(-1)
    #     flat_valid = valid.reshape(-1).astype(np.float32)

    #     probs_all = []
    #     weights_all = []

    #     for s in self.depth_prob_strides:
    #         if Hr % s != 0 or Wr % s != 0:
    #             raise ValueError(f"resize_shape {(Hr,Wr)} must be divisible by stride {s}")

    #         Hs, Ws = Hr // s, Wr // s
    #         n_patch = Hs * Ws

    #         if s in pid_flat_cache:
    #             pid_flat = pid_flat_cache[s]
    #         else:
    #             pid = (ys // s) * Ws + (xs // s)     # (Hr,Wr)
    #             pid_flat = pid.reshape(-1).astype(np.int64)
    #             pid_flat_cache[s] = pid_flat

    #         # patch distribution: (n_patch, D)
    #         prob = np.zeros((n_patch, D), dtype=np.float32)

    #         # accumulate two-bin weights into patch bins
    #         np.add.at(prob, (pid_flat, flat_i0), flat_w0)
    #         np.add.at(prob, (pid_flat, flat_i1), flat_w1)

    #         # average over pixels in patch (so it's a distribution over the patch)
    #         prob /= float(s * s)

    #         # valid_ratio per patch (for loss weight)
    #         vcnt = np.bincount(pid_flat, weights=flat_valid, minlength=n_patch).astype(np.float32)
    #         vratio = vcnt / float(s * s)

    #         # optional thresholding (same semantics as BIP3D: <threshold -> invalid)
    #         if self.depth_prob_valid_threshold >= 0:
    #             keep = vratio >= float(self.depth_prob_valid_threshold)
    #             prob[~keep] = 0.0
    #             vratio[~keep] = 0.0

    #         probs_all.append(prob)
    #         weights_all.append(vratio)

    #     depth_prob_gt = np.concatenate(probs_all, axis=0)   # (Nfeat, D)
    #     depth_prob_w  = np.concatenate(weights_all, axis=0) # (Nfeat,)

    #     # add view dim=1 for compatibility: (1, Nfeat, D) / (1, Nfeat)
    #     return depth_prob_gt[None].astype(np.float32), depth_prob_w[None].astype(np.float32)

    def build_depth_prob_gt(self, depth_m_resized: np.ndarray):
        """
        depth_m_resized: (448,448) float32 meters (aligned with resized RGB)
        Returns:
        depth_prob_gt: (1, Nfeat, 256) float32, Nfeat=224*224
        depth_prob_w : (1, Nfeat) float32, valid_ratio per 2x2 patch
        """
        Hr, Wr = self.resize_shape              # (448,448)
        s = 2
        Ht, Wt = Hr // s, Wr // s               # (224,224)
        Nfeat = Ht * Wt
        D = int(self.depth_prob_bins)
        min_d, max_d = float(self.depth_prob_min), float(self.depth_prob_max)

        depth = depth_m_resized.astype(np.float32)
        valid = depth > 0

        depth_clip = np.clip(depth, min_d, max_d)
        t = (depth_clip - min_d) / (max_d - min_d + 1e-12) * (D - 1)

        i0 = np.floor(t).astype(np.int64)
        i0 = np.clip(i0, 0, D - 1)
        i1 = np.clip(i0 + 1, 0, D - 1)

        w1 = (t - i0.astype(np.float32)).astype(np.float32)
        w0 = (1.0 - w1).astype(np.float32)

        # invalid -> contribute nothing
        inv = ~valid
        i0[inv] = 0
        i1[inv] = 0
        w0[inv] = 0.0
        w1[inv] = 0.0

        # patch id for each pixel: pid in [0, Nfeat)
        ys, xs = np.indices((Hr, Wr))
        pid = (ys // s) * Wt + (xs // s)      # (Hr,Wr)
        pid_flat = pid.reshape(-1).astype(np.int64)

        flat_i0 = i0.reshape(-1)
        flat_i1 = i1.reshape(-1)
        flat_w0 = w0.reshape(-1)
        flat_w1 = w1.reshape(-1)
        flat_valid = valid.reshape(-1).astype(np.float32)

        # accumulate
        prob = np.zeros((Nfeat, D), dtype=np.float32)
        np.add.at(prob, (pid_flat, flat_i0), flat_w0)
        np.add.at(prob, (pid_flat, flat_i1), flat_w1)

        # valid count per patch
        vcnt = np.bincount(pid_flat, weights=flat_valid, minlength=Nfeat).astype(np.float32)

        # normalize distribution by valid pixels (match softmax)
        den = np.maximum(vcnt, 1.0)
        prob = prob / den[:, None]
        prob[vcnt < 1.0] = 0.0

        # weight = valid_ratio in [0,1]
        vratio = vcnt / float(s * s)          # (Nfeat,)

        return prob[None].astype(np.float32), vratio[None].astype(np.float32)

    def _mask_and_sample(self, depth, seg, cloud, mask):
        """return sampled cloud/color/seg + sampled pixel flatten indices in original image"""
        H, W = depth.shape
        valid_flat = np.flatnonzero(mask)          # (mask_sum,)
        cloud_masked = cloud[mask]                # (mask_sum, 3)
        seg_masked = seg[mask]                    # (mask_sum,)
        return H, W, valid_flat, cloud_masked, seg_masked

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0  # (H,W,3)
        depth = np.array(Image.open(self.depthpath[index]))                            # (H,W)
        seg = np.array(Image.open(self.labelpath[index]))                              # (H,W)
        gt_depth = np.array(Image.open(self.gtdepthpath[index]))
        
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e)); print(scene)

        if self.use_gt_depth:
            depth = gt_depth

        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask

        H, W, valid_flat, cloud_masked, seg_masked = self._mask_and_sample(depth, seg, cloud, mask)
        color_masked = color[mask]

        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]

        # multi-modal: full image input + point->pixel idxs
        pix_flat = valid_flat[idxs]                            # original H*W flatten index
        resized_idxs = self.get_resized_idxs(pix_flat, (H, W)) # resized (448*448) flatten index
        img = self.img_transforms(color)                       # (3,448,448)
        
        # ---- GT depth (virtual) -> meters -> resize to 448x448 ----
        # 注意：gt_depth 可能是 uint16 (mm / factor_depth)
        fd = float(self.gt_factor_depth) if (self.gt_factor_depth is not None) else float(factor_depth)
        gt_depth_m = gt_depth.astype(np.float32) / fd
        K_resized = self.resize_intrinsics(intrinsic, (depth.shape[0], depth.shape[1]))
        
        Hr, Wr = self.resize_shape
        gt_depth_m_resized = cv2.resize(gt_depth_m, (Wr, Hr), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        depth_prob_gt, depth_prob_w = self.build_depth_prob_gt(gt_depth_m_resized)

        ret_dict = {
            'point_clouds': cloud_sampled.astype(np.float32),
            'cloud_colors': color_sampled.astype(np.float32),
            'coordinates_for_voxel': (cloud_sampled.astype(np.float32) / self.voxel_size),
            'seg': seg_sampled.astype(np.float32),
            'img': img,
            'img_idxs': resized_idxs.astype(np.int64),
            'K': K_resized,                       # (3,3) float32
            'gt_depth_m': gt_depth_m_resized,          # (448,448) float32 meters
            'depth_prob_gt': depth_prob_gt,            # (1, Nfeat, 256)
            'depth_prob_weight': depth_prob_w,         # (1, Nfeat)
        }
        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        gt_depth = np.array(Image.open(self.gtdepthpath[index]))
        graspness = np.load(self.graspnesspath[index])  # 注意：这通常对应“某个mask”的有效点序列
        gt_graspness = np.load(self.gtgraspnesspath[index])
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e)); print(scene)

        if self.use_gt_depth:
            depth = gt_depth
            graspness = gt_graspness

        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask

        fd = float(self.gt_factor_depth) if (self.gt_factor_depth is not None) else float(factor_depth)
        gt_depth_m = gt_depth.astype(np.float32) / fd

        Hr, Wr = self.resize_shape
        gt_depth_m_resized = cv2.resize(gt_depth_m, (Wr, Hr), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        depth_prob_gt, depth_prob_w = self.build_depth_prob_gt(gt_depth_m_resized)
        # resize depth and intrinsics to 448x448
        K_resized = self.resize_intrinsics(intrinsic, (depth.shape[0], depth.shape[1]))

        H, W, valid_flat, cloud_masked, seg_masked = self._mask_and_sample(depth, seg, cloud, mask)
        color_masked = color[mask]

        # sample
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]

        # graspness 对齐：默认按 “cloud_masked 的顺序” 直接取（跟你原实现一致）
        graspness_sampled = graspness[idxs]

        objectness_label = seg_sampled.copy()
        segmentation_label = objectness_label.copy()
        objectness_label[objectness_label > 1] = 1

        # multi-modal: full image + idxs
        pix_flat = valid_flat[idxs]
        resized_idxs = self.get_resized_idxs(pix_flat, (H, W))
        img = self.img_transforms(color)

        # load economic grasp labels
        grasp_labels = np.load(self.grasp_labels[scene])
        points = grasp_labels['points']
        rotations = grasp_labels['rotations'].astype(np.int32)
        depth_l = grasp_labels['depth'].astype(np.int32)
        scores = grasp_labels['scores'].astype(np.float32) / 10.
        widths = grasp_labels['widths'].astype(np.float32) / 1000.
        topview = grasp_labels['topview'].astype(np.int32)
        view_graspness = grasp_labels['vgraspness'].astype(np.float32)
        pointid = grasp_labels['pointid']

        object_poses_list = []
        grasp_points_list = []
        grasp_rotations_list = []
        grasp_depth_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        view_graspness_list = []
        top_view_index_list = []

        for i, obj_idx in enumerate(obj_idxs):
            object_poses_list.append(poses[:, :, i])
            grasp_points_list.append(points[pointid == i])
            grasp_rotations_list.append(rotations[pointid == i])
            grasp_depth_list.append(depth_l[pointid == i])
            grasp_scores_list.append(scores[pointid == i])
            grasp_widths_list.append(widths[pointid == i])
            view_graspness_list.append(view_graspness[pointid == i])
            top_view_index_list.append(topview[pointid == i])

        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        ret_dict = {
            'point_clouds': cloud_sampled.astype(np.float32),
            'cloud_colors': color_sampled.astype(np.float32),
            'coordinates_for_voxel': (cloud_sampled.astype(np.float32) / self.voxel_size),

            'img': img,
            'img_idxs': resized_idxs.astype(np.int64),

            'graspness_label': graspness_sampled.astype(np.float32),
            'objectness_label': objectness_label.astype(np.int64),
            'segmentation_label': segmentation_label.astype(np.int64),

            'object_poses_list': object_poses_list,
            'grasp_points_list': grasp_points_list,
            'grasp_rotations_list': grasp_rotations_list,
            'grasp_depth_list': grasp_depth_list,
            'grasp_widths_list': grasp_widths_list,
            'grasp_scores_list': grasp_scores_list,
            'view_graspness_list': view_graspness_list,
            'top_view_index_list': top_view_index_list,
            'sampled_masked_idxs': idxs.astype(np.int64),  # (num_points,) index into cloud_masked/graspness/seg_masked
            'pix_flat': pix_flat.astype(np.int64),         # (num_points,) original H*W flat index (optional, debug)
            'K': K_resized,                       # (3,3) float32
            'gt_depth_m': gt_depth_m_resized,          # (448,448) float32 meters
            'depth_prob_gt': depth_prob_gt,            # (1, Nfeat, 256)
            'depth_prob_weight': depth_prob_w,         # (1, Nfeat)
        }
        return ret_dict


# def collate_fn(batch):
#     if type(batch[0]).__module__ == 'numpy':
#         return torch.stack([torch.from_numpy(b) for b in batch], 0)
#     elif isinstance(batch[0], container_abcs.Mapping):
#         return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
#     elif isinstance(batch[0], container_abcs.Sequence):
#         return [[torch.from_numpy(sample) for sample in b] for b in batch]

#     raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))
def collate_fn(batch):
    elem = batch[0]

    # 1) torch tensor -> stack
    if torch.is_tensor(elem):
        return torch.stack(batch, 0)

    # 2) numpy array -> from_numpy + stack
    if isinstance(elem, np.ndarray):
        return torch.stack([torch.from_numpy(b) for b in batch], 0)

    # 3) numbers -> tensor
    if isinstance(elem, (float, int, np.number)):
        return torch.tensor(batch)

    # 4) dict -> recurse
    if isinstance(elem, container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}

    # 5) list/tuple -> keep per-sample lists (for variable-length), but convert numpy inside
    if isinstance(elem, container_abcs.Sequence) and not isinstance(elem, (str, bytes)):
        out = []
        for b in batch:  # b is one sample's list
            cur = []
            for x in b:
                if torch.is_tensor(x):
                    cur.append(x)
                elif isinstance(x, np.ndarray):
                    cur.append(torch.from_numpy(x))
                else:
                    cur.append(x)
            out.append(cur)
        return out

    raise TypeError(f"batch must contain tensors, numpy arrays, dicts or lists; found {type(elem)}")