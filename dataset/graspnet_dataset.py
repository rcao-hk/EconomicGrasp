import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import pdb

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

        # ---- multi-modal (same setting as your GraspNetMultiDataset) ----
        self.resize_shape = (448, 448)  # (H, W)
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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
        """flat_idxs: flatten indices in original image (H*W). return flatten indices in resized image (Hr*Wr)."""
        orig_h, orig_w = orig_hw
        scale_x = self.resize_shape[1] / orig_w
        scale_y = self.resize_shape[0] / orig_h

        ys, xs = np.unravel_index(flat_idxs, (orig_h, orig_w))
        new_y = np.clip((ys * scale_y).astype(np.int64), 0, self.resize_shape[0] - 1)
        new_x = np.clip((xs * scale_x).astype(np.int64), 0, self.resize_shape[1] - 1)
        return (new_y * self.resize_shape[1] + new_x).astype(np.int64)

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

        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e)); print(scene)

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

        ret_dict = {
            'point_clouds': cloud_sampled.astype(np.float32),
            'cloud_colors': color_sampled.astype(np.float32),
            'coordinates_for_voxel': (cloud_sampled.astype(np.float32) / self.voxel_size),
            'seg': seg_sampled.astype(np.float32),
            'img': img,
            'img_idxs': resized_idxs.astype(np.int64),
        }
        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]

        graspness = np.load(self.graspnesspath[index])  # 注意：这通常对应“某个mask”的有效点序列

        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e)); print(scene)

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