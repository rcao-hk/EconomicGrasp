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

# from utils.arguments import cfgs
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
                 min_depth=0.2, max_depth=1.0, bin_num=256, depth_strides=2):
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
        self.depth_prob_strides = depth_strides   # 需要更省就改成 [4]
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

    def _crop_box_from_mask(self, mask):
        H, W = mask.shape
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 0, 0, W, H
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        return int(x0), int(y0), int(x1), int(y1)
    
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

    def get_resized_idxs_from_flat_crop(self, pix_flat, orig_hw, crop_box, out_hw=(448,448)):
        H, W = orig_hw
        outH, outW = out_hw
        x0, y0, x1, y1 = crop_box
        cw, ch = (x1 - x0), (y1 - y0)

        ys, xs = np.unravel_index(pix_flat, (H, W))
        xs = xs.astype(np.float32) - float(x0)
        ys = ys.astype(np.float32) - float(y0)

        # clamp into crop
        xs = np.clip(xs, 0, cw - 1e-6)
        ys = np.clip(ys, 0, ch - 1e-6)

        # scale to 448 (use floor to match your model gather)
        xf = np.floor(xs * (outW / float(cw))).astype(np.int64)
        yf = np.floor(ys * (outH / float(ch))).astype(np.int64)
        xf = np.clip(xf, 0, outW - 1)
        yf = np.clip(yf, 0, outH - 1)
        return (yf * outW + xf).astype(np.int64)
    
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

    def resize_intrinsics_with_crop(self, intrinsic, crop_box, out_hw=(448,448)):
        x0, y0, x1, y1 = crop_box
        outH, outW = out_hw
        cw = float(x1 - x0)
        ch = float(y1 - y0)

        sx = outW / cw
        sy = outH / ch

        K = intrinsic.astype(np.float32).copy()
        K[0, 2] -= float(x0)
        K[1, 2] -= float(y0)

        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return K

    def build_resized_depth_m_with_crop(self, depth_raw, factor_depth, crop_box, out_hw=(448,448)):
        """
        depth_raw: (H,W) uint16 / float, sensor depth in raw units
        factor_depth: scalar, e.g. 1000.0
        crop_box: (x0,y0,x1,y1)
        return:
            depth_m_resized: (outH,outW) float32 meters
        """
        x0, y0, x1, y1 = crop_box
        outH, outW = out_hw

        depth_m = depth_raw.astype(np.float32) / float(factor_depth)
        depth_crop = depth_m[y0:y1, x0:x1].copy()
        depth_m_resized = cv2.resize(
            depth_crop, (outW, outH), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)
        return depth_m_resized

    def build_depth_prob_gt(self, depth_m_resized: np.ndarray):
        """
        depth_m_resized: (448,448) float32 meters (aligned with resized RGB)
        Returns:
        depth_prob_gt: (1, Nfeat, 256) float32, Nfeat=224*224
        depth_prob_w : (1, Nfeat) float32, valid_ratio per 2x2 patch
        """
        Hr, Wr = self.resize_shape              # (448,448)
        s = self.depth_prob_strides
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
        sensor_depth = np.array(Image.open(self.depthpath[index]))                            # (H,W)
        depth = sensor_depth.copy()
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
        # resized_idxs = self.get_resized_idxs(pix_flat, (H, W)) # resized (448*448) flatten index
        # img = self.img_transforms(color)                       # (3,448,448)

        crop_box = self._crop_box_from_mask(mask)
        x0, y0, x1, y1 = crop_box
        color_crop = color[y0:y1, x0:x1].copy()
        img = self.img_transforms(color_crop)
        resized_idxs = self.get_resized_idxs_from_flat_crop(pix_flat, (H, W), crop_box, out_hw=self.resize_shape)
        sensor_depth_m_resized = self.build_resized_depth_m_with_crop(
            sensor_depth, factor_depth, crop_box, out_hw=self.resize_shape
        )

        # ---- GT depth (virtual) -> meters -> resize to 448x448 ----
        # 注意：gt_depth 可能是 uint16 (mm / factor_depth)
        fd = float(self.gt_factor_depth) if (self.gt_factor_depth is not None) else float(factor_depth)
        gt_depth_m = gt_depth.astype(np.float32) / fd
        K_resized = self.resize_intrinsics_with_crop(intrinsic, crop_box, out_hw=self.resize_shape)
        
        Hr, Wr = self.resize_shape
        gt_depth_m = gt_depth_m[y0:y1, x0:x1]
        gt_depth_m_resized = cv2.resize(gt_depth_m, (Wr, Hr), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        depth_prob_gt, depth_prob_w = self.build_depth_prob_gt(gt_depth_m_resized)

        scene_idx = int(scene.split('_')[-1])
        anno_idx = int(self.frameid[index])

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
            # 'depth': sensor_depth_m_resized[None].astype(np.float32),   # (1,448,448)
            # 'sensor_depth_m': sensor_depth_m_resized.astype(np.float32),# (448,448), debug
            'scene_idx': np.int64(scene_idx),
            'anno_idx': np.int64(anno_idx),
            'dataset_idx': np.int64(index),
        }
        return ret_dict

    def get_data_label(self, index):
        # -----------------------------
        # 0) load raw data
        # -----------------------------
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0   # (H,W,3)
        sensor_depth = np.array(Image.open(self.depthpath[index]))                              # (H,W)
        depth = sensor_depth.copy()
        seg = np.array(Image.open(self.labelpath[index]))                                # (H,W)
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        gt_depth = np.array(Image.open(self.gtdepthpath[index]))                         # (H,W)

        graspness = np.load(self.graspnesspath[index])
        gt_graspness = np.load(self.gtgraspnesspath[index])

        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
            raise

        # optionally replace with GT depth / graspness
        if self.use_gt_depth:
            depth = gt_depth
            graspness = gt_graspness

        # normalize graspness to 1D masked-sequence style
        graspness = np.asarray(graspness, dtype=np.float32)
        if graspness.ndim == 2 and graspness.shape[1] == 1:
            graspness = graspness[:, 0]
        elif graspness.ndim == 2 and graspness.shape[0] == 1:
            graspness = graspness[0, :]
        elif graspness.ndim != 1:
            graspness = graspness.reshape(graspness.shape[0], -1)[:, 0]
        graspness = graspness.reshape(-1)

        camera = CameraInfo(
            1280.0, 720.0,
            intrinsic[0][0], intrinsic[1][1],
            intrinsic[0][2], intrinsic[1][2],
            factor_depth
        )
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # -----------------------------
        # 1) build mask
        # -----------------------------
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask

        H, W = depth.shape
        Hr, Wr = self.resize_shape  # expected (448,448)

        # -----------------------------
        # 2) crop box + crop-aware RGB / GT-depth / K
        # -----------------------------
        crop_box = self._crop_box_from_mask(mask)
        x0, y0, x1, y1 = crop_box

        # RGB crop -> img
        color_crop = color[y0:y1, x0:x1].copy()
        img = self.img_transforms(color_crop)

        sensor_depth_m_resized = self.build_resized_depth_m_with_crop(
            sensor_depth, factor_depth, crop_box, out_hw=self.resize_shape
        )
        
        # GT depth meters: crop -> resize
        fd = float(self.gt_factor_depth) if (self.gt_factor_depth is not None) else float(factor_depth)
        gt_depth_m = gt_depth.astype(np.float32) / fd
        gt_depth_crop = gt_depth_m[y0:y1, x0:x1].copy()
        gt_depth_m_resized = cv2.resize(
            gt_depth_crop, (Wr, Hr), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)

        # crop-aware intrinsics
        K_resized = self.resize_intrinsics_with_crop(intrinsic, crop_box, out_hw=self.resize_shape)

        # optional: keep depth_prob_gt for cls-based models; regression model can ignore it
        depth_prob_gt, depth_prob_w = self.build_depth_prob_gt(gt_depth_m_resized)

        # -----------------------------
        # 3) mask & sample masked sequences
        #    returns:
        #      H, W
        #      valid_flat   : (Nmasked,) full-image flat idx of masked points
        #      cloud_masked : (Nmasked,3)
        #      seg_masked   : (Nmasked,)
        # -----------------------------
        H, W, valid_flat, cloud_masked, seg_masked = self._mask_and_sample(depth, seg, cloud, mask)
        color_masked = color[mask]

        # sanity: graspness must align with masked sequence
        if graspness.shape[0] != valid_flat.shape[0]:
            raise ValueError(
                f"graspness length mismatch: graspness={graspness.shape}, "
                f"masked_points={valid_flat.shape}. "
                f"Expected graspness npy to align with mask sequence."
            )

        # -----------------------------
        # 4) Build token-level labels under crop-resize coordinates
        # -----------------------------
        # map ALL masked pixels into cropped+resized 448 grid
        resized_valid_all = self.get_resized_idxs_from_flat_crop(
            valid_flat, (H, W), crop_box, out_hw=self.resize_shape
        )
        resized_valid_all = np.asarray(resized_valid_all, dtype=np.int64).reshape(-1)

        # objectness for masked pixels (binary)
        obj_masked = seg_masked.copy()
        obj_masked[obj_masked > 1] = 1
        obj_masked = np.asarray(obj_masked, dtype=np.int64).reshape(-1)

        # graspness for masked pixels (already aligned 1D)
        grasp_masked = graspness.astype(np.float32).reshape(-1)

        # sanity check
        Nmasked = resized_valid_all.shape[0]
        if not (obj_masked.shape[0] == Nmasked and grasp_masked.shape[0] == Nmasked):
            raise ValueError(
                f"Masked arrays length mismatch: resized_valid_all={Nmasked}, "
                f"obj_masked={obj_masked.shape}, grasp_masked={grasp_masked.shape}. "
                f"Check that graspness npy aligns with cloud_masked/seg_masked order."
            )

        # scatter to 448x448 crop-resized plane
        obj_flat = np.zeros((Hr * Wr,), dtype=np.int64)
        gsum = np.zeros((Hr * Wr,), dtype=np.float32)
        gcnt = np.zeros((Hr * Wr,), dtype=np.float32)

        # collisions: objectness uses max; graspness uses mean
        np.maximum.at(obj_flat, resized_valid_all, obj_masked)
        np.add.at(gsum, resized_valid_all, grasp_masked)
        np.add.at(gcnt, resized_valid_all, 1.0)

        grasp_flat = gsum / np.maximum(gcnt, 1.0)
        valid_flat_resized = (gcnt > 0).astype(np.float32)

        obj_map_448 = obj_flat.reshape(Hr, Wr)
        grasp_map_448 = grasp_flat.reshape(Hr, Wr)
        valid_map_448 = valid_flat_resized.reshape(Hr, Wr)

        # aggregate 2x2 -> token (224x224)
        s_tok = 2
        Ht, Wt = Hr // s_tok, Wr // s_tok

        obj_blk = obj_map_448.reshape(Ht, s_tok, Wt, s_tok)
        grasp_blk = grasp_map_448.reshape(Ht, s_tok, Wt, s_tok)
        valid_blk = valid_map_448.reshape(Ht, s_tok, Wt, s_tok)

        valid_cnt = valid_blk.sum(axis=(1, 3)).astype(np.float32)   # (Ht,Wt)
        obj_tok = obj_blk.max(axis=(1, 3)).astype(np.int64)

        grasp_sum = (grasp_blk * valid_blk).sum(axis=(1, 3)).astype(np.float32)
        grasp_tok = grasp_sum / np.maximum(valid_cnt, 1.0)
        grasp_tok[valid_cnt < 1.0] = 0.0

        # invalid tokens: ignore in CE
        obj_tok[valid_cnt < 1.0] = -1

        # objectness_label_tok = obj_tok.reshape(-1).astype(np.int64)      # (Ntok,)
        # graspness_label_tok  = grasp_tok.reshape(-1).astype(np.float32)  # (Ntok,)
        # token_valid_mask     = (valid_cnt.reshape(-1) >= 1.0)            # (Ntok,)

        objectness_label_tok = obj_map_448.reshape(-1).astype(np.int64)
        objectness_label_tok[valid_map_448.reshape(-1) < 1.0] = -1
        graspness_label_tok = grasp_map_448.reshape(-1).astype(np.float32)
        token_valid_mask = (valid_map_448.reshape(-1) >= 1.0)

        # -----------------------------
        # 5) Sample point cloud (point-level)
        # -----------------------------
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        graspness_sampled = grasp_masked[idxs].astype(np.float32)

        objectness_label = seg_sampled.copy()
        segmentation_label = objectness_label.copy()
        objectness_label[objectness_label > 1] = 1

        # sampled pixels -> crop-resized img_idxs
        pix_flat = valid_flat[idxs]
        resized_idxs = self.get_resized_idxs_from_flat_crop(
            pix_flat, (H, W), crop_box, out_hw=self.resize_shape
        )

        # -----------------------------
        # 6) Load economic grasp labels (object-level lists)
        # -----------------------------
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


        scene_idx = int(scene.split('_')[-1])
        anno_idx = int(self.frameid[index])
        # -----------------------------
        # 7) return dict
        # -----------------------------
        ret_dict = {
            # point-level
            'point_clouds': cloud_sampled.astype(np.float32),
            'cloud_colors': color_sampled.astype(np.float32),
            'coordinates_for_voxel': (cloud_sampled.astype(np.float32) / self.voxel_size),

            'img': img,
            'img_idxs': resized_idxs.astype(np.int64),  # sampled pixels in cropped+resized 448 grid

            'graspness_label': graspness_sampled.astype(np.float32),
            'objectness_label': objectness_label.astype(np.int64),
            'segmentation_label': segmentation_label.astype(np.int64),

            # token-level
            'objectness_label_tok': objectness_label_tok,
            'graspness_label_tok': graspness_label_tok,
            'token_valid_mask': token_valid_mask.astype(np.bool_),

            # economic grasp labels
            'object_poses_list': object_poses_list,
            'grasp_points_list': grasp_points_list,
            'grasp_rotations_list': grasp_rotations_list,
            'grasp_depth_list': grasp_depth_list,
            'grasp_widths_list': grasp_widths_list,
            'grasp_scores_list': grasp_scores_list,
            'view_graspness_list': view_graspness_list,
            'top_view_index_list': top_view_index_list,

            # debug / bookkeeping
            'sampled_masked_idxs': idxs.astype(np.int64),
            'pix_flat': pix_flat.astype(np.int64),
            'crop_box': np.asarray(crop_box, dtype=np.int64),

            # camera / depth supervision
            'K': K_resized.astype(np.float32),
            'gt_depth_m': gt_depth_m_resized.astype(np.float32),
            'depth_prob_gt': depth_prob_gt.astype(np.float32),
            'depth_prob_weight': depth_prob_w.astype(np.float32),

            # 'depth': sensor_depth_m_resized[None].astype(np.float32),
            # 'sensor_depth_m': sensor_depth_m_resized.astype(np.float32),

            'scene_idx': np.int64(scene_idx),
            'anno_idx': np.int64(anno_idx),
            'dataset_idx': np.int64(index),
        }
        return ret_dict


class GraspNetTransDataset(GraspNetMultiDataset):
    def __init__(
        self,
        graspnet_root,
        rgb_root,
        camera='realsense',
        split='train',
        num_points=20000,
        voxel_size=0.005,
        remove_outlier=False,
        remove_invisible=True,
        augment=False,
        load_label=True,
        use_gt_depth=True,
        min_depth=0.2,
        max_depth=1.0,
        bin_num=256,
        depth_strides=2,
    ):
        self.root = graspnet_root
        self.rgb_root = rgb_root
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

        # split -> scene integer ids (same as GraspNet)
        if split == 'train':
            scene_ids = list(range(100))
        elif split == 'test':
            scene_ids = list(range(100, 190))
        elif split == 'test_seen':
            scene_ids = list(range(100, 130))
        elif split == 'test_similar':
            scene_ids = list(range(130, 160))
        elif split == 'test_novel':
            scene_ids = list(range(160, 190))
        else:
            raise ValueError(f"Unknown split={split}")

        # GraspNet uses: scene_0000 (4 digits)
        self.sceneIds = [f"scene_{sid:04d}" for sid in scene_ids]
        # Trans RGB uses: 00000 (5 digits)
        self.rgbSceneIds = [f"{sid:05d}" for sid in scene_ids]

        # ---- multi-modal same as yours ----
        self.resize_shape = (448, 448)
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.depth_prob_min = min_depth
        self.depth_prob_max = max_depth
        self.depth_prob_bins = bin_num
        self.depth_prob_strides = depth_strides
        self.depth_prob_valid_threshold = -1
        self.gt_factor_depth = None

        # ---- paths ----
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

        for g_scene, r_scene in tqdm(
            list(zip(self.sceneIds, self.rgbSceneIds)),
            desc='Loading Trans scene data...'
        ):
            for img_num in range(256):
                fid = str(img_num).zfill(4)

                # RGB: dataset_root/scenes/xxxxx/xxxx_color.png   (xxxxx is 5 digits)
                self.colorpath.append(os.path.join(
                    self.rgb_root, 'scenes', r_scene, f'{fid}_color.png'
                ))

                # virtual GT depth/label: graspnet_root/virtual_scene/scene_xxxx/realsense/xxxx_depth.png
                vbase = os.path.join(self.root, 'virtual_scenes', g_scene, camera)
                vdepth = os.path.join(vbase, f'{fid}_depth.png')

                # 这里 depth 输入也用 virtual depth（与你的描述一致）
                self.depthpath.append(vdepth)
                self.gtdepthpath.append(vdepth)

                # meta: original graspnet structure
                self.metapath.append(os.path.join(
                    self.root, 'scenes', g_scene, camera, 'meta', f'{fid}.mat'
                ))
                self.labelpath.append(os.path.join(
                    self.root, 'scenes', g_scene, camera, 'label', f'{fid}.png'
                ))
                
                # graspness: graspnet_root/virtual_graspness/scene_xxxx/realsense/xxxx.npy
                gpath = os.path.join(self.root, 'virtual_graspness', g_scene, camera, f'{fid}.npy')
                self.graspnesspath.append(gpath)
                self.gtgraspnesspath.append(gpath)

                # IMPORTANT: keep scenename as GraspNet scene_xxxx (used by parent methods to find camera_poses, labels, etc.)
                self.scenename.append(g_scene)
                self.frameid.append(img_num)

            if self.load_label:
                self.grasp_labels[g_scene] = os.path.join(
                    self.root, 'economic_grasp_label_300views', f'{g_scene}_labels.npz'
                )


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