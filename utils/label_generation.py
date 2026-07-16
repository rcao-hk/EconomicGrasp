import os
import sys
import time
import pdb

import numpy as np
import torch
import os
import sys
import torch

# from libs.knn.knn_modules import knn
from pytorch3d.ops.knn import knn_points
from utils.loss_utils import (batch_viewpoint_params_to_matrix, transform_point_cloud,
                              generate_grasp_views, compute_pointwise_dists)
from utils.arguments import cfgs


def process_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['xyz_graspable']  # [B (batch size), 1024 (scene points after sample), 3]
    pred_top_view_inds = end_points['grasp_top_view_inds']  # [B (batch size), 1024 (scene points after sample)]
    batch_size, num_samples, _ = seed_xyzs.size()

    valid_points_count = 0
    valid_views_count = 0

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_view_graspness = []
    batch_grasp_rotations = []
    batch_grasp_depth = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_grasp_collisions = []
    batch_valid_mask = []
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # [1024 (scene points after sample), 3]
        pred_top_view = pred_top_view_inds[i]  # [1024 (scene points after sample)]
        poses = end_points['object_poses_list'][i]  # a list with length of object amount, each has size [3, 4]

        # get merged grasp points for label computation
        # transform the view from object coordinate system to scene coordinate system
        grasp_points_merged = []
        grasp_views_rot_merged = []
        grasp_rotations_merged = []
        grasp_depth_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        grasp_collisions_merged = []
        view_graspness_merged = []
        top_view_index_merged = []
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][i][obj_idx]  # [objects points, 3]
            grasp_rotations = end_points['grasp_rotations_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_depth = end_points['grasp_depth_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_scores = end_points['grasp_scores_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_widths = end_points['grasp_widths_list'][i][obj_idx]  # [objects points, num_of_view]
            view_graspness = end_points['view_graspness_list'][i][obj_idx]  # [objects points, 300]
            top_view_index = end_points['top_view_index_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_collisions = end_points['grasp_collision_list'][i][obj_idx].float()
            num_grasp_points = grasp_points.size(0)

            # generate and transform template grasp views
            grasp_views = generate_grasp_views(cfgs.num_view).to(pose.device)  # [300 (views), 3 (coordinate)]
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')
            # [300 (views), 3 (coordinate)], after translation to scene coordinate system

            # generate and transform template grasp view rotation
            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)
            # [300 (views), 3, 3 (the rotation matrix)]

            # assign views after transform (the view will not exactly match)
            # grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            # grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            # view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1  # [300]
            
            grasp_views_ = grasp_views.unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.unsqueeze(0)
            _, view_inds, _ = knn_points(grasp_views_, grasp_views_trans_, K=1)
            view_inds = view_inds.squeeze(-1).squeeze(0)
            
            view_graspness_trans = torch.index_select(view_graspness, 1, view_inds)  # [object points, 300]
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)
            # [object points, 300, 3, 3]

            # -1 means that when we transform the top 60 views into the scene coordinate,
            # some views will have no matching
            # It means that two views in the object coordinate match to one view in the scene coordinate
            top_view_index_trans = (-1 * torch.ones((num_grasp_points, grasp_rotations.shape[1]), dtype=torch.long)
                                    .to(seed_xyz.device))
            tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
            top_view_index_trans[tpid, tvip] = tids  # [objects points, num_of_view]

            # add to list
            grasp_points_merged.append(grasp_points_trans)
            view_graspness_merged.append(view_graspness_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_rotations_merged.append(grasp_rotations)
            grasp_depth_merged.append(grasp_depth)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)
            grasp_collisions_merged.append(grasp_collisions)
            grasp_views_rot_merged.append(grasp_views_rot_trans)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # [all object points, 3]
        view_graspness_merged = torch.cat(view_graspness_merged, dim=0)  # [all object points, 300]
        top_view_index_merged = torch.cat(top_view_index_merged, dim=0)  # [all object points, num_of_view]
        grasp_rotations_merged = torch.cat(grasp_rotations_merged, dim=0)  # [all object points, num_of_view]
        grasp_depth_merged = torch.cat(grasp_depth_merged, dim=0)  # [all object points, num_of_view]
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # [all object points, num_of_view]
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # [all object points, num_of_view]
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # [all object points, 300, 3, 3]
        grasp_collisions_merged = torch.cat(grasp_collisions_merged, dim=0)
        
        # compute nearest neighbors
        # seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)
        # grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)
        # nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1

        seed_xyz_ = seed_xyz.unsqueeze(0)  # (1, Ns, 3)
        grasp_points_merged_ = grasp_points_merged.unsqueeze(0)  # (1, Np', 3)
        _, nn_inds, _ = knn_points(seed_xyz_, grasp_points_merged_, K=1) # (Ns)
        nn_inds = nn_inds.squeeze(-1).squeeze(0)
        
        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
        # [1024 (scene points after sample), 3]
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)
        # [1024 (scene points after sample), 300, 3, 3]
        view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)
        # [1024 (scene points after sample), 300]
        top_view_index_merged = torch.index_select(top_view_index_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_rotations_merged = torch.index_select(grasp_rotations_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_depth_merged = torch.index_select(grasp_depth_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_collisions_merged = torch.index_select(grasp_collisions_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        
        # select top view's rot, score and width
        # we only assign labels when the pred view is in the pre-defined 60 top view, others are zero
        pred_top_view_ = pred_top_view.view(num_samples, 1, 1, 1).expand(-1, -1, 3, 3)
        # [1024 (points after sample), 1, 3, 3]
        top_grasp_views_rot = torch.gather(grasp_views_rot_merged, 1, pred_top_view_).squeeze(1)
        # [1024 (points after sample), 3, 3]
        pid, vid = torch.where(pred_top_view.unsqueeze(-1) == top_view_index_merged)
        # both pid and vid are [true numbers], where(condition) equals to nonzero(condition)
        top_grasp_rotations = 12 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_depth = 4 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_scores = torch.zeros(num_samples, dtype=torch.float32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_widths = 0.1 * torch.ones(num_samples, dtype=torch.float32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_collisions = torch.zeros(num_samples, dtype=torch.float32).to(seed_xyz.device)
        # [1024 (points after sample)]
        
        top_grasp_rotations[pid] = torch.gather(grasp_rotations_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_depth[pid] = torch.gather(grasp_depth_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_scores[pid] = torch.gather(grasp_scores_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_widths[pid] = torch.gather(grasp_widths_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_collisions[pid] = torch.gather(grasp_collisions_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        
        # only compute loss in the points with correct matching (so compute the mask first)
        dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)
        valid_point_mask = dist < 0.005
        valid_view_mask = torch.zeros(num_samples, dtype=torch.bool).to(seed_xyz.device)
        valid_view_mask[pid] = True
        valid_points_count = valid_points_count + torch.sum(valid_point_mask)
        valid_views_count = valid_views_count + torch.sum(valid_view_mask)
        valid_mask = valid_point_mask & valid_view_mask

        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(top_grasp_views_rot)
        batch_view_graspness.append(view_graspness_merged)
        batch_grasp_rotations.append(top_grasp_rotations)
        batch_grasp_depth.append(top_grasp_depth)
        batch_grasp_scores.append(top_grasp_scores)
        batch_grasp_widths.append(top_grasp_widths)
        batch_grasp_collisions.append(top_grasp_collisions)
        batch_valid_mask.append(valid_mask)
        
        # print("[dbg] seed_xyz z:", seed_xyz[:,2].min().item(), seed_xyz[:,2].median().item(), seed_xyz[:,2].max().item())
        # print("[dbg] grasp_xyz z:", grasp_points_merged[:,2].min().item(), grasp_points_merged[:,2].median().item(), grasp_points_merged[:,2].max().item())

        # print("[dbg] seed mean:", seed_xyz.mean(0))
        # print("[dbg] grasp mean:", grasp_points_merged.mean(0))
        # print("[dbg] mean delta:", (seed_xyz.mean(0) - grasp_points_merged.mean(0)))

    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    # [B (batch size), 1024 (scene points after sample), 3]
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
    # [B (batch size), 1024 (scene points after sample), 3, 3]
    batch_view_graspness = torch.stack(batch_view_graspness, 0)
    # [B (batch size), 1024 (scene points after sample), 300]
    batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_depth = torch.stack(batch_grasp_depth, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_collisions = torch.stack(batch_grasp_collisions, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_valid_mask = torch.stack(batch_valid_mask, 0)
    # [B (batch size), 1024 (scene points after sample)]

    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_rotations'] = batch_grasp_rotations
    end_points['batch_grasp_depth'] = batch_grasp_depth
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_view_graspness'] = batch_view_graspness
    end_points['batch_grasp_collision'] = batch_grasp_collisions
    end_points['batch_valid_mask'] = batch_valid_mask
    end_points['C: Valid Points'] = valid_points_count / batch_size
    return batch_grasp_views_rot, end_points



def _depth_cls_to_metric(depth_cls, depth_start=0.01, depth_interval=0.01):
    """
    depth_cls: 0,1,2,3 -> 0.01,0.02,0.03,0.04
    """
    return depth_start + depth_cls.float() * depth_interval


def _metric_depth_to_cls(
    depth_metric,
    num_depth,
    depth_start=0.01,
    depth_interval=0.01,
):
    """
    Convert compensated metric depth to nearest depth class.

    Class definition is fixed:
      class 0 -> 0.01
      class 1 -> 0.02
      class 2 -> 0.03
      class 3 -> 0.04

    A continuous depth is valid if it lies within half-bin range:
      [0.01 - 0.005, 0.04 + 0.005]
    i.e. [0.005, 0.045] when num_depth=4 and interval=0.01.
    """
    depth_start = float(depth_start)
    depth_interval = float(depth_interval)

    depth_cls_float = (depth_metric - depth_start) / depth_interval
    depth_cls = torch.floor(depth_cls_float + 0.5).long()

    valid = (
        (depth_cls_float >= -0.5)
        & (depth_cls_float <= (float(num_depth) - 0.5))
        & torch.isfinite(depth_metric)
    )

    depth_cls = depth_cls.clamp(0, num_depth - 1)
    return depth_cls, valid


def process_grasp_labels_depth_cls_compensated(
    end_points,
    point_match_thresh=0.005,
    tolerated_depth=0.03,
    depth_start=0.01,
    depth_interval=0.01,
    approach_axis_col=0,
    approach_axis_sign=1.0,
    depth_adjust_sign=1.0,
):
    """
    Depth-class compensation version.

    Difference from original process_grasp_labels:
      - Keep original nearest CAD/grasp point matching.
      - Add approach-line compensated matching.
      - Absorb approach-direction offset into depth class label.
      - Still output batch_grasp_depth as a classification target.

    Original valid:
      ||P - Q|| < point_match_thresh

    Compensated valid:
      e = Q - P
      a = predicted approach direction
      delta = dot(e, a)
      lateral = ||e - delta * a||

      valid_comp = lateral < point_match_thresh and |delta| < tolerated_depth

    Depth label:
      D_base = original class depth
      D_comp = D_base + depth_adjust_sign * delta
      depth_cls_comp = nearest depth anchor class

    Required end_points:
      - xyz_graspable
      - grasp_top_view_inds
      - object_poses_list
      - grasp_points_list
      - grasp_rotations_list
      - grasp_depth_list
      - grasp_scores_list
      - grasp_widths_list
      - view_graspness_list
      - top_view_index_list
    """
    seed_xyzs = end_points["xyz_graspable"]                  # (B,M,3)
    pred_top_view_inds = end_points["grasp_top_view_inds"]   # (B,M)
    batch_size, num_samples, _ = seed_xyzs.size()

    valid_points_count = seed_xyzs.new_tensor(0.0)
    valid_origin_count = seed_xyzs.new_tensor(0.0)
    valid_comp_count = seed_xyzs.new_tensor(0.0)
    valid_comp_only_count = seed_xyzs.new_tensor(0.0)
    valid_geom_count = seed_xyzs.new_tensor(0.0)
    bad_label_count = seed_xyzs.new_tensor(0.0)
    invalid_depth_cls_count = seed_xyzs.new_tensor(0.0)
    valid_views_count = seed_xyzs.new_tensor(0.0)

    batch_grasp_points = []
    batch_grasp_points_comp = []
    batch_grasp_views_rot = []
    batch_view_graspness = []
    batch_grasp_rotations = []
    batch_grasp_depth = []
    batch_grasp_depth_original = []
    batch_grasp_depth_base = []
    batch_grasp_depth_comp_metric = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_valid_mask = []

    batch_match_dist = []
    batch_lateral_dist = []
    batch_depth_delta = []
    batch_valid_point_origin = []
    batch_valid_point_comp = []
    batch_valid_point_comp_only = []
    batch_valid_geom_mask = []
    batch_valid_label_mask = []
    batch_valid_depth_cls = []

    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]                # (M,3)
        pred_top_view = pred_top_view_inds[i]  # (M,)
        poses = end_points["object_poses_list"][i]

        grasp_points_merged = []
        grasp_views_rot_merged = []
        grasp_rotations_merged = []
        grasp_depth_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        view_graspness_merged = []
        top_view_index_merged = []

        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points["grasp_points_list"][i][obj_idx]
            grasp_rotations = end_points["grasp_rotations_list"][i][obj_idx]
            grasp_depth = end_points["grasp_depth_list"][i][obj_idx]
            grasp_scores = end_points["grasp_scores_list"][i][obj_idx]
            grasp_widths = end_points["grasp_widths_list"][i][obj_idx]
            view_graspness = end_points["view_graspness_list"][i][obj_idx]
            top_view_index = end_points["top_view_index_list"][i][obj_idx]
            num_grasp_points = grasp_points.size(0)

            grasp_views = generate_grasp_views(cfgs.num_view).to(pose.device)

            grasp_points_trans = transform_point_cloud(grasp_points, pose, "3x4")
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], "3x3")

            angles = torch.zeros(
                grasp_views.size(0),
                dtype=grasp_views.dtype,
                device=grasp_views.device,
            )
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)

            grasp_views_ = grasp_views.unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.unsqueeze(0)
            _, view_inds, _ = knn_points(grasp_views_, grasp_views_trans_, K=1)
            view_inds = view_inds.squeeze(-1).squeeze(0)

            view_graspness_trans = torch.index_select(view_graspness, 1, view_inds)

            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(
                num_grasp_points, -1, -1, -1
            )

            top_view_index_trans = -1 * torch.ones(
                (num_grasp_points, grasp_rotations.shape[1]),
                dtype=torch.long,
                device=seed_xyz.device,
            )
            tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
            top_view_index_trans[tpid, tvip] = tids

            grasp_points_merged.append(grasp_points_trans)
            view_graspness_merged.append(view_graspness_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_rotations_merged.append(grasp_rotations)
            grasp_depth_merged.append(grasp_depth)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)
            grasp_views_rot_merged.append(grasp_views_rot_trans)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)
        view_graspness_merged = torch.cat(view_graspness_merged, dim=0)
        top_view_index_merged = torch.cat(top_view_index_merged, dim=0)
        grasp_rotations_merged = torch.cat(grasp_rotations_merged, dim=0)
        grasp_depth_merged = torch.cat(grasp_depth_merged, dim=0)
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)

        # ------------------------------------------------------------
        # 1) nearest CAD-derived grasp point
        # ------------------------------------------------------------
        seed_xyz_ = seed_xyz.unsqueeze(0)                         # (1,M,3)
        grasp_points_merged_ = grasp_points_merged.unsqueeze(0)   # (1,Np,3)
        _, nn_inds, _ = knn_points(seed_xyz_, grasp_points_merged_, K=1)
        nn_inds = nn_inds.squeeze(-1).squeeze(0)

        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)
        view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)
        top_view_index_merged = torch.index_select(top_view_index_merged, 0, nn_inds)
        grasp_rotations_merged = torch.index_select(grasp_rotations_merged, 0, nn_inds)
        grasp_depth_merged = torch.index_select(grasp_depth_merged, 0, nn_inds)
        grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)

        # ------------------------------------------------------------
        # 2) labels under predicted top view
        # ------------------------------------------------------------
        pred_top_view_ = pred_top_view.view(num_samples, 1, 1, 1).expand(-1, -1, 3, 3)
        top_grasp_views_rot = torch.gather(
            grasp_views_rot_merged, 1, pred_top_view_
        ).squeeze(1)

        pid, vid = torch.where(pred_top_view.unsqueeze(-1) == top_view_index_merged)

        # Sentinel labels.
        top_grasp_rotations = cfgs.num_angle * torch.ones(
            num_samples, dtype=torch.long, device=seed_xyz.device
        )
        top_grasp_depth_original = cfgs.num_depth * torch.ones(
            num_samples, dtype=torch.long, device=seed_xyz.device
        )
        top_grasp_scores = torch.zeros(
            num_samples, dtype=torch.float32, device=seed_xyz.device
        )
        top_grasp_widths = 0.1 * torch.ones(
            num_samples, dtype=torch.float32, device=seed_xyz.device
        )

        if pid.numel() > 0:
            top_grasp_rotations[pid] = torch.gather(
                grasp_rotations_merged[pid], 1, vid.view(-1, 1)
            ).squeeze(1).long()

            top_grasp_depth_original[pid] = torch.gather(
                grasp_depth_merged[pid], 1, vid.view(-1, 1)
            ).squeeze(1).long()

            top_grasp_scores[pid] = torch.gather(
                grasp_scores_merged[pid], 1, vid.view(-1, 1)
            ).squeeze(1)

            top_grasp_widths[pid] = torch.gather(
                grasp_widths_merged[pid], 1, vid.view(-1, 1)
            ).squeeze(1)

        valid_view_mask = torch.zeros(num_samples, dtype=torch.bool, device=seed_xyz.device)
        valid_view_mask[pid] = True

        # ------------------------------------------------------------
        # 3) original hard point matching
        # ------------------------------------------------------------
        match_dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)
        valid_point_origin = match_dist < point_match_thresh

        # ------------------------------------------------------------
        # 4) approach-depth compensated matching
        # ------------------------------------------------------------
        approach_axis = top_grasp_views_rot[:, :, approach_axis_col].contiguous()
        approach_axis = approach_axis * float(approach_axis_sign)
        approach_axis = approach_axis / approach_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        e = grasp_points_merged - seed_xyz
        delta = (e * approach_axis).sum(dim=-1)
        e_perp = e - delta.unsqueeze(-1) * approach_axis
        lateral_dist = e_perp.norm(dim=-1)

        valid_point_comp = (
            (lateral_dist < point_match_thresh)
            & (delta.abs() < tolerated_depth)
        )
        valid_point_comp_only = valid_point_comp & (~valid_point_origin)

        valid_point_mask = valid_point_origin | valid_point_comp

        # ------------------------------------------------------------
        # 5) absorb offset into depth class label
        # ------------------------------------------------------------
        original_depth_valid = (
            (top_grasp_depth_original >= 0)
            & (top_grasp_depth_original < cfgs.num_depth)
        )

        safe_depth_cls = top_grasp_depth_original.clamp(0, cfgs.num_depth - 1)
        depth_base = _depth_cls_to_metric(
            safe_depth_cls,
            depth_start=depth_start,
            depth_interval=depth_interval,
        )

        depth_comp_metric = depth_base + float(depth_adjust_sign) * delta

        depth_comp_cls, valid_depth_cls = _metric_depth_to_cls(
            depth_comp_metric,
            num_depth=cfgs.num_depth,
            depth_start=depth_start,
            depth_interval=depth_interval
        )

        valid_depth_cls = valid_depth_cls & original_depth_valid & torch.isfinite(depth_comp_metric)

        # Sentinel by default. Only valid entries get adjusted class.
        top_grasp_depth_comp = cfgs.num_depth * torch.ones(
            num_samples, dtype=torch.long, device=seed_xyz.device
        )
        top_grasp_depth_comp[valid_depth_cls] = depth_comp_cls[valid_depth_cls]

        # Geometry validity.
        valid_geom_mask = valid_point_mask & valid_view_mask & valid_depth_cls

        # Label validity for CE/regression losses.
        valid_label_mask = (
            (top_grasp_rotations >= 0)
            & (top_grasp_rotations < cfgs.num_angle)
            & (top_grasp_depth_comp >= 0)
            & (top_grasp_depth_comp < cfgs.num_depth)
            & torch.isfinite(top_grasp_scores)
            & torch.isfinite(top_grasp_widths)
        )

        valid_mask = valid_geom_mask & valid_label_mask

        compensated_points = seed_xyz + delta.unsqueeze(-1) * approach_axis

        valid_origin_final = valid_point_origin & valid_view_mask & valid_depth_cls & valid_label_mask
        valid_comp_final = valid_point_comp & valid_view_mask & valid_depth_cls & valid_label_mask
        valid_comp_only_final = valid_point_comp_only & valid_view_mask & valid_depth_cls & valid_label_mask

        valid_origin_count = valid_origin_count + valid_origin_final.float().sum()
        valid_comp_count = valid_comp_count + valid_comp_final.float().sum()
        valid_comp_only_count = valid_comp_only_count + valid_comp_only_final.float().sum()
        valid_points_count = valid_points_count + valid_mask.float().sum()
        valid_geom_count = valid_geom_count + valid_geom_mask.float().sum()
        bad_label_count = bad_label_count + (valid_geom_mask & (~valid_label_mask)).float().sum()
        invalid_depth_cls_count = invalid_depth_cls_count + (valid_point_mask & valid_view_mask & (~valid_depth_cls)).float().sum()
        valid_views_count = valid_views_count + valid_view_mask.float().sum()

        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_points_comp.append(compensated_points)
        batch_grasp_views_rot.append(top_grasp_views_rot)
        batch_view_graspness.append(view_graspness_merged)
        batch_grasp_rotations.append(top_grasp_rotations)
        batch_grasp_depth.append(top_grasp_depth_comp)
        batch_grasp_depth_original.append(top_grasp_depth_original)
        batch_grasp_depth_base.append(depth_base)
        batch_grasp_depth_comp_metric.append(depth_comp_metric)
        batch_grasp_scores.append(top_grasp_scores)
        batch_grasp_widths.append(top_grasp_widths)
        batch_valid_mask.append(valid_mask)

        batch_match_dist.append(match_dist)
        batch_lateral_dist.append(lateral_dist)
        batch_depth_delta.append(delta)
        batch_valid_point_origin.append(valid_point_origin)
        batch_valid_point_comp.append(valid_point_comp)
        batch_valid_point_comp_only.append(valid_point_comp_only)
        batch_valid_geom_mask.append(valid_geom_mask)
        batch_valid_label_mask.append(valid_label_mask)
        batch_valid_depth_cls.append(valid_depth_cls)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    batch_grasp_points_comp = torch.stack(batch_grasp_points_comp, 0)
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
    batch_view_graspness = torch.stack(batch_view_graspness, 0)
    batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
    batch_grasp_depth = torch.stack(batch_grasp_depth, 0)
    batch_grasp_depth_original = torch.stack(batch_grasp_depth_original, 0)
    batch_grasp_depth_base = torch.stack(batch_grasp_depth_base, 0)
    batch_grasp_depth_comp_metric = torch.stack(batch_grasp_depth_comp_metric, 0)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
    batch_valid_mask = torch.stack(batch_valid_mask, 0)

    end_points["batch_grasp_point"] = batch_grasp_points
    end_points["batch_grasp_point_compensated"] = batch_grasp_points_comp
    end_points["batch_grasp_rotations"] = batch_grasp_rotations

    # Main depth classification target after compensation.
    end_points["batch_grasp_depth"] = batch_grasp_depth

    # Debug / ablation.
    end_points["batch_grasp_depth_original"] = batch_grasp_depth_original
    end_points["batch_grasp_depth_base"] = batch_grasp_depth_base
    end_points["batch_grasp_depth_comp_metric"] = batch_grasp_depth_comp_metric

    end_points["batch_grasp_score"] = batch_grasp_scores
    end_points["batch_grasp_width"] = batch_grasp_widths
    end_points["batch_grasp_view_graspness"] = batch_view_graspness
    end_points["batch_valid_mask"] = batch_valid_mask

    # Debug masks.
    end_points["batch_match_dist"] = torch.stack(batch_match_dist, 0)
    end_points["batch_lateral_dist"] = torch.stack(batch_lateral_dist, 0)
    end_points["batch_depth_delta"] = torch.stack(batch_depth_delta, 0)
    end_points["batch_valid_point_origin"] = torch.stack(batch_valid_point_origin, 0)
    end_points["batch_valid_point_comp"] = torch.stack(batch_valid_point_comp, 0)
    end_points["batch_valid_point_comp_only"] = torch.stack(batch_valid_point_comp_only, 0)
    end_points["batch_valid_geom_mask"] = torch.stack(batch_valid_geom_mask, 0)
    end_points["batch_valid_label_mask"] = torch.stack(batch_valid_label_mask, 0)
    end_points["batch_valid_depth_cls"] = torch.stack(batch_valid_depth_cls, 0)

    end_points["C: Valid Points"] = valid_points_count / float(batch_size)
    end_points["C: Valid Points Origin"] = valid_origin_count / float(batch_size)
    end_points["C: Valid Points Comp"] = valid_comp_count / float(batch_size)
    end_points["C: Valid Points CompOnly"] = valid_comp_only_count / float(batch_size)
    end_points["C: Valid Geom Points"] = valid_geom_count / float(batch_size)
    end_points["C: Bad Label Points"] = bad_label_count / float(batch_size)
    end_points["C: Invalid DepthCls Points"] = invalid_depth_cls_count / float(batch_size)
    end_points["C: Valid Views"] = valid_views_count / float(batch_size)

    with torch.no_grad():
        valid = batch_valid_mask
        zero = seed_xyzs.new_tensor(0.0)

        if valid.any():
            end_points["C: MatchDist mean(valid)"] = end_points["batch_match_dist"][valid].mean()
            end_points["C: LateralDist mean(valid)"] = end_points["batch_lateral_dist"][valid].mean()
            end_points["C: Delta mean(valid)"] = end_points["batch_depth_delta"][valid].mean()
            end_points["C: Delta abs mean(valid)"] = end_points["batch_depth_delta"][valid].abs().mean()
            end_points["C: DepthBase mean(valid)"] = batch_grasp_depth_base[valid].mean()
            end_points["C: DepthCompMetric mean(valid)"] = batch_grasp_depth_comp_metric[valid].mean()
            end_points["C: DepthCompMetric min(valid)"] = batch_grasp_depth_comp_metric[valid].min()
            end_points["C: DepthCompMetric max(valid)"] = batch_grasp_depth_comp_metric[valid].max()

            changed = (batch_grasp_depth != batch_grasp_depth_original) & valid
            end_points["C: DepthCls Changed"] = changed.float().sum() / (valid.float().sum() + 1e-6)
        else:
            end_points["C: MatchDist mean(valid)"] = zero
            end_points["C: LateralDist mean(valid)"] = zero
            end_points["C: Delta mean(valid)"] = zero
            end_points["C: Delta abs mean(valid)"] = zero
            end_points["C: DepthBase mean(valid)"] = zero
            end_points["C: DepthCompMetric mean(valid)"] = zero
            end_points["C: DepthCompMetric min(valid)"] = zero
            end_points["C: DepthCompMetric max(valid)"] = zero
            end_points["C: DepthCls Changed"] = zero

    return batch_grasp_views_rot, end_points


def process_grasp_labels_c5_1(end_points, valid_dist_thresh: float = 0.005):
    """
    Top-K latent matching version for c5.1.
    Stores extra tensors for visualization:
      - batch_best_hyp_k
      - batch_best_hyp_dist
      - batch_grasp_point (matched GT point)
    """
    seed_xyz_hyp = end_points.get('xyz_graspable_hyp', None)
    if seed_xyz_hyp is None:
        seed_xyz_mu = end_points['xyz_graspable']
        seed_xyz_hyp = seed_xyz_mu.unsqueeze(2)
    seed_xyz_mu = end_points['xyz_graspable']
    pred_top_view_inds = end_points['grasp_top_view_inds']

    batch_size, num_seed, K_hyp, _ = seed_xyz_hyp.size()
    valid_points_count = 0
    valid_views_count = 0

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_view_graspness = []
    batch_grasp_rotations = []
    batch_grasp_depth = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_valid_mask = []
    batch_best_k = []
    batch_best_dist = []

    for i in range(batch_size):
        pred_top_view = pred_top_view_inds[i]
        poses = end_points['object_poses_list'][i]

        grasp_points_merged = []
        grasp_views_rot_merged = []
        grasp_rotations_merged = []
        grasp_depth_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        view_graspness_merged = []
        top_view_index_merged = []

        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][i][obj_idx]
            grasp_rotations = end_points['grasp_rotations_list'][i][obj_idx]
            grasp_depth = end_points['grasp_depth_list'][i][obj_idx]
            grasp_scores = end_points['grasp_scores_list'][i][obj_idx]
            grasp_widths = end_points['grasp_widths_list'][i][obj_idx]
            view_graspness = end_points['view_graspness_list'][i][obj_idx]
            top_view_index = end_points['top_view_index_list'][i][obj_idx]
            num_grasp_points = grasp_points.size(0)

            grasp_views = generate_grasp_views(cfgs.num_view).to(pose.device)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')

            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)

            _, view_inds, _ = knn_points(grasp_views.unsqueeze(0), grasp_views_trans.unsqueeze(0), K=1)
            view_inds = view_inds.squeeze(-1).squeeze(0)

            view_graspness_trans = torch.index_select(view_graspness, 1, view_inds)
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)

            top_view_index_trans = (-1 * torch.ones((num_grasp_points, grasp_rotations.shape[1]), dtype=torch.long)
                                    .to(grasp_points.device))
            tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
            top_view_index_trans[tpid, tvip] = tids

            grasp_points_merged.append(grasp_points_trans)
            view_graspness_merged.append(view_graspness_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_rotations_merged.append(grasp_rotations)
            grasp_depth_merged.append(grasp_depth)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)
            grasp_views_rot_merged.append(grasp_views_rot_trans)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)
        view_graspness_merged = torch.cat(view_graspness_merged, dim=0)
        top_view_index_merged = torch.cat(top_view_index_merged, dim=0)
        grasp_rotations_merged = torch.cat(grasp_rotations_merged, dim=0)
        grasp_depth_merged = torch.cat(grasp_depth_merged, dim=0)
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)

        hyp_flat = seed_xyz_hyp[i].reshape(1, num_seed * K_hyp, 3)
        gt_pts = grasp_points_merged.unsqueeze(0)
        dists, nn_inds, _ = knn_points(hyp_flat, gt_pts, K=1)
        dists = dists.squeeze(0).squeeze(-1).view(num_seed, K_hyp)
        nn_inds = nn_inds.squeeze(0).squeeze(-1).view(num_seed, K_hyp)

        best_k = torch.argmin(dists, dim=1)
        best_dist = dists.gather(1, best_k.unsqueeze(-1)).squeeze(-1)
        best_nn = nn_inds.gather(1, best_k.unsqueeze(-1)).squeeze(-1)

        matched_points = grasp_points_merged.index_select(0, best_nn)
        matched_views_rot = grasp_views_rot_merged.index_select(0, best_nn)
        matched_view_graspness = view_graspness_merged.index_select(0, best_nn)
        matched_top_view_index = top_view_index_merged.index_select(0, best_nn)
        matched_grasp_rotations = grasp_rotations_merged.index_select(0, best_nn)
        matched_grasp_depth = grasp_depth_merged.index_select(0, best_nn)
        matched_grasp_scores = grasp_scores_merged.index_select(0, best_nn)
        matched_grasp_widths = grasp_widths_merged.index_select(0, best_nn)

        pred_top_view_ = pred_top_view.view(num_seed, 1, 1, 1).expand(-1, -1, 3, 3)
        top_grasp_views_rot = torch.gather(matched_views_rot, 1, pred_top_view_).squeeze(1)

        pid, vid = torch.where(pred_top_view.unsqueeze(-1) == matched_top_view_index)
        top_grasp_rotations = 12 * torch.ones(num_seed, dtype=torch.int32, device=best_nn.device)
        top_grasp_depth = 4 * torch.ones(num_seed, dtype=torch.int32, device=best_nn.device)
        top_grasp_scores = torch.zeros(num_seed, dtype=torch.float32, device=best_nn.device)
        top_grasp_widths = 0.1 * torch.ones(num_seed, dtype=torch.float32, device=best_nn.device)
        top_grasp_rotations[pid] = torch.gather(matched_grasp_rotations[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_depth[pid] = torch.gather(matched_grasp_depth[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_scores[pid] = torch.gather(matched_grasp_scores[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_widths[pid] = torch.gather(matched_grasp_widths[pid], 1, vid.view(-1, 1)).squeeze(1)

        valid_point_mask = best_dist < valid_dist_thresh
        valid_view_mask = torch.zeros(num_seed, dtype=torch.bool, device=best_nn.device)
        valid_view_mask[pid] = True
        valid_mask = valid_point_mask & valid_view_mask

        valid_points_count += torch.sum(valid_point_mask)
        valid_views_count += torch.sum(valid_view_mask)

        batch_grasp_points.append(matched_points)
        batch_grasp_views_rot.append(top_grasp_views_rot)
        batch_view_graspness.append(matched_view_graspness)
        batch_grasp_rotations.append(top_grasp_rotations)
        batch_grasp_depth.append(top_grasp_depth)
        batch_grasp_scores.append(top_grasp_scores)
        batch_grasp_widths.append(top_grasp_widths)
        batch_valid_mask.append(valid_mask)
        batch_best_k.append(best_k)
        batch_best_dist.append(best_dist)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
    batch_view_graspness = torch.stack(batch_view_graspness, 0)
    batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
    batch_grasp_depth = torch.stack(batch_grasp_depth, 0)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
    batch_valid_mask = torch.stack(batch_valid_mask, 0)
    batch_best_k = torch.stack(batch_best_k, 0)
    batch_best_dist = torch.stack(batch_best_dist, 0)

    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_rotations'] = batch_grasp_rotations
    end_points['batch_grasp_depth'] = batch_grasp_depth
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_view_graspness'] = batch_view_graspness
    end_points['batch_valid_mask'] = batch_valid_mask
    end_points['batch_best_hyp_k'] = batch_best_k
    end_points['batch_best_hyp_dist'] = batch_best_dist
    end_points['C: Valid Points'] = valid_points_count / batch_size

    with torch.no_grad():
        hist = []
        for k in range(K_hyp):
            hist.append((batch_best_k == k).float().mean())
        hist = torch.stack(hist)
        end_points['D: RayMix Valid Ratio'] = batch_valid_mask.float().mean()
        end_points['D: RayMix BestK Hist'] = hist
        end_points['D: RayMix BestDist'] = batch_best_dist.mean()

    return batch_grasp_views_rot, end_points


def _batch_get_key_points(centers: torch.Tensor, Rs: torch.Tensor, widths: torch.Tensor, depths: torch.Tensor):
    """Same role as MMGNet keypoint matching; handles gripper y/z symmetry."""
    height = 0.02
    R_sym = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=centers.dtype,
        device=centers.device,
    )
    key_points = torch.zeros((centers.size(0), 4, 3), dtype=centers.dtype, device=centers.device)
    key_points[:, :, 0] -= depths.unsqueeze(1)
    key_points[:, 1:, 1] -= widths.unsqueeze(1) / 2
    key_points[:, 2, 2] += height / 2
    key_points[:, 3, 2] -= height / 2

    key_points_sym = key_points.detach().clone()
    key_points = torch.matmul(Rs, key_points.transpose(1, 2)).transpose(1, 2)
    key_points_sym = torch.matmul(torch.matmul(Rs, R_sym), key_points_sym.transpose(1, 2)).transpose(1, 2)
    key_points = key_points + centers.unsqueeze(1)
    key_points_sym = key_points_sym + centers.unsqueeze(1)
    return key_points, key_points_sym


def _require_extended(name: str, x: torch.Tensor, num_angle: int) -> torch.Tensor:
    if x.dim() != 3 or x.shape[-1] != num_angle:
        raise RuntimeError(
            f"{name} must be extended-angle label [P,K,A={num_angle}], got {tuple(x.shape)}. "
            f"Regenerate labels with generate_economic.py --extend_angle and ensure dataloader keeps the angle dimension."
        )
    return x

def _build_view_angle_rot_grid(num_view: int, num_angle: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return views [V,3] and canonical rotations [V,A,3,3]."""
    views = generate_grasp_views(num_view).to(device=device, dtype=dtype)
    # GraspNet/EconomicGrasp convention: in-plane angle samples in [0, pi).
    angles = torch.arange(num_angle, device=device, dtype=dtype) * (np.pi / float(num_angle))
    views_repeat = views.repeat_interleave(num_angle, dim=0)
    angles_repeat = angles.repeat(num_view)
    rot = batch_viewpoint_params_to_matrix(-views_repeat, angles_repeat).view(num_view, num_angle, 3, 3)
    return views, rot


def _build_angle_alignment_perm(
    canonical_rot: torch.Tensor,
    transformed_scene_rot: torch.Tensor,
) -> torch.Tensor:
    """Map scene canonical angle index -> transformed/object angle index.

    Args:
        canonical_rot:          [V,A,3,3], R(view_scene, angle_scene)
        transformed_scene_rot:  [V,A,3,3], pose_R @ R(view_obj, angle_obj),
                                already remapped to scene view order by view_inds.

    Returns:
        perm: [V,A], where aligned_label[..., scene_angle] =
              original_label[..., perm[scene_view, scene_angle]].
    """
    if canonical_rot.shape != transformed_scene_rot.shape:
        raise RuntimeError(
            f"canonical_rot and transformed_scene_rot must have same shape, "
            f"got {tuple(canonical_rot.shape)} and {tuple(transformed_scene_rot.shape)}"
        )
    V, A = canonical_rot.shape[:2]
    device = canonical_rot.device
    dtype = canonical_rot.dtype
    N = V * A
    centers = torch.zeros((N, 3), device=device, dtype=dtype)
    widths = 0.02 * torch.ones((N,), device=device, dtype=dtype)
    depths = 0.02 * torch.ones((N,), device=device, dtype=dtype)

    pred_kp, _pred_kp_sym_unused = _batch_get_key_points(centers, canonical_rot.reshape(N, 3, 3), widths, depths)
    trans_kp, trans_kp_sym = _batch_get_key_points(centers, transformed_scene_rot.reshape(N, 3, 3), widths, depths)

    pred_kp = pred_kp.contiguous().view(V, A, -1)
    trans_kp = trans_kp.contiguous().view(V, A, -1)
    trans_kp_sym = trans_kp_sym.contiguous().view(V, A, -1)

    dis, inds, _ = knn_points(pred_kp, trans_kp, K=1)
    dis_sym, inds_sym, _ = knn_points(pred_kp, trans_kp_sym, K=1)
    use_normal = dis <= dis_sym
    perm = torch.where(use_normal, inds, inds_sym).squeeze(-1).long()  # [V,A]
    return perm



def _align_topk_angle_labels(label_topk: torch.Tensor, top_view_index_trans: torch.Tensor, angle_perm: torch.Tensor) -> torch.Tensor:
    """Align [P,K,A] top-K-view angle labels to scene canonical angle indices.

    top_view_index_trans[p,k] is the scene canonical view index of the k-th stored view.
    angle_perm[v,a_scene] gives the original/transformed angle index to gather.
    """
    if label_topk.dim() != 3:
        raise RuntimeError(f"extended angle label must be [P,K,A], got {tuple(label_topk.shape)}")
    if top_view_index_trans.shape != label_topk.shape[:2]:
        raise RuntimeError(
            f"top_view_index_trans shape {tuple(top_view_index_trans.shape)} does not match "
            f"label_topk[:2] {tuple(label_topk.shape[:2])}"
        )
    P, K, A = label_topk.shape
    V = angle_perm.shape[0]
    view_idx = top_view_index_trans.clamp(0, V - 1).long()
    perm_slot = angle_perm[view_idx]  # [P,K,A]
    aligned = torch.gather(label_topk, dim=2, index=perm_slot.to(label_topk.device))
    # For invalid top-view slots, keep deterministic zero labels. They should never be selected.
    invalid = top_view_index_trans < 0
    if invalid.any():
        aligned = aligned.clone()
        aligned[invalid] = 0
    return aligned


def _compute_pointwise_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: [N,3]. Return per-row Euclidean distances [N]."""
    return torch.linalg.norm(a - b, dim=-1)


# -----------------------------------------------------------------------------
# Main function.
# -----------------------------------------------------------------------------

def process_grasp_labels_extend_angle(end_points):
    """Process EconomicGrasp extended-angle labels.

    This is a drop-in alternative to process_grasp_labels() for labels where each
    stored top view has A angle entries. It keeps original outputs and adds
    per-angle candidate labels for Center-View-Angle query training.
    """
    if cfgs is None:
        raise RuntimeError("cfgs is not available. Import utils.arguments.cfgs or paste this function into your loss module.")

    num_view = int(cfgs.num_view)
    num_angle = int(cfgs.num_angle)
    num_depth = int(cfgs.num_depth)
    max_width = float(getattr(cfgs, "grasp_max_width", 0.1))

    seed_xyzs = end_points["xyz_graspable"]            # [B,Q,3]
    pred_top_view_inds = end_points["grasp_top_view_inds"].long()  # [B,Q]
    batch_size, num_samples, _ = seed_xyzs.size()
    device = seed_xyzs.device
    dtype = seed_xyzs.dtype

    valid_points_count = 0.0
    valid_views_count = 0.0

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_view_graspness = []

    # Backward-compatible best-angle labels.
    batch_grasp_rotations = []
    batch_grasp_depth = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_grasp_collisions = []
    batch_valid_mask = []

    # Extended angle labels for candidate-query losses.
    batch_grasp_depth_angle = []
    batch_grasp_score_angle = []
    batch_grasp_width_angle = []
    batch_grasp_collision_angle = []
    batch_grasp_angle_valid_mask = []
    batch_grasp_angle_pos_mask = []

    canonical_views, canonical_rot = _build_view_angle_rot_grid(
        num_view=num_view,
        num_angle=num_angle,
        device=device,
        dtype=dtype,
    )
    zero_angles = torch.zeros(num_view, dtype=dtype, device=device)
    canonical_view_rot_zero = batch_viewpoint_params_to_matrix(-canonical_views, zero_angles)  # [V,3,3]

    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]
        pred_top_view = pred_top_view_inds[i]
        poses = end_points["object_poses_list"][i]

        grasp_points_merged = []
        grasp_views_rot_merged = []
        view_graspness_merged = []
        top_view_index_merged = []
        grasp_depth_angle_merged = []
        grasp_score_angle_merged = []
        grasp_width_angle_merged = []
        grasp_collision_angle_merged = []

        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points["grasp_points_list"][i][obj_idx].to(device=device, dtype=dtype)  # [P,3]
            view_graspness = end_points["view_graspness_list"][i][obj_idx].to(device=device, dtype=dtype)  # [P,V]
            top_view_index = end_points["top_view_index_list"][i][obj_idx].long().to(device=device)       # [P,K]

            # Extended labels [P,K,A]. Strict by design.
            grasp_depth_angle = _require_extended(
                "grasp_depth_list", end_points["grasp_depth_list"][i][obj_idx].long().to(device=device), num_angle
            )
            grasp_score_angle = _require_extended(
                "grasp_scores_list", end_points["grasp_scores_list"][i][obj_idx].to(device=device, dtype=dtype), num_angle
            )
            grasp_width_angle = _require_extended(
                "grasp_widths_list", end_points["grasp_widths_list"][i][obj_idx].to(device=device, dtype=dtype), num_angle
            )
            grasp_collision_angle = _require_extended(
                "grasp_collision_list", end_points["grasp_collision_list"][i][obj_idx].float().to(device=device), num_angle
            )

            num_grasp_points = grasp_points.size(0)
            pose = pose.to(device=device, dtype=dtype)

            # Transform points and views from object frame to scene frame.
            grasp_points_trans = transform_point_cloud(grasp_points, pose, "3x4")
            grasp_views_trans = transform_point_cloud(canonical_views, pose[:3, :3], "3x3")  # object view -> scene

            # view_inds[scene_view] = object_view whose transformed direction is closest to scene_view.
            _, view_inds, _ = knn_points(canonical_views.unsqueeze(0), grasp_views_trans.unsqueeze(0), K=1)
            view_inds = view_inds.squeeze(-1).squeeze(0).long()  # [V]

            # Remap view graspness and zero-angle view rotations to scene view order.
            view_graspness_trans = torch.index_select(view_graspness, 1, view_inds)  # [P,V]
            view_rot_zero_trans = torch.matmul(pose[:3, :3], canonical_view_rot_zero)  # object view order [V,3,3]
            view_rot_zero_trans = torch.index_select(view_rot_zero_trans, 0, view_inds)  # scene view order [V,3,3]
            view_rot_zero_trans = view_rot_zero_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)

            # Map stored top-K object view indices to scene canonical view indices.
            top_view_index_trans = -torch.ones_like(top_view_index, dtype=torch.long, device=device)
            tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
            top_view_index_trans[tpid, tvip] = tids

            # Build angle alignment permutation for this object pose.
            # transformed_rot_scene[scene_view, angle_obj_trans] = pose_R @ R(object_view=view_inds[scene_view], angle_obj)
            transformed_rot = torch.matmul(pose[:3, :3], canonical_rot.reshape(-1, 3, 3)).view(num_view, num_angle, 3, 3)
            transformed_rot_scene = torch.index_select(transformed_rot, 0, view_inds)  # [V,A,3,3]
            angle_perm = _build_angle_alignment_perm(canonical_rot, transformed_rot_scene)  # [V,A]

            # Align [P,K,A] labels to scene canonical angle basis for each stored top-view slot.
            grasp_depth_angle = _align_topk_angle_labels(grasp_depth_angle, top_view_index_trans, angle_perm).long()
            grasp_score_angle = _align_topk_angle_labels(grasp_score_angle, top_view_index_trans, angle_perm).to(dtype)
            grasp_width_angle = _align_topk_angle_labels(grasp_width_angle, top_view_index_trans, angle_perm).to(dtype)
            grasp_collision_angle = _align_topk_angle_labels(grasp_collision_angle, top_view_index_trans, angle_perm).to(dtype)

            grasp_points_merged.append(grasp_points_trans)
            grasp_views_rot_merged.append(view_rot_zero_trans)
            view_graspness_merged.append(view_graspness_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_depth_angle_merged.append(grasp_depth_angle)
            grasp_score_angle_merged.append(grasp_score_angle)
            grasp_width_angle_merged.append(grasp_width_angle)
            grasp_collision_angle_merged.append(grasp_collision_angle)

        # Merge objects.
        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)             # [Pall,3]
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)       # [Pall,V,3,3]
        view_graspness_merged = torch.cat(view_graspness_merged, dim=0)         # [Pall,V]
        top_view_index_merged = torch.cat(top_view_index_merged, dim=0)         # [Pall,K]
        grasp_depth_angle_merged = torch.cat(grasp_depth_angle_merged, dim=0)   # [Pall,K,A]
        grasp_score_angle_merged = torch.cat(grasp_score_angle_merged, dim=0)   # [Pall,K,A]
        grasp_width_angle_merged = torch.cat(grasp_width_angle_merged, dim=0)   # [Pall,K,A]
        grasp_collision_angle_merged = torch.cat(grasp_collision_angle_merged, dim=0) # [Pall,K,A]

        # Nearest object grasp point for each seed.
        _, nn_inds, _ = knn_points(seed_xyz.unsqueeze(0), grasp_points_merged.unsqueeze(0), K=1)
        nn_inds = nn_inds.squeeze(-1).squeeze(0).long()

        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)
        view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)
        top_view_index_merged = torch.index_select(top_view_index_merged, 0, nn_inds)
        grasp_depth_angle_merged = torch.index_select(grasp_depth_angle_merged, 0, nn_inds)
        grasp_score_angle_merged = torch.index_select(grasp_score_angle_merged, 0, nn_inds)
        grasp_width_angle_merged = torch.index_select(grasp_width_angle_merged, 0, nn_inds)
        grasp_collision_angle_merged = torch.index_select(grasp_collision_angle_merged, 0, nn_inds)

        # Gather selected view zero-angle rotation, same as original process_grasp_labels.
        pred_top_view_rot_idx = pred_top_view.view(num_samples, 1, 1, 1).expand(-1, -1, 3, 3)
        top_grasp_views_rot = torch.gather(grasp_views_rot_merged, 1, pred_top_view_rot_idx).squeeze(1)

        # Find selected view slot among stored top-K views.
        pid, vid = torch.where(pred_top_view.unsqueeze(-1) == top_view_index_merged)

        selected_score_angle = torch.zeros((num_samples, num_angle), dtype=dtype, device=device)
        selected_depth_angle = num_depth * torch.ones((num_samples, num_angle), dtype=torch.long, device=device)
        selected_width_angle = max_width * torch.ones((num_samples, num_angle), dtype=dtype, device=device)
        selected_collision_angle = torch.zeros((num_samples, num_angle), dtype=dtype, device=device)
        valid_view_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)

        if pid.numel() > 0:
            selected_score_angle[pid] = grasp_score_angle_merged[pid, vid]
            selected_depth_angle[pid] = grasp_depth_angle_merged[pid, vid].long().clamp(0, num_depth)
            selected_width_angle[pid] = grasp_width_angle_merged[pid, vid].clamp(min=0.0, max=max_width)
            selected_collision_angle[pid] = grasp_collision_angle_merged[pid, vid]
            valid_view_mask[pid] = True

        # Point-view validity, same semantics as original.
        dist = _compute_pointwise_dists(seed_xyz, grasp_points_merged)
        valid_point_mask = dist < 0.005
        valid_points_count = valid_points_count + valid_point_mask.float().sum()
        valid_views_count = valid_views_count + valid_view_mask.float().sum()
        valid_mask = valid_point_mask & valid_view_mask

        # Per-angle valid/positive masks.
        angle_valid_mask = valid_mask.unsqueeze(-1).expand(-1, num_angle).contiguous()
        angle_pos_mask = angle_valid_mask & (selected_score_angle > 0)

        # Backward-compatible collapsed labels: best angle by extended score.
        best_score, best_angle = selected_score_angle.max(dim=-1)  # [Q]
        best_angle_valid = valid_mask & (best_score > 0)
        best_angle_label = best_angle.long()
        best_angle_label = torch.where(
            best_angle_valid,
            best_angle_label,
            torch.full_like(best_angle_label, fill_value=num_angle),
        )
        best_depth = selected_depth_angle.gather(1, best_angle.clamp(0, num_angle - 1).view(-1, 1)).squeeze(1)
        best_depth = torch.where(
            best_angle_valid,
            best_depth,
            torch.full_like(best_depth, fill_value=num_depth),
        )
        best_width = selected_width_angle.gather(1, best_angle.clamp(0, num_angle - 1).view(-1, 1)).squeeze(1)
        best_collision = selected_collision_angle.gather(1, best_angle.clamp(0, num_angle - 1).view(-1, 1)).squeeze(1)

        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(top_grasp_views_rot)
        batch_view_graspness.append(view_graspness_merged)

        batch_grasp_rotations.append(best_angle_label.int())
        batch_grasp_depth.append(best_depth.long())
        batch_grasp_scores.append(best_score.to(dtype))
        batch_grasp_widths.append(best_width.to(dtype))
        batch_grasp_collisions.append(best_collision.to(dtype))
        batch_valid_mask.append(valid_mask)

        batch_grasp_depth_angle.append(selected_depth_angle.long())
        batch_grasp_score_angle.append(selected_score_angle.to(dtype))
        batch_grasp_width_angle.append(selected_width_angle.to(dtype))
        batch_grasp_collision_angle.append(selected_collision_angle.to(dtype))
        batch_grasp_angle_valid_mask.append(angle_valid_mask)
        batch_grasp_angle_pos_mask.append(angle_pos_mask)

    # Stack batch outputs.
    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
    batch_view_graspness = torch.stack(batch_view_graspness, 0)
    batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
    batch_grasp_depth = torch.stack(batch_grasp_depth, 0)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
    batch_grasp_collisions = torch.stack(batch_grasp_collisions, 0)
    batch_valid_mask = torch.stack(batch_valid_mask, 0)

    batch_grasp_depth_angle = torch.stack(batch_grasp_depth_angle, 0)
    batch_grasp_score_angle = torch.stack(batch_grasp_score_angle, 0)
    batch_grasp_width_angle = torch.stack(batch_grasp_width_angle, 0)
    batch_grasp_collision_angle = torch.stack(batch_grasp_collision_angle, 0)
    batch_grasp_angle_valid_mask = torch.stack(batch_grasp_angle_valid_mask, 0)
    batch_grasp_angle_pos_mask = torch.stack(batch_grasp_angle_pos_mask, 0)

    # Original-compatible keys.
    end_points["batch_grasp_point"] = batch_grasp_points
    end_points["batch_grasp_rotations"] = batch_grasp_rotations
    end_points["batch_grasp_depth"] = batch_grasp_depth
    end_points["batch_grasp_score"] = batch_grasp_scores
    end_points["batch_grasp_width"] = batch_grasp_widths
    end_points["batch_grasp_view_graspness"] = batch_view_graspness
    end_points["batch_grasp_collision"] = batch_grasp_collisions
    end_points["batch_valid_mask"] = batch_valid_mask

    # Extended angle-candidate keys.
    end_points["batch_grasp_depth_angle"] = batch_grasp_depth_angle
    end_points["batch_grasp_score_angle"] = batch_grasp_score_angle
    end_points["batch_grasp_width_angle"] = batch_grasp_width_angle
    end_points["batch_grasp_collision_angle"] = batch_grasp_collision_angle
    end_points["batch_grasp_angle_valid_mask"] = batch_grasp_angle_valid_mask
    end_points["batch_grasp_angle_pos_mask"] = batch_grasp_angle_pos_mask

    # Debug scalars.
    end_points["C: Valid Points"] = valid_points_count / float(batch_size)
    with torch.no_grad():
        end_points["D: ExtAngle valid ratio"] = batch_grasp_angle_valid_mask.float().mean()
        end_points["D: ExtAngle pos ratio"] = batch_grasp_angle_pos_mask.float().mean()
        if batch_grasp_angle_valid_mask.any():
            end_points["D: ExtAngle score valid mean"] = batch_grasp_score_angle[batch_grasp_angle_valid_mask].mean()
            end_points["D: ExtAngle score>0 valid"] = (batch_grasp_score_angle[batch_grasp_angle_valid_mask] > 0).float().mean()
        else:
            z = batch_grasp_score_angle.sum() * 0.0
            end_points["D: ExtAngle score valid mean"] = z
            end_points["D: ExtAngle score>0 valid"] = z
        if batch_grasp_angle_pos_mask.any():
            end_points["D: ExtAngle depth01 pos"] = (batch_grasp_depth_angle[batch_grasp_angle_pos_mask] <= 1).float().mean()
            end_points["D: ExtAngle width pos mean"] = batch_grasp_width_angle[batch_grasp_angle_pos_mask].mean()
        else:
            z = batch_grasp_score_angle.sum() * 0.0
            end_points["D: ExtAngle depth01 pos"] = z
            end_points["D: ExtAngle width pos mean"] = z

    return batch_grasp_views_rot, end_points


@torch.no_grad()
def process_grasp_rotation_field_labels(end_points):
    """Generate dense [B,M,V,A] supervision for GeometryAwareDenseFieldRotNet.

    Call this while ``xyz_graspable`` is still the base seed set [B,M,3],
    before RotNet proposals are expanded to Q=M*L.

    This function reuses the same helpers as process_grasp_labels_extend_angle:
      _build_view_angle_rot_grid
      _build_angle_alignment_perm
      _align_topk_angle_labels
      _require_extended
      _compute_pointwise_dists
      transform_point_cloud
      knn_points

    Outputs:
      batch_grasp_rotation_score:        [B,M,V,A]
      batch_grasp_rotation_valid_mask:   [B,M,V,A]
      batch_grasp_rotation_pos_mask:     [B,M,V,A]

    Valid-mask semantics:
      matched seed + stored top-view -> all angles are valid supervision;
      score > 0 is positive, score == 0 is a valid negative;
      unmatched seed or unavailable view is ignored.
    """
    if cfgs is None:
        raise RuntimeError(
            "cfgs is unavailable. Import utils.arguments.cfgs or place this "
            "function in the same label-processing module."
        )

    num_view = int(cfgs.num_view)
    num_angle = int(cfgs.num_angle)
    point_match_thresh = float(
        getattr(cfgs, "grasp_label_point_match_thresh", 0.005)
    )

    seed_xyzs = end_points["xyz_graspable"]  # [B,M,3]
    if seed_xyzs.dim() != 3 or seed_xyzs.shape[-1] != 3:
        raise RuntimeError(
            f"xyz_graspable must be [B,M,3], got {tuple(seed_xyzs.shape)}"
        )

    batch_size, num_seed, _ = seed_xyzs.shape
    device = seed_xyzs.device
    dtype = seed_xyzs.dtype

    canonical_views, canonical_rot = _build_view_angle_rot_grid(
        num_view=num_view,
        num_angle=num_angle,
        device=device,
        dtype=dtype,
    )

    batch_rotation_score = []
    batch_rotation_valid_mask = []
    batch_rotation_pos_mask = []
    batch_rotation_points = []
    batch_point_valid_mask = []

    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # [M,3]
        poses = end_points["object_poses_list"][i]

        grasp_points_merged = []
        top_view_index_merged = []
        grasp_score_angle_merged = []

        if len(poses) == 0:
            raise RuntimeError(f"Batch item {i} contains no object labels")

        for obj_idx, pose in enumerate(poses):
            pose = pose.to(device=device, dtype=dtype)
            grasp_points = end_points["grasp_points_list"][i][obj_idx].to(
                device=device,
                dtype=dtype,
            )
            top_view_index = end_points["top_view_index_list"][i][obj_idx].long().to(
                device=device
            )  # [P,K], object-frame view ids
            grasp_score_angle = _require_extended(
                "grasp_scores_list",
                end_points["grasp_scores_list"][i][obj_idx].to(
                    device=device,
                    dtype=dtype,
                ),
                num_angle,
            )  # [P,K,A]

            if top_view_index.shape != grasp_score_angle.shape[:2]:
                raise RuntimeError(
                    f"top_view_index {tuple(top_view_index.shape)} and score "
                    f"[P,K]={tuple(grasp_score_angle.shape[:2])} do not match"
                )
            if grasp_score_angle.numel() > 0:
                score_min = float(grasp_score_angle.min())
                score_max = float(grasp_score_angle.max())
                if score_min < -1e-5 or score_max > 1.0001:
                    raise RuntimeError(
                        "grasp_scores_list must be decoded to [0,1]. "
                        f"Observed [{score_min:.4f}, {score_max:.4f}]. "
                        "Divide uint8 score*10 labels by 10 in the dataloader."
                    )

            num_grasp_points = grasp_points.shape[0]
            grasp_points_trans = transform_point_cloud(grasp_points, pose, "3x4")
            grasp_views_trans = transform_point_cloud(
                canonical_views,
                pose[:3, :3],
                "3x3",
            )

            # view_inds[scene_view] = object_view.
            _, view_inds, _ = knn_points(
                canonical_views.unsqueeze(0),
                grasp_views_trans.unsqueeze(0),
                K=1,
            )
            view_inds = view_inds.squeeze(0).squeeze(-1).long()  # [V]

            # Convert each stored object-frame top-view id to scene canonical id.
            top_view_index_trans = torch.full_like(top_view_index, -1)
            point_id, slot_id, scene_view_id = torch.where(
                view_inds.view(1, 1, num_view)
                == top_view_index.unsqueeze(-1)
            )
            top_view_index_trans[point_id, slot_id] = scene_view_id

            # Full-rotation angle alignment after applying object pose.
            transformed_rot = torch.matmul(
                pose[:3, :3],
                canonical_rot.reshape(-1, 3, 3),
            ).view(num_view, num_angle, 3, 3)
            transformed_rot_scene = torch.index_select(
                transformed_rot,
                0,
                view_inds,
            )
            angle_perm = _build_angle_alignment_perm(
                canonical_rot,
                transformed_rot_scene,
            )  # [V,A_scene] -> A_object
            grasp_score_angle = _align_topk_angle_labels(
                grasp_score_angle,
                top_view_index_trans,
                angle_perm,
            ).to(dtype)

            grasp_points_merged.append(grasp_points_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_score_angle_merged.append(grasp_score_angle)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)       # [Pall,3]
        top_view_index_merged = torch.cat(top_view_index_merged, dim=0) # [Pall,K]
        grasp_score_angle_merged = torch.cat(
            grasp_score_angle_merged,
            dim=0,
        )  # [Pall,K,A]

        # Match each base seed to one extended-label point.
        _, nn_inds, _ = knn_points(
            seed_xyz.unsqueeze(0),
            grasp_points_merged.unsqueeze(0),
            K=1,
        )
        nn_inds = nn_inds.squeeze(0).squeeze(-1).long()

        matched_points = torch.index_select(
            grasp_points_merged,
            0,
            nn_inds,
        )
        top_view_index_nn = torch.index_select(
            top_view_index_merged,
            0,
            nn_inds,
        )  # [M,K]
        score_angle_nn = torch.index_select(
            grasp_score_angle_merged,
            0,
            nn_inds,
        )  # [M,K,A]

        # Scatter stored top-K view-angle labels into a dense [M,V,A] field.
        dense_score = torch.zeros(
            (num_seed, num_view, num_angle),
            device=device,
            dtype=dtype,
        )
        dense_view_valid = torch.zeros(
            (num_seed, num_view, num_angle),
            device=device,
            dtype=torch.bool,
        )

        seed_id, slot_id = torch.where(top_view_index_nn >= 0)
        if seed_id.numel() > 0:
            scene_view_id = top_view_index_nn[seed_id, slot_id].long()
            dense_score[seed_id, scene_view_id, :] = score_angle_nn[
                seed_id,
                slot_id,
                :,
            ]
            dense_view_valid[seed_id, scene_view_id, :] = True

        point_dist = _compute_pointwise_dists(seed_xyz, matched_points)
        point_valid = point_dist < point_match_thresh
        dense_valid = dense_view_valid & point_valid.view(num_seed, 1, 1)
        dense_score = torch.where(
            dense_valid,
            dense_score,
            torch.zeros_like(dense_score),
        )
        dense_pos = dense_valid & (dense_score > 0)

        # Per-seed top-view remapping coverage before dense scatter.
        # valid_slots: number of stored top-K slots that found a scene-view id.
        # unique_views: number of unique scene-view ids after scatter.
        # duplicate_slots: valid slots lost because multiple slots mapped to the same scene view.
        # missing_slots: top-K slots that remained -1 after view remapping.
        topview_valid_slot_count = (top_view_index_nn >= 0).float().sum(dim=1)  # [M]
        unique_view_count = dense_view_valid.any(dim=-1).float().sum(dim=1)      # [M]
        duplicate_slot_count = (topview_valid_slot_count - unique_view_count).clamp_min(0.0)
        missing_slot_count = float(top_view_index_nn.shape[1]) - topview_valid_slot_count

        tmp_valid_slots = end_points.get("_rotlabel_valid_slots_per_seed", [])
        tmp_duplicate_slots = end_points.get("_rotlabel_duplicate_slots_per_seed", [])
        tmp_missing_slots = end_points.get("_rotlabel_missing_slots_per_seed", [])
        tmp_valid_slots.append(topview_valid_slot_count)
        tmp_duplicate_slots.append(duplicate_slot_count)
        tmp_missing_slots.append(missing_slot_count)
        end_points["_rotlabel_valid_slots_per_seed"] = tmp_valid_slots
        end_points["_rotlabel_duplicate_slots_per_seed"] = tmp_duplicate_slots
        end_points["_rotlabel_missing_slots_per_seed"] = tmp_missing_slots

        batch_rotation_score.append(dense_score)
        batch_rotation_valid_mask.append(dense_valid)
        batch_rotation_pos_mask.append(dense_pos)
        batch_rotation_points.append(matched_points)
        batch_point_valid_mask.append(point_valid)

    batch_rotation_score = torch.stack(batch_rotation_score, dim=0).contiguous()
    batch_rotation_valid_mask = torch.stack(
        batch_rotation_valid_mask,
        dim=0,
    ).contiguous()
    batch_rotation_pos_mask = torch.stack(
        batch_rotation_pos_mask,
        dim=0,
    ).contiguous()
    batch_rotation_points = torch.stack(batch_rotation_points, dim=0).contiguous()
    batch_point_valid_mask = torch.stack(
        batch_point_valid_mask,
        dim=0,
    ).contiguous()

    end_points["batch_grasp_rotation_score"] = batch_rotation_score
    end_points["batch_grasp_rotation_valid_mask"] = (
        batch_rotation_valid_mask
    )
    end_points["batch_grasp_rotation_pos_mask"] = batch_rotation_pos_mask
    end_points["batch_grasp_rotation_point"] = batch_rotation_points
    end_points["batch_grasp_rotation_point_valid"] = batch_point_valid_mask

    # Stack temporary per-seed view-remapping diagnostics for logging.
    if isinstance(end_points.get("_rotlabel_valid_slots_per_seed", None), list):
        end_points["_rotlabel_valid_slots_per_seed"] = torch.stack(
            end_points["_rotlabel_valid_slots_per_seed"], dim=0
        ).to(device=batch_rotation_score.device)
    if isinstance(end_points.get("_rotlabel_duplicate_slots_per_seed", None), list):
        end_points["_rotlabel_duplicate_slots_per_seed"] = torch.stack(
            end_points["_rotlabel_duplicate_slots_per_seed"], dim=0
        ).to(device=batch_rotation_score.device)
    if isinstance(end_points.get("_rotlabel_missing_slots_per_seed", None), list):
        end_points["_rotlabel_missing_slots_per_seed"] = torch.stack(
            end_points["_rotlabel_missing_slots_per_seed"], dim=0
        ).to(device=batch_rotation_score.device)

    with torch.no_grad():
        end_points["D: RotLabel point valid ratio"] = (
            batch_point_valid_mask.float().mean()
        )
        end_points["D: RotLabel field valid ratio"] = (
            batch_rotation_valid_mask.float().mean()
        )
        # Dense label statistics over the full [B,M,V,A] rotation field.
        # Keep this name distinct from RotNet proposal-positive statistics,
        # which are computed only over the selected top-L proposals.
        end_points["D: RotLabel dense positive ratio"] = (
            batch_rotation_pos_mask.float().mean()
        )
        # Backward-compatible alias; remove after log parsers are updated.
        end_points["D: RotLabel positive ratio"] = end_points[
            "D: RotLabel dense positive ratio"
        ]

        if batch_rotation_valid_mask.any():
            end_points["D: RotLabel score valid mean"] = (
                batch_rotation_score[batch_rotation_valid_mask].mean()
            )
        else:
            z = batch_rotation_score.sum() * 0.0
            end_points["D: RotLabel score valid mean"] = z

        has_positive = batch_rotation_pos_mask.flatten(2).any(dim=-1)
        end_points["D: RotLabel seeds with positive"] = (
            has_positive.float().mean()
        )

        best_score = batch_rotation_score.flatten(2).max(dim=-1).values
        if batch_point_valid_mask.any():
            end_points["D: RotLabel best score"] = (
                best_score[batch_point_valid_mask].mean()
            )
        else:
            end_points["D: RotLabel best score"] = best_score.sum() * 0.0

        # View-coverage diagnostics.  These are used to diagnose the common
        # case where a supposedly full-300 view label field only covers ~272
        # unique scene views after object-to-scene view remapping.
        valid_views = batch_rotation_valid_mask.any(dim=-1).float().sum(dim=-1)  # [B,M]
        valid_slots = end_points.get("_rotlabel_valid_slots_per_seed", None)
        duplicate_slots = end_points.get("_rotlabel_duplicate_slots_per_seed", None)
        missing_slots = end_points.get("_rotlabel_missing_slots_per_seed", None)
        if batch_point_valid_mask.any():
            m = batch_point_valid_mask
            end_points["D: RotLabel unique valid views per seed"] = valid_views[m].mean()
            # Backward-compatible alias.
            end_points["D: RotLabel valid views per seed"] = end_points[
                "D: RotLabel unique valid views per seed"
            ]
            end_points["D: RotLabel missing unique views per seed"] = (
                float(num_view) - valid_views[m]
            ).mean()
            end_points["D: RotLabel unique view coverage"] = (
                valid_views[m] / float(num_view)
            ).mean()
            if torch.is_tensor(valid_slots):
                end_points["D: RotLabel topview valid slots per seed"] = valid_slots[m].float().mean()
            if torch.is_tensor(missing_slots):
                end_points["D: RotLabel topview missing slots per seed"] = missing_slots[m].float().mean()
            if torch.is_tensor(duplicate_slots):
                end_points["D: RotLabel topview duplicate slots per seed"] = duplicate_slots[m].float().mean()
        else:
            z = valid_views.sum() * 0.0
            end_points["D: RotLabel unique valid views per seed"] = z
            end_points["D: RotLabel valid views per seed"] = z
            end_points["D: RotLabel missing unique views per seed"] = z
            end_points["D: RotLabel unique view coverage"] = z
            if torch.is_tensor(valid_slots):
                end_points["D: RotLabel topview valid slots per seed"] = z
            if torch.is_tensor(missing_slots):
                end_points["D: RotLabel topview missing slots per seed"] = z
            if torch.is_tensor(duplicate_slots):
                end_points["D: RotLabel topview duplicate slots per seed"] = z

        # Do not keep temporary tensors in end_points after logging.
        end_points.pop("_rotlabel_valid_slots_per_seed", None)
        end_points.pop("_rotlabel_duplicate_slots_per_seed", None)
        end_points.pop("_rotlabel_missing_slots_per_seed", None)

    return None, end_points
