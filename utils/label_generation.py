import os
import sys
import time
import pdb

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
