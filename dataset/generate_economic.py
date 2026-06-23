import os
import torch
import numpy as np
import scipy.io as scio
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/data/robotarm/dataset/graspnet', help='the root of the GraspNet dataset')
parser.add_argument('--camera_type', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--keeping_views_numbers', type=int, default=300, help='number of top views to keep per point')
parser.add_argument(
    '--extend_angle',
    action='store_true',
    help=(
        'Save per-view-per-angle labels. If disabled, keep original EconomicGrasp labels, '
        'where each kept view stores only one best angle/depth/width/score/collision.'
    ),
)
parser.add_argument(
    '--save_folder',
    default=None,
    help=(
        'Optional output folder name under dataset_root. Default: '
        'economic_grasp_label_{K}views or economic_grasp_label_{K}views_extend_angle.'
    ),
)

cfgs = parser.parse_args()

obj_data_folders = os.path.join(cfgs.dataset_root, 'grasp_label')
scenes_data_folders = os.path.join(cfgs.dataset_root, 'scenes')
collision_data_folders = os.path.join(cfgs.dataset_root, 'collision_label')


def _normalize_view_graspness(grasp_view_graspness: torch.Tensor) -> torch.Tensor:
    """Min-max normalize per point over views."""
    vmin, _ = torch.min(grasp_view_graspness, dim=-1, keepdim=True)
    vmax, _ = torch.max(grasp_view_graspness, dim=-1, keepdim=True)
    return (grasp_view_graspness - vmin) / (vmax - vmin + 1e-5)


def _build_original_view_labels(
    scene_scores: torch.Tensor,
    scene_width: torch.Tensor,
    scene_collisions: torch.Tensor,
):
    """Original EconomicGrasp label compression.

    Input after score normalization:
        scene_scores:     [Ns, V, A, D], larger is better, 0 means invalid/bad.
        scene_width:      [Ns, V, A, D]
        scene_collisions: [Ns, V, A, D]

    Output:
        rotations:  [Ns, V]      best angle id per view.
        depth:      [Ns, V]      best depth id for the selected angle.
        scores:     [Ns, V]      best score per view.
        widths:     [Ns, V]      width at selected angle/depth.
        collisions: [Ns, V]      collision at selected angle/depth.
    """
    grasp_score_label = scene_scores
    grasp_width_label = scene_width

    # Best depth for each (point, view, angle).
    score_max_depth, depth_idx = grasp_score_label.max(-1)              # [Ns,V,A]
    width_at_depth = grasp_width_label.gather(-1, depth_idx.unsqueeze(-1)).squeeze(-1)  # [Ns,V,A]
    collision_at_depth = scene_collisions.gather(-1, depth_idx.unsqueeze(-1)).squeeze(-1)  # [Ns,V,A]

    # Best angle for each (point, view).
    score_max_angle, angle_idx = score_max_depth.max(-1)                # [Ns,V]
    depth = depth_idx.gather(-1, angle_idx.unsqueeze(-1)).squeeze(-1)   # [Ns,V]
    widths = width_at_depth.gather(-1, angle_idx.unsqueeze(-1)).squeeze(-1)  # [Ns,V]
    collisions = collision_at_depth.gather(-1, angle_idx.unsqueeze(-1)).squeeze(-1)  # [Ns,V]

    return angle_idx, depth, score_max_angle, widths, collisions


def _build_extended_angle_labels(
    scene_scores: torch.Tensor,
    scene_width: torch.Tensor,
    scene_collisions: torch.Tensor,
):
    """Per-view-per-angle labels.

    Input after score normalization:
        scene_scores:     [Ns, V, A, D], larger is better, 0 means invalid/bad.
        scene_width:      [Ns, V, A, D]
        scene_collisions: [Ns, V, A, D]

    Output:
        rotations:  [Ns, V, A]  angle id itself, i.e. rotations[..., a] = a.
        depth:      [Ns, V, A]  best depth id for that fixed angle.
        scores:     [Ns, V, A]  best score over depth for that fixed angle.
        widths:     [Ns, V, A]  width at selected depth.
        collisions: [Ns, V, A]  collision at selected depth.
        valids:     [Ns, V, A]  scores > 0. Useful for masking depth/width losses.
    """
    Ns, V, A, D = scene_scores.shape

    # For each fixed (view, angle), select best depth.
    score_max_depth, depth_idx = scene_scores.max(-1)  # [Ns,V,A]
    widths = scene_width.gather(-1, depth_idx.unsqueeze(-1)).squeeze(-1)  # [Ns,V,A]
    collisions = scene_collisions.gather(-1, depth_idx.unsqueeze(-1)).squeeze(-1)  # [Ns,V,A]

    angle_ids = torch.arange(A, device=scene_scores.device, dtype=torch.long)
    rotations = angle_ids.view(1, 1, A).expand(Ns, V, A).contiguous()
    valids = score_max_depth > 0

    return rotations, depth_idx, score_max_depth, widths, collisions, valids


def _gather_top_views_2d(x: torch.Tensor, view_index: torch.Tensor) -> torch.Tensor:
    """Gather [Ns,V] tensor along view dim with [Ns,K] index -> [Ns,K]."""
    return torch.gather(x, 1, view_index)


def _gather_top_views_3d(x: torch.Tensor, view_index: torch.Tensor) -> torch.Tensor:
    """Gather [Ns,V,A] tensor along view dim with [Ns,K] index -> [Ns,K,A]."""
    A = x.shape[-1]
    idx = view_index.unsqueeze(-1).expand(-1, -1, A)
    return torch.gather(x, 1, idx)


if __name__ == '__main__':
    keeping_views_numbers = int(cfgs.keeping_views_numbers)

    if cfgs.save_folder is not None:
        save_folder_name = cfgs.save_folder
    elif cfgs.extend_angle:
        save_folder_name = f'economic_grasp_label_{keeping_views_numbers}views_extend_angle'
    else:
        save_folder_name = f'economic_grasp_label_{keeping_views_numbers}views'

    save_data_folders = os.path.join(cfgs.dataset_root, save_folder_name)
    os.makedirs(save_data_folders, exist_ok=True)

    # collect the labels from object-level to scene-level
    number = 0
    for label_path in sorted(os.listdir(scenes_data_folders)):
        print(f'---------The {number} scenes----------')
        meta = scio.loadmat(os.path.join(scenes_data_folders, label_path, cfgs.camera_type, 'meta', '0000.mat'))
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        scene_collision = np.load(os.path.join(collision_data_folders, label_path, 'collision_labels.npz'))

        scene_points = []
        scene_pointid = []
        scene_scores = []
        scene_width = []
        scene_collisions = []

        for i, obj_idx in enumerate(obj_idxs):
            object_labels = np.load(os.path.join(obj_data_folders, f'{str(obj_idx - 1).zfill(3)}_labels.npz'))
            points = torch.from_numpy(object_labels['points'])
            pointid = torch.ones(points.shape[0]) * i
            width = torch.from_numpy(object_labels['offsets'][:, :, :, :, 2])
            scores = torch.from_numpy(object_labels['scores'])
            collision = torch.from_numpy(scene_collision[f'arr_{i}']).bool()

            # Collision grasps are invalid for score learning; keep collision tensor separately.
            scores[collision] = 0
            scores[scores < 0] = 0

            scene_points.append(points)
            scene_pointid.append(pointid)
            scene_scores.append(scores)
            scene_width.append(width)
            scene_collisions.append(collision)

        scene_points = torch.cat(scene_points, dim=0)
        scene_pointid = torch.cat(scene_pointid, dim=0)
        scene_scores = torch.cat(scene_scores, dim=0)
        scene_width = torch.cat(scene_width, dim=0)
        scene_collisions = torch.cat(scene_collisions, dim=0)

        # filtering labels in bad points
        threshold = 0.4
        Ns, V, A, D = scene_scores.size()
        grasp_num = V * A * D
        grasp_mask = (scene_scores <= threshold) & (scene_scores > 0)
        graspness = grasp_mask.float().view(Ns, -1).sum(dim=-1) / grasp_num
        filter_mask = graspness > 0

        ori_number = scene_points.shape[0]
        scene_points = scene_points[filter_mask]
        scene_pointid = scene_pointid[filter_mask]
        scene_scores = scene_scores[filter_mask]
        scene_width = scene_width[filter_mask]
        scene_collisions = scene_collisions[filter_mask]
        result_number = scene_points.shape[0]
        print(result_number, ori_number)

        # compute view graspness
        view_u_threshold = 0.6
        grasp_view_valid_mask = (scene_scores <= view_u_threshold) & (scene_scores > 0)
        grasp_view_valid = grasp_view_valid_mask.float()
        grasp_view_graspness = torch.sum(torch.sum(grasp_view_valid, dim=-1), dim=-1) / float(A * D)  # [Ns,V]
        grasp_view_graspness = _normalize_view_graspness(grasp_view_graspness)  # [Ns,V]

        # normalize the score: original GraspNet score is friction/tolerance-like; smaller is better.
        # EconomicGrasp convention: larger is better, 0 means invalid.
        label_mask = (scene_scores > 0) & (scene_width <= 0.1)
        scene_scores[~label_mask] = 0
        po_mask = scene_scores > 0
        scene_scores[po_mask] = 1.1 - scene_scores[po_mask]

        # Build either original per-view labels or extended per-view-per-angle labels.
        if cfgs.extend_angle:
            scene_rotations, scene_depth, scene_scores_out, scene_width_out, scene_collision_selected, scene_valids = \
                _build_extended_angle_labels(scene_scores, scene_width, scene_collisions)
        else:
            scene_rotations, scene_depth, scene_scores_out, scene_width_out, scene_collision_selected = \
                _build_original_view_labels(scene_scores, scene_width, scene_collisions)
            scene_valids = None

        # further view filtering
        _, index = torch.topk(grasp_view_graspness, k=keeping_views_numbers)
        scene_top_view_index = index

        if cfgs.extend_angle:
            scene_rotations = _gather_top_views_3d(scene_rotations, index)  # [Ns,K,A]
            scene_depth = _gather_top_views_3d(scene_depth, index)          # [Ns,K,A]
            scene_scores_out = _gather_top_views_3d(scene_scores_out, index)# [Ns,K,A]
            scene_width_out = _gather_top_views_3d(scene_width_out, index)  # [Ns,K,A]
            scene_collision_selected = _gather_top_views_3d(scene_collision_selected.to(torch.uint8), index).bool()
            scene_valids = _gather_top_views_3d(scene_valids.to(torch.uint8), index).bool()
        else:
            scene_rotations = _gather_top_views_2d(scene_rotations, index)  # [Ns,K]
            scene_depth = _gather_top_views_2d(scene_depth, index)          # [Ns,K]
            scene_scores_out = _gather_top_views_2d(scene_scores_out, index)# [Ns,K]
            scene_width_out = _gather_top_views_2d(scene_width_out, index)  # [Ns,K]
            scene_collision_selected = _gather_top_views_2d(scene_collision_selected.to(torch.uint8), index).bool()

        # save the results
        scene_points_np = scene_points.numpy()
        grasp_rotations = scene_rotations.numpy().astype(np.uint8)
        grasp_depth = scene_depth.numpy().astype(np.uint8)
        grasp_scores = (scene_scores_out.numpy() * 10).astype(np.uint8)
        grasp_widths = (scene_width_out.numpy() * 1000).astype(np.uint8)
        scene_pointid_np = scene_pointid.numpy().astype(np.uint8)
        grasp_view_graspness_np = grasp_view_graspness.numpy()
        grasp_top_view_index = scene_top_view_index.numpy().astype(np.uint16)
        grasp_collisions = scene_collision_selected.numpy().astype(np.uint8)

        save_dict = dict(
            points=scene_points_np,
            rotations=grasp_rotations,
            depth=grasp_depth,
            scores=grasp_scores,
            widths=grasp_widths,
            pointid=scene_pointid_np,
            vgraspness=grasp_view_graspness_np,
            topview=grasp_top_view_index,
            collisions=grasp_collisions,
            extend_angle=np.array([1 if cfgs.extend_angle else 0], dtype=np.uint8),
        )
        if cfgs.extend_angle:
            save_dict['valids'] = scene_valids.numpy().astype(np.uint8)
            save_dict['num_angle'] = np.array([A], dtype=np.uint8)
            save_dict['num_depth'] = np.array([D], dtype=np.uint8)

        np.savez(os.path.join(save_data_folders, f'{label_path}_labels.npz'), **save_dict)

        number += 1

    print('---------Finishing----------')
