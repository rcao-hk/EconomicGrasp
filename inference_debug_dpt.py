#!/usr/bin/env python3
import csv
import importlib
import inspect
import json
import math
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils.arguments import cfgs
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn
from utils.loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix

try:
    from graspnetAPI.grasp import GraspGroup
except Exception:
    try:
        from graspnetAPI import GraspGroup  # type: ignore
    except Exception:
        GraspGroup = None  # type: ignore


# -----------------------------------------------------------------------------
# Basic setup
# -----------------------------------------------------------------------------
SAVE_ROOT = getattr(cfgs, 'save_dir', None) or getattr(cfgs, 'log_dir', None) or 'results'
os.makedirs(SAVE_ROOT, exist_ok=True)


def my_worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# -----------------------------------------------------------------------------
# Dataset / dataloader
# -----------------------------------------------------------------------------
def build_dataset(args) -> Any:
    load_label = True  # label matching always needs labels
    if getattr(args, 'multi_modal', False):
        dataset = GraspNetMultiDataset(
            args.dataset_root,
            split='{}'.format(args.test_mode),
            camera=args.camera,
            num_points=args.num_point,
            remove_outlier=True,
            augment=False,
            load_label=load_label,
        )
    else:
        dataset = GraspNetDataset(
            args.dataset_root,
            split='{}'.format(args.test_mode),
            camera=args.camera,
            num_points=args.num_point,
            remove_outlier=True,
            augment=False,
            load_label=load_label,
        )
    return dataset


def build_eval_subset(dataset: Any, sample_interval: float, annos_per_scene: int = 256):
    total = len(dataset)
    if sample_interval <= 0:
        raise ValueError(f'sample_interval must be > 0, got {sample_interval}')
    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices

    stride = max(1, int(round(1.0 / sample_interval)))
    indices: List[int] = []
    num_scenes = (total + annos_per_scene - 1) // annos_per_scene
    for scene_id in range(num_scenes):
        start = scene_id * annos_per_scene
        end = min((scene_id + 1) * annos_per_scene, total)
        scene_len = end - start
        local_indices = list(range(0, scene_len, stride))
        indices.extend([start + idx for idx in local_indices])

    subset = Subset(dataset, indices)
    return subset, indices


def build_dataloader(args):
    full_dataset = build_dataset(args)
    sample_interval = float(getattr(args, 'sample_interval', 1.0))
    eval_dataset, sampled_indices = build_eval_subset(full_dataset, sample_interval)
    test_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, 'num_workers', 2),
        worker_init_fn=my_worker_init_fn,
        collate_fn=collate_fn,
    )
    return full_dataset, eval_dataset, test_dataloader, sampled_indices


FULL_TEST_DATASET, TEST_DATASET, TEST_DATALOADER, SAMPLED_INDICES = build_dataloader(cfgs)
print(f'Total test samples: {len(FULL_TEST_DATASET)}')
print(f'Evaluated samples:  {len(TEST_DATASET)}')
print(f'sample_interval:    {getattr(cfgs, "sample_interval", 1.0)}')

sample0 = FULL_TEST_DATASET[0]
need_keys = [
    'object_poses_list',
    'grasp_points_list',
    'grasp_rotations_list',
    'grasp_depth_list',
    'grasp_scores_list',
    'grasp_widths_list',
    'view_graspness_list',
    'top_view_index_list',
]
miss = [k for k in need_keys if k not in sample0]
if miss:
    raise KeyError('[Eval] Dataset does not provide label-matching keys. Missing: ' + ', '.join(miss))


# -----------------------------------------------------------------------------
# Model init helpers
# -----------------------------------------------------------------------------
def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            out[k[len('module.'):]] = v
        else:
            out[k] = v
    return out


def _add_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            out[k] = v
        else:
            out['module.' + k] = v
    return out


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ['model_state_dict', 'state_dict', 'model', 'net']:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f'Unsupported checkpoint type: {type(ckpt)}')


def import_model_module() -> Tuple[Any, str]:
    module_name = getattr(cfgs, 'model_module', None) or getattr(cfgs, 'model', None)
    if module_name is None:
        raise ValueError('cfgs.model or cfgs.model_module must be set')
    module_name = str(module_name).replace('.py', '')
    full_name = module_name if module_name.startswith('models.') else f'models.{module_name}'
    module = importlib.import_module(full_name)
    return module, full_name


def _is_model_ctor(name: str, obj: Any) -> bool:
    if name.startswith('_'):
        return False
    if not callable(obj):
        return False
    if inspect.isclass(obj):
        return issubclass(obj, torch.nn.Module)
    return True


def find_model_ctor(module: Any) -> Tuple[Callable[..., Any], str]:
    preferred_names = [
        getattr(cfgs, 'model_class', None),
        getattr(cfgs, 'model_ctor', None),
        getattr(cfgs, 'model', None),
    ]
    for name in preferred_names:
        if name and hasattr(module, name):
            obj = getattr(module, name)
            if _is_model_ctor(name, obj):
                return obj, name

    candidates: List[Tuple[str, Any]] = []
    for name, obj in vars(module).items():
        if _is_model_ctor(name, obj) and ('economicgrasp' in name):
            candidates.append((name, obj))

    if not candidates:
        raise RuntimeError(f'No economicgrasp-like ctor found in module {module.__name__}')

    # prefer exact DPT-like naming
    def _score(item: Tuple[str, Any]) -> Tuple[int, int, str]:
        name = item[0]
        exact_dpt = 0 if name == 'economicgrasp_dpt' else 1
        starts_dpt = 0 if name.startswith('economicgrasp_dpt') else 1
        return (exact_dpt, starts_dpt, name)

    candidates.sort(key=_score)
    return candidates[0][1], candidates[0][0]


def instantiate_model(module: Any, ctor: Callable[..., Any]) -> torch.nn.Module:
    candidate_kwargs = {
        'encoder': getattr(cfgs, 'encoder', 'vitb'),
        'tok_feat_dim': getattr(cfgs, 'tok_feat_dim', 128),
        'seed_feat_dim': getattr(cfgs, 'seed_feat_dim', 512),
        'fuse_type': getattr(cfgs, 'fuse_type', None),
        'cylinder_radius': getattr(cfgs, 'cylinder_radius', 0.05),
        'min_depth': getattr(cfgs, 'min_depth', 0.2),
        'max_depth': getattr(cfgs, 'max_depth', 1.0),
        'bin_num': getattr(cfgs, 'bin_num', 256),
        'freeze_backbone': getattr(cfgs, 'freeze_backbone', True),
        'use_gt_xyz_for_train': getattr(cfgs, 'use_gt_xyz_for_train', False),
        'is_training': False,
        'vis_dir': None,
        'vis_every': 10**9,
        'debug_print_every': 10**9,
    }

    sig = inspect.signature(ctor)
    kwargs = {}
    for k, v in candidate_kwargs.items():
        if k in sig.parameters and v is not None:
            kwargs[k] = v

    # net = ctor(**kwargs)
    net = build_model_from_cfg(cfgs)
    if not isinstance(net, torch.nn.Module):
        raise TypeError(f'Model ctor {ctor} did not return nn.Module, got {type(net)}')
    return net


def build_model_from_cfg(cfgs):
    model_spec = cfgs.model

    # 只支持模块名；若用户误传 xxx.yyy，也截掉后半段
    module_name = model_spec.split(".")[0]

    MODEL = importlib.import_module(f"models.{module_name}")

    if hasattr(MODEL, "get_model"):
        return MODEL.get_model(cfgs)

    raise ValueError(
        f"models.{module_name} does not provide get_model(cfgs)."
    )


def load_model() -> Tuple[torch.nn.Module, Callable[..., Any], str, str]:
    module, full_module_name = import_model_module()
    ctor, ctor_name = find_model_ctor(module)
    net = instantiate_model(module, ctor)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    if getattr(cfgs, 'checkpoint_path', None) is None:
        raise ValueError('cfgs.checkpoint_path must be set for evaluation')

    ckpt = torch.load(cfgs.checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)

    loaded = False
    load_errors: List[str] = []
    for cand in [
        state_dict,
        _strip_module_prefix(state_dict),
        _add_module_prefix(state_dict),
    ]:
        try:
            net.load_state_dict(cand, strict=True)
            loaded = True
            break
        except Exception as e:
            load_errors.append(str(e))

    if not loaded:
        # final fallback: non-strict with stripped keys
        try:
            msg = net.load_state_dict(_strip_module_prefix(state_dict), strict=False)
            print(f'[WARN] Non-strict checkpoint load used: {msg}')
            loaded = True
        except Exception as e:
            load_errors.append(str(e))

    if not loaded:
        raise RuntimeError('Failed to load checkpoint. Errors:\n' + '\n----\n'.join(load_errors))

    print(f'-> imported model module: {full_module_name}')
    print(f'-> using model ctor:      {ctor_name}')
    print(f'-> loaded checkpoint:     {cfgs.checkpoint_path}')

    process_grasp_labels = getattr(module, 'process_grasp_labels', None)
    if process_grasp_labels is None:
        from utils.label_generation import process_grasp_labels  # type: ignore

    return net, process_grasp_labels, full_module_name, ctor_name


NET, PROCESS_GRASP_LABELS, MODEL_MODULE_NAME, MODEL_CTOR_NAME = load_model()
DEVICE = next(NET.parameters()).device
VIEW_DIRS = generate_grasp_views(int(cfgs.num_view)).float().to(DEVICE)
GRASP_DEPTH_BIN_SIZE = float(getattr(cfgs, 'grasp_depth_bin_size', 0.01))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def move_batch_to_device(batch_data: Dict[str, Any], device: torch.device):
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
    return batch_data


def _safe_class_argmax(x: torch.Tensor, num_class: int) -> torch.Tensor:
    valid_cands = {num_class, num_class + 1}
    if x.dim() == 3:
        if x.shape[1] in valid_cands:      # (B,C,M)
            return torch.argmax(x, dim=1)
        if x.shape[2] in valid_cands:      # (B,M,C)
            return torch.argmax(x, dim=2)
    elif x.dim() == 4:
        if x.shape[1] == 1 and x.shape[2] in valid_cands:
            return torch.argmax(x.squeeze(1), dim=1)
        if x.shape[2] == 1 and x.shape[1] in valid_cands:
            return torch.argmax(x.squeeze(2), dim=1)
    raise ValueError(
        f'Unsupported shape for classification tensor: {tuple(x.shape)}, '
        f'num_class={num_class} (or num_class+1={num_class+1})'
    )


def _safe_score_composite(score_pred: torch.Tensor) -> torch.Tensor:
    bins = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1.0], device=score_pred.device, dtype=score_pred.dtype)
    if score_pred.dim() != 3:
        raise ValueError(f'Unsupported score_pred shape: {tuple(score_pred.shape)}')
    if score_pred.shape[1] == 6:  # (B,6,M)
        prob = score_pred
        s = prob.sum(dim=1, keepdim=True)
        if torch.mean(torch.abs(s - 1.0)) > 1e-2:
            prob = torch.softmax(prob, dim=1)
        return torch.sum(prob * bins.view(1, 6, 1), dim=1)
    if score_pred.shape[2] == 6:  # (B,M,6)
        prob = score_pred
        s = prob.sum(dim=2, keepdim=True)
        if torch.mean(torch.abs(s - 1.0)) > 1e-2:
            prob = torch.softmax(prob, dim=2)
        return torch.sum(prob * bins.view(1, 1, 6), dim=2)
    raise ValueError(f'Unsupported score_pred shape: {tuple(score_pred.shape)}')


def _safe_width_decode(width_pred: torch.Tensor) -> torch.Tensor:
    if width_pred.dim() == 3 and width_pred.shape[1] == 1:
        width_pred = width_pred.squeeze(1)
    elif width_pred.dim() == 3 and width_pred.shape[2] == 1:
        width_pred = width_pred.squeeze(2)
    elif width_pred.dim() != 2:
        raise ValueError(f'Unsupported width_pred shape: {tuple(width_pred.shape)}')
    width = 1.2 * width_pred / 10.0
    width = torch.clamp(width, min=0.0, max=float(cfgs.grasp_max_width))
    return width


def _find_view_score_tensor(end_points: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    candidate_names = ['view_score', 'grasp_view_score', 'view_scores', 'grasp_view_scores']
    for k in candidate_names:
        if k in end_points and torch.is_tensor(end_points[k]):
            return end_points[k], k
    for k, v in end_points.items():
        if not torch.is_tensor(v):
            continue
        if v.dim() not in (3, 4):
            continue
        shape = tuple(v.shape)
        if shape.count(int(cfgs.num_view)) >= 1 and 'batch_' not in k and 'label' not in k:
            return v, k
    return None, None


def _prepare_view_score(pred_view_score: Optional[torch.Tensor], gt_view_score: torch.Tensor) -> Optional[torch.Tensor]:
    if pred_view_score is None:
        return None
    if pred_view_score.dim() == 3:
        if pred_view_score.shape == gt_view_score.shape:
            return pred_view_score
        if pred_view_score.shape[1] == gt_view_score.shape[2] and pred_view_score.shape[2] == gt_view_score.shape[1]:
            return pred_view_score.transpose(1, 2).contiguous()
    elif pred_view_score.dim() == 4:
        if pred_view_score.shape[1] == 1:
            return _prepare_view_score(pred_view_score.squeeze(1), gt_view_score)
        if pred_view_score.shape[2] == 1:
            return _prepare_view_score(pred_view_score.squeeze(2), gt_view_score)
    raise ValueError(f'Unsupported view score shape: pred={tuple(pred_view_score.shape)}, gt={tuple(gt_view_score.shape)}')


def _find_gt_depth(batch_or_end_points: Dict[str, Any]) -> Optional[torch.Tensor]:
    candidates = [
        'gt_depth_m',
        'gt_depth',
        'depth_gt',
        'depth_map_gt',
        'depth',
    ]
    for key in candidates:
        if key in batch_or_end_points and torch.is_tensor(batch_or_end_points[key]):
            x = batch_or_end_points[key]
            if x.dim() == 3:
                x = x.unsqueeze(1)
            elif x.dim() == 4 and x.shape[1] != 1:
                x = x[:, :1]
            return x
    return None


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode='nearest')


def _gather_depth_by_token_idx(depth_b1hw: torch.Tensor, idx_bm: torch.Tensor) -> torch.Tensor:
    B, _, H, W = depth_b1hw.shape
    flat = depth_b1hw.view(B, -1)
    return torch.gather(flat, 1, idx_bm.long())


def attach_label_matching_eval_local(end_points: Dict[str, Any]) -> Dict[str, Any]:
    _, end_points = PROCESS_GRASP_LABELS(end_points)
    return end_points


# -----------------------------------------------------------------------------
# Meters
# -----------------------------------------------------------------------------
def _init_grasp_meter() -> Dict[str, Any]:
    return {
        'num_valid': 0.0,
        'center_err_sum': 0.0,
        'center_z_err_sum': 0.0,
        'view_acc_sum': 0.0,
        'view_ang_err_sum': 0.0,
        'view_score_mae_sum': 0.0,
        'view_score_mae_weight': 0.0,
        'view_score_key': None,
        'angle_acc_sum': 0.0,
        'angle_err_deg_sum': 0.0,
        'depth_acc_sum': 0.0,
        'grasp_depth_idx_mae_sum': 0.0,
        'grasp_depth_err_m_sum': 0.0,
        'width_mae_sum': 0.0,
        'score_mae_sum': 0.0,
        'seed_depth_mae_sum': 0.0,
        'seed_depth_mae_weight': 0.0,
    }


def _init_depth_meter() -> Dict[str, float]:
    return {
        'num_valid_pix': 0.0,
        'mae_sum': 0.0,
        'rmse_sum': 0.0,
        'abs_rel_sum': 0.0,
        'sq_rel_sum': 0.0,
        'log_rmse_sum': 0.0,
        'delta1_sum': 0.0,
        'delta2_sum': 0.0,
        'delta3_sum': 0.0,
    }


def _update_grasp_meter(meter: Dict[str, Any], stats: Dict[str, Any]) -> None:
    for k in list(meter.keys()):
        if k == 'view_score_key':
            if stats.get(k) is not None:
                meter[k] = stats[k]
            continue
        meter[k] += float(stats.get(k, 0.0))


def _update_depth_meter(meter: Dict[str, float], stats: Dict[str, float]) -> None:
    for k in meter:
        meter[k] += float(stats.get(k, 0.0))


def _finalize_grasp_meter(meter: Dict[str, Any]) -> Dict[str, Any]:
    n = max(float(meter['num_valid']), 1e-8)
    seed_w = max(float(meter['seed_depth_mae_weight']), 1e-8)
    vw = max(float(meter['view_score_mae_weight']), 1e-8)
    return {
        'num_valid': float(meter['num_valid']),
        'center_err_mean_m': float(meter['center_err_sum'] / n),
        'center_z_err_mean_m': float(meter['center_z_err_sum'] / n),
        'view_acc': float(meter['view_acc_sum'] / n),
        'view_ang_err_mean_deg': float(meter['view_ang_err_sum'] / n),
        'view_score_mae': None if meter['view_score_mae_weight'] <= 0 else float(meter['view_score_mae_sum'] / vw),
        'view_score_key': meter['view_score_key'],
        'angle_acc': float(meter['angle_acc_sum'] / n),
        'angle_err_mean_deg': float(meter['angle_err_deg_sum'] / n),
        'grasp_depth_acc': float(meter['depth_acc_sum'] / n),
        'grasp_depth_idx_mae': float(meter['grasp_depth_idx_mae_sum'] / n),
        'grasp_depth_err_mean_m': float(meter['grasp_depth_err_m_sum'] / n),
        'width_mae_m': float(meter['width_mae_sum'] / n),
        'score_mae': float(meter['score_mae_sum'] / n),
        'seed_depth_mae_m': None if meter['seed_depth_mae_weight'] <= 0 else float(meter['seed_depth_mae_sum'] / seed_w),
    }


def _finalize_depth_meter(meter: Dict[str, float]) -> Dict[str, Any]:
    n = max(float(meter['num_valid_pix']), 1e-8)
    return {
        'num_valid_pix': float(meter['num_valid_pix']),
        'depth_map_mae_m': float(meter['mae_sum'] / n),
        'depth_map_rmse_m': float(math.sqrt(meter['rmse_sum'] / n)),
        'depth_map_abs_rel': float(meter['abs_rel_sum'] / n),
        'depth_map_sq_rel': float(meter['sq_rel_sum'] / n),
        'depth_map_log_rmse': float(math.sqrt(meter['log_rmse_sum'] / n)),
        'depth_map_delta1': float(meter['delta1_sum'] / n),
        'depth_map_delta2': float(meter['delta2_sum'] / n),
        'depth_map_delta3': float(meter['delta3_sum'] / n),
    }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compute_depth_metrics(end_points: Dict[str, Any], batch_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
    pred = end_points.get('depth_map_pred', None)
    gt = _find_gt_depth(batch_data)
    if pred is None or gt is None or (not torch.is_tensor(pred)) or (not torch.is_tensor(gt)):
        return None

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if pred.dim() != 4:
        return None

    gt = _resize_like(gt.to(pred.device).float(), pred)
    pred = pred.float()

    valid = torch.isfinite(pred) & torch.isfinite(gt)
    valid &= (gt > 0)
    valid &= (gt >= float(getattr(cfgs, 'min_depth', 0.0)))
    valid &= (gt <= float(getattr(cfgs, 'max_depth', 1e9)))

    if not valid.any():
        return None

    pred_v = pred[valid]
    gt_v = gt[valid]
    abs_err = (pred_v - gt_v).abs()
    sq_err = (pred_v - gt_v) ** 2
    ratio = torch.maximum(pred_v / gt_v.clamp_min(1e-8), gt_v / pred_v.clamp_min(1e-8))
    log_sq = (torch.log(pred_v.clamp_min(1e-8)) - torch.log(gt_v.clamp_min(1e-8))) ** 2

    return {
        'num_valid_pix': float(gt_v.numel()),
        'mae_sum': float(abs_err.sum().item()),
        'rmse_sum': float(sq_err.sum().item()),
        'abs_rel_sum': float((abs_err / gt_v.clamp_min(1e-8)).sum().item()),
        'sq_rel_sum': float((sq_err / gt_v.clamp_min(1e-8)).sum().item()),
        'log_rmse_sum': float(log_sq.sum().item()),
        'delta1_sum': float((ratio < 1.25).float().sum().item()),
        'delta2_sum': float((ratio < 1.25 ** 2).float().sum().item()),
        'delta3_sum': float((ratio < 1.25 ** 3).float().sum().item()),
    }


def compute_grasp_metrics(end_points: Dict[str, Any], batch_data: Dict[str, Any]) -> Dict[str, Any]:
    valid = end_points['batch_valid_mask'].bool()  # (B,M)
    eps = 1e-8

    center_err = torch.norm(end_points['xyz_graspable'] - end_points['batch_grasp_point'], dim=-1)  # (B,M)
    center_z_err = (end_points['xyz_graspable'][..., 2] - end_points['batch_grasp_point'][..., 2]).abs()

    gt_view_score = end_points['batch_grasp_view_graspness']  # (B,M,V)
    gt_view_inds = torch.argmax(gt_view_score, dim=-1)
    pred_view_inds = end_points['grasp_top_view_inds']
    view_acc = (pred_view_inds == gt_view_inds).float()

    pred_top_view_xyz = end_points.get('grasp_top_view_xyz', None)
    if pred_top_view_xyz is None or (not torch.is_tensor(pred_top_view_xyz)):
        pred_top_view_xyz = VIEW_DIRS.index_select(0, pred_view_inds.reshape(-1)).view_as(end_points['grasp_top_view_xyz'])
    gt_top_view_xyz = VIEW_DIRS.index_select(0, gt_view_inds.reshape(-1)).view(pred_view_inds.shape[0], pred_view_inds.shape[1], 3)
    view_cos = F.cosine_similarity(pred_top_view_xyz.float(), gt_top_view_xyz.float(), dim=-1).clamp(-1.0, 1.0)
    view_ang_err_deg = torch.rad2deg(torch.acos(view_cos))

    pred_view_score_raw, pred_view_score_key = _find_view_score_tensor(end_points)
    pred_view_score = _prepare_view_score(pred_view_score_raw, gt_view_score) if pred_view_score_raw is not None else None
    if pred_view_score is not None:
        view_score_mae = (pred_view_score.float() - gt_view_score.float()).abs()
        view_score_mae_sum = float((view_score_mae * valid.unsqueeze(-1).float()).sum().item())
        view_score_mae_weight = float(valid.unsqueeze(-1).float().sum().item() * gt_view_score.shape[-1])
    else:
        view_score_mae_sum = 0.0
        view_score_mae_weight = 0.0

    pred_angle_idx = _safe_class_argmax(end_points['grasp_angle_pred'], int(cfgs.num_angle))
    gt_angle_idx = end_points['batch_grasp_rotations'].long().clamp(min=0, max=int(cfgs.num_angle) - 1)
    angle_acc = (pred_angle_idx == gt_angle_idx).float()
    angle_diff = (pred_angle_idx - gt_angle_idx).abs().float()
    angle_diff = torch.minimum(angle_diff, float(cfgs.num_angle) - angle_diff)
    angle_err_deg = angle_diff * (180.0 / float(cfgs.num_angle))

    pred_depth_idx = _safe_class_argmax(end_points['grasp_depth_pred'], int(cfgs.num_depth))
    gt_depth_idx = end_points['batch_grasp_depth'].long().clamp(min=0, max=int(cfgs.num_depth) - 1)
    depth_acc = (pred_depth_idx == gt_depth_idx).float()
    grasp_depth_idx_mae = (pred_depth_idx.float() - gt_depth_idx.float()).abs()
    grasp_depth_err_m = grasp_depth_idx_mae * GRASP_DEPTH_BIN_SIZE

    pred_width = _safe_width_decode(end_points['grasp_width_pred'])
    gt_width = end_points['batch_grasp_width'].float()
    width_mae = (pred_width - gt_width).abs()

    pred_score = _safe_score_composite(end_points['grasp_score_pred'])
    gt_score = end_points['batch_grasp_score'].float()
    score_mae = (pred_score - gt_score).abs()

    gt_depth_map = _find_gt_depth(batch_data)
    seed_depth_mae_sum = 0.0
    seed_depth_mae_weight = 0.0
    if gt_depth_map is not None and 'token_sel_idx' in end_points and 'depth_map_pred' in end_points:
        pred_map = end_points['depth_map_pred']
        if pred_map.dim() == 3:
            pred_map = pred_map.unsqueeze(1)
        gt_map = _resize_like(gt_depth_map.to(pred_map.device).float(), pred_map)
        idx = end_points['token_sel_idx'].long()
        pred_seed_depth = _gather_depth_by_token_idx(pred_map, idx)
        gt_seed_depth = _gather_depth_by_token_idx(gt_map, idx)
        seed_depth_valid = valid & torch.isfinite(pred_seed_depth) & torch.isfinite(gt_seed_depth) & (gt_seed_depth > 0)
        if seed_depth_valid.any():
            seed_depth_mae_sum = float((pred_seed_depth - gt_seed_depth).abs()[seed_depth_valid].sum().item())
            seed_depth_mae_weight = float(seed_depth_valid.float().sum().item())

    denom = valid.float().sum().clamp_min(eps)
    return {
        'num_valid': float(valid.float().sum().item()),
        'center_err_sum': float((center_err * valid.float()).sum().item()),
        'center_z_err_sum': float((center_z_err * valid.float()).sum().item()),
        'view_acc_sum': float((view_acc * valid.float()).sum().item()),
        'view_ang_err_sum': float((view_ang_err_deg * valid.float()).sum().item()),
        'view_score_mae_sum': view_score_mae_sum,
        'view_score_mae_weight': view_score_mae_weight,
        'view_score_key': pred_view_score_key,
        'angle_acc_sum': float((angle_acc * valid.float()).sum().item()),
        'angle_err_deg_sum': float((angle_err_deg * valid.float()).sum().item()),
        'depth_acc_sum': float((depth_acc * valid.float()).sum().item()),
        'grasp_depth_idx_mae_sum': float((grasp_depth_idx_mae * valid.float()).sum().item()),
        'grasp_depth_err_m_sum': float((grasp_depth_err_m * valid.float()).sum().item()),
        'width_mae_sum': float((width_mae * valid.float()).sum().item()),
        'score_mae_sum': float((score_mae * valid.float()).sum().item()),
        'seed_depth_mae_sum': seed_depth_mae_sum,
        'seed_depth_mae_weight': seed_depth_mae_weight,
        '_denom_check': float(denom.item()),
    }


def compute_per_sample_rows(end_points: Dict[str, Any], batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    valid = end_points['batch_valid_mask'].bool()
    gt_depth_map = _find_gt_depth(batch_data)
    pred_depth_map = end_points.get('depth_map_pred', None)
    if torch.is_tensor(pred_depth_map) and pred_depth_map.dim() == 3:
        pred_depth_map = pred_depth_map.unsqueeze(1)
    if gt_depth_map is not None and torch.is_tensor(pred_depth_map):
        gt_depth_map = _resize_like(gt_depth_map.to(pred_depth_map.device).float(), pred_depth_map)

    pred_angle_idx = _safe_class_argmax(end_points['grasp_angle_pred'], int(cfgs.num_angle))
    gt_angle_idx = end_points['batch_grasp_rotations'].long().clamp(min=0, max=int(cfgs.num_angle) - 1)
    angle_diff = (pred_angle_idx - gt_angle_idx).abs().float()
    angle_diff = torch.minimum(angle_diff, float(cfgs.num_angle) - angle_diff)
    angle_err_deg = angle_diff * (180.0 / float(cfgs.num_angle))

    pred_depth_idx = _safe_class_argmax(end_points['grasp_depth_pred'], int(cfgs.num_depth))
    gt_depth_idx = end_points['batch_grasp_depth'].long().clamp(min=0, max=int(cfgs.num_depth) - 1)

    pred_width = _safe_width_decode(end_points['grasp_width_pred'])
    pred_score = _safe_score_composite(end_points['grasp_score_pred'])

    gt_view_inds = torch.argmax(end_points['batch_grasp_view_graspness'], dim=-1)
    pred_view_inds = end_points['grasp_top_view_inds']
    gt_top_view_xyz = VIEW_DIRS.index_select(0, gt_view_inds.reshape(-1)).view(pred_view_inds.shape[0], pred_view_inds.shape[1], 3)
    pred_top_view_xyz = end_points['grasp_top_view_xyz']
    view_cos = F.cosine_similarity(pred_top_view_xyz.float(), gt_top_view_xyz.float(), dim=-1).clamp(-1.0, 1.0)
    view_ang_err_deg = torch.rad2deg(torch.acos(view_cos))

    for b in range(valid.shape[0]):
        vb = valid[b]
        num_valid = int(vb.float().sum().item())
        row: Dict[str, Any] = {
            'scene_idx': int(batch_data['scene_idx'][b].item()) if 'scene_idx' in batch_data else -1,
            'anno_idx': int(batch_data['anno_idx'][b].item()) if 'anno_idx' in batch_data else -1,
            'num_valid': num_valid,
        }
        if num_valid > 0:
            row.update({
                'center_err_mean_m': float(torch.norm(end_points['xyz_graspable'][b] - end_points['batch_grasp_point'][b], dim=-1)[vb].mean().item()),
                'view_acc': float((pred_view_inds[b] == gt_view_inds[b])[vb].float().mean().item()),
                'view_ang_err_mean_deg': float(view_ang_err_deg[b][vb].mean().item()),
                'angle_acc': float((pred_angle_idx[b] == gt_angle_idx[b])[vb].float().mean().item()),
                'angle_err_mean_deg': float(angle_err_deg[b][vb].mean().item()),
                'grasp_depth_acc': float((pred_depth_idx[b] == gt_depth_idx[b])[vb].float().mean().item()),
                'grasp_depth_idx_mae': float((pred_depth_idx[b].float() - gt_depth_idx[b].float()).abs()[vb].mean().item()),
                'grasp_depth_err_mean_m': float(((pred_depth_idx[b].float() - gt_depth_idx[b].float()).abs() * GRASP_DEPTH_BIN_SIZE)[vb].mean().item()),
                'width_mae_m': float((pred_width[b] - end_points['batch_grasp_width'][b].float()).abs()[vb].mean().item()),
                'score_mae': float((pred_score[b] - end_points['batch_grasp_score'][b].float()).abs()[vb].mean().item()),
            })
        else:
            row.update({
                'center_err_mean_m': None,
                'view_acc': None,
                'view_ang_err_mean_deg': None,
                'angle_acc': None,
                'angle_err_mean_deg': None,
                'grasp_depth_acc': None,
                'grasp_depth_idx_mae': None,
                'grasp_depth_err_mean_m': None,
                'width_mae_m': None,
                'score_mae': None,
            })

        if gt_depth_map is not None and torch.is_tensor(pred_depth_map):
            pred_b = pred_depth_map[b:b + 1]
            gt_b = gt_depth_map[b:b + 1]
            valid_pix = torch.isfinite(pred_b) & torch.isfinite(gt_b) & (gt_b > 0)
            valid_pix &= (gt_b >= float(getattr(cfgs, 'min_depth', 0.0)))
            valid_pix &= (gt_b <= float(getattr(cfgs, 'max_depth', 1e9)))
            if valid_pix.any():
                abs_err = (pred_b - gt_b).abs()[valid_pix]
                sq_err = ((pred_b - gt_b) ** 2)[valid_pix]
                gt_v = gt_b[valid_pix]
                pred_v = pred_b[valid_pix]
                ratio = torch.maximum(pred_v / gt_v.clamp_min(1e-8), gt_v / pred_v.clamp_min(1e-8))
                row.update({
                    'depth_map_num_valid_pix': int(valid_pix.float().sum().item()),
                    'depth_map_mae_m': float(abs_err.mean().item()),
                    'depth_map_rmse_m': float(torch.sqrt(sq_err.mean()).item()),
                    'depth_map_abs_rel': float((abs_err / gt_v.clamp_min(1e-8)).mean().item()),
                    'depth_map_delta1': float((ratio < 1.25).float().mean().item()),
                })
            else:
                row.update({
                    'depth_map_num_valid_pix': 0,
                    'depth_map_mae_m': None,
                    'depth_map_rmse_m': None,
                    'depth_map_abs_rel': None,
                    'depth_map_delta1': None,
                })
        else:
            row.update({
                'depth_map_num_valid_pix': 0,
                'depth_map_mae_m': None,
                'depth_map_rmse_m': None,
                'depth_map_abs_rel': None,
                'depth_map_delta1': None,
            })
        rows.append(row)
    return rows



# -----------------------------------------------------------------------------
# NMS Top-K grasp diagnostics
# -----------------------------------------------------------------------------
def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average-rank implementation without scipy; ranks start from 1."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)  # ranks are 1-based; last rank in group is j
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _pearson_np(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-12:
        return None
    return float((a * b).sum() / denom)


def _spearman_np(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[mask]
    gt = gt[mask]
    if pred.size < 2:
        return None
    return _pearson_np(_rankdata_average(pred), _rankdata_average(gt))


def _ndcg_at_k(gt_relevance_in_pred_order: np.ndarray, k: int) -> Optional[float]:
    rel = np.asarray(gt_relevance_in_pred_order, dtype=np.float64)
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return None
    rel = rel[:k]
    discounts = 1.0 / np.log2(np.arange(rel.size, dtype=np.float64) + 2.0)
    # gt scores are already in [0, 1], so use them directly as relevance.
    dcg = float((rel * discounts).sum())
    ideal = np.sort(rel)[::-1]
    idcg = float((ideal * discounts).sum())
    if idcg <= 1e-12:
        return None
    return dcg / idcg


def _grasp_group_to_array(grasp_group: Any) -> np.ndarray:
    if hasattr(grasp_group, 'grasp_group_array'):
        return np.asarray(grasp_group.grasp_group_array)
    return np.asarray(grasp_group)


def _init_top50_meter() -> Dict[str, float]:
    return {
        'num_samples': 0.0,
        'pre_nms_count_sum': 0.0,
        'post_nms_count_sum': 0.0,
        'topk_count_sum': 0.0,
        'topk_valid_label_count_sum': 0.0,
        'topk_depth_acc_sum': 0.0,
        'topk_depth_idx_mae_sum': 0.0,
        'topk_depth_err_m_sum': 0.0,
        'topk_score_mae_sum': 0.0,
        'topk_pred_score_sum': 0.0,
        'topk_gt_score_sum': 0.0,
        'topk_gt_positive_sum': 0.0,
        'topk_gt_ge_0p4_sum': 0.0,
        'topk_gt_ge_0p6_sum': 0.0,
        'topk_spearman_sum': 0.0,
        'topk_spearman_count': 0.0,
        'topk_ndcg_sum': 0.0,
        'topk_ndcg_count': 0.0,
        'top1_valid_count': 0.0,
        'top1_depth_err_m_sum': 0.0,
        'top1_depth_acc_sum': 0.0,
        'top1_score_mae_sum': 0.0,
        'top1_pred_score_sum': 0.0,
        'top1_gt_score_sum': 0.0,
    }


def _update_top50_meter(meter: Dict[str, float], stats: Dict[str, float]) -> None:
    for k, v in stats.items():
        if k in meter and v is not None and np.isfinite(v):
            meter[k] += float(v)


def _finalize_top50_meter(meter: Dict[str, float]) -> Dict[str, Any]:
    ns = max(meter['num_samples'], 1e-8)
    ntop = max(meter['topk_count_sum'], 1e-8)
    nvalid = max(meter['topk_valid_label_count_sum'], 1e-8)
    ntop1 = max(meter['top1_valid_count'], 1e-8)
    return {
        'num_samples': float(meter['num_samples']),
        'pre_nms_count_mean': float(meter['pre_nms_count_sum'] / ns),
        'post_nms_count_mean': float(meter['post_nms_count_sum'] / ns),
        'topk_count_mean': float(meter['topk_count_sum'] / ns),
        'topk_valid_label_ratio': float(meter['topk_valid_label_count_sum'] / ntop),
        'topk_depth_acc': float(meter['topk_depth_acc_sum'] / nvalid),
        'topk_depth_idx_mae': float(meter['topk_depth_idx_mae_sum'] / nvalid),
        'topk_depth_err_mean_m': float(meter['topk_depth_err_m_sum'] / nvalid),
        'topk_score_mae': float(meter['topk_score_mae_sum'] / nvalid),
        'topk_pred_score_mean': float(meter['topk_pred_score_sum'] / nvalid),
        'topk_gt_score_mean': float(meter['topk_gt_score_sum'] / nvalid),
        'topk_gt_positive_ratio': float(meter['topk_gt_positive_sum'] / nvalid),
        'topk_gt_ge_0p4_ratio': float(meter['topk_gt_ge_0p4_sum'] / nvalid),
        'topk_gt_ge_0p6_ratio': float(meter['topk_gt_ge_0p6_sum'] / nvalid),
        'topk_spearman_pred_gt_score': None if meter['topk_spearman_count'] <= 0 else float(meter['topk_spearman_sum'] / meter['topk_spearman_count']),
        'topk_ndcg': None if meter['topk_ndcg_count'] <= 0 else float(meter['topk_ndcg_sum'] / meter['topk_ndcg_count']),
        'top1_depth_acc': float(meter['top1_depth_acc_sum'] / ntop1),
        'top1_depth_err_mean_m': float(meter['top1_depth_err_m_sum'] / ntop1),
        'top1_score_mae': float(meter['top1_score_mae_sum'] / ntop1),
        'top1_pred_score_mean': float(meter['top1_pred_score_sum'] / ntop1),
        'top1_gt_score_mean': float(meter['top1_gt_score_sum'] / ntop1),
    }


def _decode_grasps_for_one_sample(
    end_points: Dict[str, Any],
    b: int,
    pred_angle_idx: torch.Tensor,
    pred_depth_idx: torch.Tensor,
    pred_width: torch.Tensor,
    pred_score: torch.Tensor,
) -> np.ndarray:
    """
    Decode M seed-level predictions into GraspGroup-compatible array.
    The last column stores the seed index as object_id, so after NMS we can map
    top grasps back to batch_grasp_depth / batch_grasp_score for diagnostics.
    """
    centers = end_points['xyz_graspable'][b].float()  # (M,3)
    M = centers.shape[0]
    top_view_xyz = end_points.get('grasp_top_view_xyz', None)
    if top_view_xyz is None or (not torch.is_tensor(top_view_xyz)):
        view_inds = end_points['grasp_top_view_inds'][b].long()
        top_view_xyz_b = VIEW_DIRS.index_select(0, view_inds.reshape(-1)).view(M, 3)
    else:
        top_view_xyz_b = top_view_xyz[b].float()

    angle = pred_angle_idx[b].float() * (math.pi / float(cfgs.num_angle))
    # Same convention as common EconomicGrasp decoding: use -top_view as approaching vector.
    rot = batch_viewpoint_params_to_matrix(-top_view_xyz_b.reshape(-1, 3), angle.reshape(-1)).view(M, 3, 3)

    depth = (pred_depth_idx[b].float() + 1.0) * GRASP_DEPTH_BIN_SIZE
    height = torch.full_like(depth, 0.02)
    score = pred_score[b].float()
    width = pred_width[b].float()
    obj_id = torch.arange(M, device=centers.device, dtype=torch.float32)

    arr = torch.cat([
        score.view(M, 1),
        width.view(M, 1),
        height.view(M, 1),
        depth.view(M, 1),
        rot.reshape(M, 9),
        centers.reshape(M, 3),
        obj_id.view(M, 1),
    ], dim=1)
    return _to_numpy(arr).astype(np.float64)


def _recover_seed_ids_from_top_array(top_arr: np.ndarray, centers_np: np.ndarray, num_seed: int) -> np.ndarray:
    """Prefer object_id column; fallback to nearest center matching."""
    if top_arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    ids = np.rint(top_arr[:, -1]).astype(np.int64)
    if np.all((ids >= 0) & (ids < num_seed)):
        return ids
    # Fallback: nearest center.  top_arr translation columns are 13:16.
    top_centers = top_arr[:, 13:16]
    d2 = ((top_centers[:, None, :] - centers_np[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(d2, axis=1).astype(np.int64)


def compute_top50_nms_metrics(end_points: Dict[str, Any], batch_data: Dict[str, Any]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if GraspGroup is None:
        raise ImportError(
            'GraspGroup is required for NMS Top-50 diagnostics. '
            'Please install/import graspnetAPI so that `from graspnetAPI.grasp import GraspGroup` works.'
        )

    topk = int(getattr(cfgs, 'topk_error', getattr(cfgs, 'topk', 50)))
    nms_trans = float(getattr(cfgs, 'topk_nms_trans_thresh', 0.03))
    nms_rot = float(getattr(cfgs, 'topk_nms_rot_thresh', 30.0 / 180.0 * math.pi))

    pred_angle_idx = _safe_class_argmax(end_points['grasp_angle_pred'], int(cfgs.num_angle))
    pred_depth_idx = _safe_class_argmax(end_points['grasp_depth_pred'], int(cfgs.num_depth))
    pred_width = _safe_width_decode(end_points['grasp_width_pred'])
    pred_score = _safe_score_composite(end_points['grasp_score_pred'])

    gt_depth_idx = end_points['batch_grasp_depth'].long().clamp(min=0, max=int(cfgs.num_depth))
    gt_score = end_points['batch_grasp_score'].float()
    valid = end_points['batch_valid_mask'].bool()

    B, M = valid.shape
    agg = _init_top50_meter()
    rows: List[Dict[str, Any]] = []

    for b in range(B):
        arr = _decode_grasps_for_one_sample(end_points, b, pred_angle_idx, pred_depth_idx, pred_width, pred_score)
        pre_nms_count = int(arr.shape[0])

        gg = GraspGroup(arr)
        # Score sorting before and after NMS makes behavior explicit and close to evaluation-time usage.
        if hasattr(gg, 'sort_by_score'):
            gg = gg.sort_by_score()
        gg = gg.nms(nms_trans, nms_rot)
        if hasattr(gg, 'sort_by_score'):
            gg = gg.sort_by_score()

        gg_arr = _grasp_group_to_array(gg)
        if gg_arr.ndim == 1:
            gg_arr = gg_arr.reshape(1, -1)
        post_nms_count = int(gg_arr.shape[0]) if gg_arr.size > 0 else 0
        top_arr = gg_arr[:min(topk, post_nms_count)] if post_nms_count > 0 else np.zeros((0, 17), dtype=np.float64)
        top_count = int(top_arr.shape[0])

        centers_np = _to_numpy(end_points['xyz_graspable'][b].float())
        seed_ids = _recover_seed_ids_from_top_array(top_arr, centers_np, M)

        if top_count > 0:
            seed_ids_t = torch.as_tensor(seed_ids, device=valid.device, dtype=torch.long)
            valid_top = valid[b].index_select(0, seed_ids_t)
            pred_depth_top = pred_depth_idx[b].index_select(0, seed_ids_t).float()
            gt_depth_top = gt_depth_idx[b].index_select(0, seed_ids_t).float()
            pred_score_top = pred_score[b].index_select(0, seed_ids_t).float()
            gt_score_top = gt_score[b].index_select(0, seed_ids_t).float()

            valid_np = _to_numpy(valid_top.bool())
            pred_depth_np = _to_numpy(pred_depth_top)
            gt_depth_np = _to_numpy(gt_depth_top)
            pred_score_np = _to_numpy(pred_score_top)
            gt_score_np = _to_numpy(gt_score_top)

            if valid_np.any():
                pd = pred_depth_np[valid_np]
                gd = gt_depth_np[valid_np]
                ps = pred_score_np[valid_np]
                gs = gt_score_np[valid_np]
                depth_idx_abs = np.abs(pd - gd)
                depth_acc = (pd == gd).astype(np.float64)
                score_abs = np.abs(ps - gs)

                spearman = _spearman_np(ps, gs)
                ndcg = _ndcg_at_k(gs, k=min(topk, gs.size))

                valid_count = int(valid_np.sum())
                row_top = {
                    'top50_valid_label_count': valid_count,
                    'top50_valid_label_ratio': float(valid_count / max(top_count, 1)),
                    'top50_depth_acc': float(depth_acc.mean()),
                    'top50_depth_idx_mae': float(depth_idx_abs.mean()),
                    'top50_depth_err_mean_m': float((depth_idx_abs * GRASP_DEPTH_BIN_SIZE).mean()),
                    'top50_score_mae': float(score_abs.mean()),
                    'top50_pred_score_mean': float(ps.mean()),
                    'top50_gt_score_mean': float(gs.mean()),
                    'top50_gt_positive_ratio': float((gs > 0.0).mean()),
                    'top50_gt_ge_0p4_ratio': float((gs >= 0.4).mean()),
                    'top50_gt_ge_0p6_ratio': float((gs >= 0.6).mean()),
                    'top50_spearman_pred_gt_score': spearman,
                    'top50_ndcg': ndcg,
                }

                agg['topk_valid_label_count_sum'] += float(valid_count)
                agg['topk_depth_acc_sum'] += float(depth_acc.sum())
                agg['topk_depth_idx_mae_sum'] += float(depth_idx_abs.sum())
                agg['topk_depth_err_m_sum'] += float((depth_idx_abs * GRASP_DEPTH_BIN_SIZE).sum())
                agg['topk_score_mae_sum'] += float(score_abs.sum())
                agg['topk_pred_score_sum'] += float(ps.sum())
                agg['topk_gt_score_sum'] += float(gs.sum())
                agg['topk_gt_positive_sum'] += float((gs > 0.0).sum())
                agg['topk_gt_ge_0p4_sum'] += float((gs >= 0.4).sum())
                agg['topk_gt_ge_0p6_sum'] += float((gs >= 0.6).sum())
                if spearman is not None and np.isfinite(spearman):
                    agg['topk_spearman_sum'] += float(spearman)
                    agg['topk_spearman_count'] += 1.0
                if ndcg is not None and np.isfinite(ndcg):
                    agg['topk_ndcg_sum'] += float(ndcg)
                    agg['topk_ndcg_count'] += 1.0

                # Top-1 after NMS and sorting.
                if bool(valid_np[0]):
                    top1_depth_idx_abs = abs(float(pred_depth_np[0] - gt_depth_np[0]))
                    top1_score_abs = abs(float(pred_score_np[0] - gt_score_np[0]))
                    agg['top1_valid_count'] += 1.0
                    agg['top1_depth_err_m_sum'] += top1_depth_idx_abs * GRASP_DEPTH_BIN_SIZE
                    agg['top1_depth_acc_sum'] += float(pred_depth_np[0] == gt_depth_np[0])
                    agg['top1_score_mae_sum'] += top1_score_abs
                    agg['top1_pred_score_sum'] += float(pred_score_np[0])
                    agg['top1_gt_score_sum'] += float(gt_score_np[0])
                    row_top.update({
                        'top1_depth_acc': float(pred_depth_np[0] == gt_depth_np[0]),
                        'top1_depth_err_m': float(top1_depth_idx_abs * GRASP_DEPTH_BIN_SIZE),
                        'top1_score_mae': float(top1_score_abs),
                        'top1_pred_score': float(pred_score_np[0]),
                        'top1_gt_score': float(gt_score_np[0]),
                    })
                else:
                    row_top.update({
                        'top1_depth_acc': None,
                        'top1_depth_err_m': None,
                        'top1_score_mae': None,
                        'top1_pred_score': float(pred_score_np[0]) if pred_score_np.size > 0 else None,
                        'top1_gt_score': float(gt_score_np[0]) if gt_score_np.size > 0 else None,
                    })
            else:
                row_top = {
                    'top50_valid_label_count': 0,
                    'top50_valid_label_ratio': 0.0,
                    'top50_depth_acc': None,
                    'top50_depth_idx_mae': None,
                    'top50_depth_err_mean_m': None,
                    'top50_score_mae': None,
                    'top50_pred_score_mean': float(pred_score_np.mean()) if pred_score_np.size > 0 else None,
                    'top50_gt_score_mean': None,
                    'top50_gt_positive_ratio': None,
                    'top50_gt_ge_0p4_ratio': None,
                    'top50_gt_ge_0p6_ratio': None,
                    'top50_spearman_pred_gt_score': None,
                    'top50_ndcg': None,
                    'top1_depth_acc': None,
                    'top1_depth_err_m': None,
                    'top1_score_mae': None,
                    'top1_pred_score': float(pred_score_np[0]) if pred_score_np.size > 0 else None,
                    'top1_gt_score': None,
                }
        else:
            row_top = {
                'top50_valid_label_count': 0,
                'top50_valid_label_ratio': 0.0,
                'top50_depth_acc': None,
                'top50_depth_idx_mae': None,
                'top50_depth_err_mean_m': None,
                'top50_score_mae': None,
                'top50_pred_score_mean': None,
                'top50_gt_score_mean': None,
                'top50_gt_positive_ratio': None,
                'top50_gt_ge_0p4_ratio': None,
                'top50_gt_ge_0p6_ratio': None,
                'top50_spearman_pred_gt_score': None,
                'top50_ndcg': None,
                'top1_depth_acc': None,
                'top1_depth_err_m': None,
                'top1_score_mae': None,
                'top1_pred_score': None,
                'top1_gt_score': None,
            }

        row = {
            'scene_idx': int(batch_data['scene_idx'][b].item()) if 'scene_idx' in batch_data else -1,
            'anno_idx': int(batch_data['anno_idx'][b].item()) if 'anno_idx' in batch_data else -1,
            'pre_nms_count': pre_nms_count,
            'post_nms_count': post_nms_count,
            'top50_count': top_count,
            'top50_nms_trans_thresh': nms_trans,
            'top50_nms_rot_thresh_rad': nms_rot,
        }
        row.update(row_top)
        rows.append(row)

        agg['num_samples'] += 1.0
        agg['pre_nms_count_sum'] += float(pre_nms_count)
        agg['post_nms_count_sum'] += float(post_nms_count)
        agg['topk_count_sum'] += float(top_count)

    return agg, rows

# -----------------------------------------------------------------------------
# Main eval loop
# -----------------------------------------------------------------------------
def run_eval() -> None:
    batch_interval = 20
    NET.eval()
    tic = time.time()

    grasp_meter = _init_grasp_meter()
    depth_meter = _init_depth_meter()
    per_sample_rows: List[Dict[str, Any]] = []
    top50_meter = _init_top50_meter()

    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        batch_data = move_batch_to_device(batch_data, DEVICE)

        with torch.no_grad():
            end_points = NET(batch_data)
            end_points = attach_label_matching_eval_local(end_points)
            grasp_stats = compute_grasp_metrics(end_points, batch_data)
            depth_stats = compute_depth_metrics(end_points, batch_data)
            top50_stats, top50_rows = compute_top50_nms_metrics(end_points, batch_data)

        _update_grasp_meter(grasp_meter, grasp_stats)
        if depth_stats is not None:
            _update_depth_meter(depth_meter, depth_stats)
        _update_top50_meter(top50_meter, top50_stats)

        sample_rows = compute_per_sample_rows(end_points, batch_data)
        if len(sample_rows) == len(top50_rows):
            for r, tr in zip(sample_rows, top50_rows):
                r.update({k: v for k, v in tr.items() if k not in ('scene_idx', 'anno_idx')})
        else:
            print(f'[WARN] per-sample row count mismatch: base={len(sample_rows)} top50={len(top50_rows)}')
        per_sample_rows.extend(sample_rows)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            running = {
                'grasp': _finalize_grasp_meter(grasp_meter),
                'depth': _finalize_depth_meter(depth_meter),
                'top50_nms': _finalize_top50_meter(top50_meter),
            }
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc - tic) / batch_interval))
            print(json.dumps(running, indent=2))
            tic = time.time()

    final_summary = {
        'config': {
            'model_module': MODEL_MODULE_NAME,
            'model_ctor': MODEL_CTOR_NAME,
            'checkpoint_path': cfgs.checkpoint_path,
            'camera': cfgs.camera,
            'test_mode': cfgs.test_mode,
            'batch_size': cfgs.batch_size,
            'sample_interval': getattr(cfgs, 'sample_interval', 1.0),
            'grasp_depth_bin_size': GRASP_DEPTH_BIN_SIZE,
        },
        'grasp_metrics': _finalize_grasp_meter(grasp_meter),
        'depth_metrics': _finalize_depth_meter(depth_meter),
        'top50_nms_metrics': _finalize_top50_meter(top50_meter),
    }

    save_dir = os.path.join(SAVE_ROOT, 'economicgrasp_dpt_error_eval')
    os.makedirs(save_dir, exist_ok=True)

    tag = f'{str(cfgs.test_mode)}_{str(cfgs.camera)}_{MODEL_CTOR_NAME}'
    summary_path = os.path.join(save_dir, f'summary_{tag}.json')
    csv_path = os.path.join(save_dir, f'per_sample_{tag}.csv')

    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)

    fieldnames: List[str] = []
    for row in per_sample_rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_sample_rows:
            writer.writerow(row)

    print('[FINAL]')
    print(json.dumps(final_summary, indent=2))
    print(f'[SAVE] {summary_path}')
    print(f'[SAVE] {csv_path}')


if __name__ == '__main__':
    run_eval()
