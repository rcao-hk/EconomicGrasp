"""Main CDF task objective plus flag-gated geometry objectives.

Main's metric L1 remains necessary for metric grounding. With MoGe enabled it
trains only the metric anchor/pose path, while aligned pointmap losses train the
shape DPT. No new ranking loss, teacher, or action mining is introduced here.
"""
from __future__ import annotations
import torch
from .geometry import shape_losses,ray_consistency_loss,interval_score


def objective(ep,model_cfg,loss_cfg):
    from models.loss_economicgrasp_depth_kview_transformer import get_loss
    total,ep=get_loss(ep,use_cdf=True)
    task=total-ep['A: DepthReg Loss']
    zero=total*0
    extras={'global_shape':zero,'local_shape':zero,'reprojection':zero,'interval':zero}
    if model_cfg.use_moge or (model_cfg.use_rayrope and model_cfg.uncertainty=='learned'):
        if 'gt_depth_m' not in ep: raise KeyError('Geometry loss requires gt_depth_m supervision')
        gt=ep['gt_depth_m'].float()
        if gt.ndim==3: gt=gt[:,None]
        if gt.ndim!=4 or gt.shape[1]!=1: raise ValueError('gt_depth_m must be [B,1,H,W]')
        valid=torch.isfinite(gt)&(gt>model_cfg.min_depth)&(gt<model_cfg.max_depth)
        safe=torch.where(valid,gt,torch.zeros_like(gt))
        g=ep['mr_geometry']
        if model_cfg.use_moge:
            extras.update(shape_losses(g['points'],safe,ep['K'],gt.shape[-2:],loss_cfg))
            extras['reprojection']=ray_consistency_loss(g['canonical'],g['rays'])
        if model_cfg.use_rayrope and model_cfg.uncertainty=='learned':
            extras['interval']=interval_score(g['depth'],g['sigma'],safe,valid,model_cfg.interval_coverage)
    weighted=(loss_cfg.global_shape*extras['global_shape']+
              loss_cfg.local_shape*extras['local_shape']+
              loss_cfg.reprojection*extras['reprojection']+loss_cfg.interval*extras['interval'])
    total=total+weighted
    stats={k:v.detach() for k,v in extras.items()}
    stats.update(loss=total.detach(),task=task.detach(),metric_depth=ep['A: DepthReg Loss'].detach())
    depth=ep.get('depth_map_pred')
    if depth is not None:
        dd=depth.detach().float()
        stats['depth_mean_m']=dd.mean()
        stats['depth_spatial_std_m']=dd.flatten(1).std(dim=1).mean()
        stats['depth_out_of_range_fraction']=((~torch.isfinite(dd))|(dd<=model_cfg.min_depth)|(dd>=model_cfg.max_depth)).float().mean()
    width=ep.get('grasp_width_pred_angle_depth')
    if width is not None:
        stats['decoded_width_mean_m']=(1.2*width.detach().float()/10).clamp(0,.1).mean()
        stats['decoded_width_zero_fraction']=(width.detach()<=0).float().mean()
    if model_cfg.use_moge:
        stats['metric_anchor_mean_m']=ep['mr_geometry']['anchor'].detach().mean()
        stats['shape_gauge_mean']=ep['mr_geometry']['gauge'].detach().mean()
    return total,task,stats
