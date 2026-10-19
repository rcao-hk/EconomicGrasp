"""Angle-depth CDF and depth-wise width head for the original point-centric grasp backbone.

Each group is defined by an XYZ center and selected 300-view approach direction.
Unlike the legacy point head, the score field covers all 12 in-plane angles
and four 1-cm insertion depths, aligned with CVA-CDF supervision. The backbone,
center sampling, view selection and cylinder grouping are unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PointGraspCDFHead(nn.Module):
    """Predictions: CDF logits [B,T,Q,A,D], width x10 [B,D,Q,A]."""

    def __init__(self, in_channels=256, hidden_channels=256,
                 num_angle=12, num_depth=4, num_thresholds=6):
        super().__init__()
        self.num_angle=int(num_angle)
        self.num_depth=int(num_depth)
        self.num_thresholds=int(num_thresholds)
        if min(self.num_angle,self.num_depth,self.num_thresholds)<1:
            raise ValueError("Angles, depths and thresholds must be positive")
        self.shared = nn.Sequential(
            nn.Conv1d(in_channels,hidden_channels,1),
            nn.ReLU(inplace=True),
        )
        self.cdf = nn.Conv1d(
            hidden_channels,self.num_thresholds*self.num_angle*self.num_depth,1)
        self.width = nn.Conv1d(hidden_channels,self.num_angle*self.num_depth,1)

    def forward(self,group_features,end_points):
        if group_features.dim()!=3 or group_features.shape[1]!=256:
            raise ValueError("group_features must be [B,256,Q]")
        b,_,q=group_features.shape
        shared=self.shared(group_features)
        cdf_logits=self.cdf(shared)
        width_scaled=self.width(shared)
        cdf_logits=cdf_logits.reshape(
            b,self.num_thresholds,self.num_angle,self.num_depth,q
        ).permute(0,1,4,2,3).contiguous()
        width_scaled=width_scaled.reshape(
            b,self.num_angle,self.num_depth,q
        ).permute(0,2,3,1).contiguous()
        if cdf_logits.shape!=(b,self.num_thresholds,q,self.num_angle,self.num_depth):
            raise AssertionError("CDF output ordering error")
        if width_scaled.shape!=(b,self.num_depth,q,self.num_angle):
            raise AssertionError("Width output ordering error")
        end_points["grasp_cdf_pred_angle_depth"]=cdf_logits
        end_points["grasp_width_pred_angle_depth"]=width_scaled
        return end_points


def decode_point_cdf(end_points,*,num_angle=12,num_depth=4,max_width=0.1):
    """The same expected-CDF-utility, argmax(A*D) and width units as CVA-CDF.

    Inference only, with no ground-truth point/pose/depth inputs.
    Output per image: [Q,17] in standard GraspNet GraspGroup ordering.
    """
    from utils.loss_utils import batch_viewpoint_params_to_matrix

    centers=end_points["xyz_graspable"]
    views=end_points["grasp_top_view_xyz"]
    cdf=end_points["grasp_cdf_pred_angle_depth"]
    width=end_points["grasp_width_pred_angle_depth"]
    if centers.dim()!=3 or centers.shape[-1]!=3:
        raise ValueError("centers must be [B,Q,3]")
    b,q,_=centers.shape
    A,D=int(num_angle),int(num_depth)
    if views.shape!=(b,q,3) or cdf.dim()!=5 or cdf.shape[0]!=b or cdf.shape[2:]!=(q,A,D):
        raise ValueError("View/CDF tensor shape mismatch")
    if width.shape!=(b,D,q,A):
        raise ValueError("Width tensor shape mismatch")
    if not all(bool(torch.isfinite(v).all()) for v in (centers,views,cdf,width)):
        raise FloatingPointError("Non-finite point-CDF grasp parameters")
    utility=torch.sigmoid(cdf.float()).mean(dim=1)  # [B,Q,A,D]
    values,idx=utility.reshape(b,q,A*D).max(dim=-1)
    angle_idx=idx//D
    depth_idx=idx%D
    width_qad=width.float().permute(0,2,3,1).contiguous()
    batch_id=torch.arange(b,device=centers.device).unsqueeze(1).expand(b,q)
    query_id=torch.arange(q,device=centers.device).unsqueeze(0).expand(b,q)
    selected_width=width_qad[batch_id,query_id,angle_idx,depth_idx]
    selected_width=(1.2*selected_width/10.0).clamp(0.0,float(max_width))
    approach=(-views.float()).reshape(b*q,3)
    angle=(angle_idx.float()*np.pi/float(A)).reshape(b*q)
    rot=batch_viewpoint_params_to_matrix(approach,angle).reshape(b,q,9)
    depth=(depth_idx.float()+1.0)*0.01
    height=torch.full_like(depth,0.02)
    object_id=-torch.ones_like(depth)
    grasp=torch.cat(
        (values.unsqueeze(-1),
         selected_width.unsqueeze(-1),
         height.unsqueeze(-1),
         depth.unsqueeze(-1),
         rot,
         centers.float(),
         object_id.unsqueeze(-1)),dim=-1)
    if grasp.shape!=(b,q,17):
        raise AssertionError("Bad GraspGroup shape")
    return [grasp[i] for i in range(b)]
