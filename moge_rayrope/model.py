"""Additive integration with pinned EconomicGrasp main; no edits to old models.

Flags off retain main's metric depth and CVA grouping. MoGe replaces the depth
parameterization. RayRoPE replaces only the center-view-angle grouping module;
main's proposal, view, CDF, width, label matcher and final decoder are reused.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import image_rays,canonicalize_points,metric_depth_from_shape
from .attention import GraspRayRoPEGrouping

EMBEDS={'vits':384,'vitb':768,'vitl':1024}
FEATURES={'vits':64,'vitb':128,'vitl':256}
OUTS={'vits':[48,96,192,384],'vitb':[96,192,384,768],'vitl':[256,512,1024,1024]}
CPU_LABELS=('object_poses_list','grasp_points_list','view_graspness_list',
            'top_view_index_list','grasp_cdf_bins_list','grasp_widths_depth_list',
            'grasp_width_valids_depth_list')


def filter_empty_objects(batch):
    """Remove zero-point object entries in ALL aligned CPU annotation lists."""
    if 'grasp_points_list' not in batch: return dict(batch)
    out=dict(batch); lists={k:[] for k in CPU_LABELS}
    for k in CPU_LABELS:
        if k not in batch: raise KeyError(f'Missing compact CDF annotation list {k}')
    for b,points in enumerate(batch['grasp_points_list']):
        n=len(points)
        if any(len(batch[k][b])!=n for k in CPU_LABELS):
            raise ValueError('Misaligned object labels')
        keep=[i for i,p in enumerate(points) if p.shape[0]>0]
        if not keep: raise ValueError(f'Image {b} has no nonempty grasp annotations')
        for k in CPU_LABELS:
            values=[batch[k][b][i] for i in keep]
            if any(t.device.type!='cpu' for t in values):
                raise RuntimeError(f'{k} must stay on CPU until online matching')
            lists[k].append(values)
    out.update(lists); return out


class FactorizedMetricDepth(nn.Module):
    """MoGe-inspired affine pointmap + known-K gauge + learned metric anchor.

    Pointmap supervision is aligned; metric L1 does NOT rewrite local shape.
    Pose-conditioned DINO tokens predict one median-depth anchor per image.
    Unlike original MoGe we know K and predict metric grounding for robotics;
    we do not infer intrinsics, use MoGe weights, or claim their training scale.
    """
    def __init__(self,old,cfg):
        super().__init__()
        from models.dinov2_dpt import DPTHead
        self.cfg=cfg; self.depthnet=old.depthnet
        # Drop old scalar DPT parameters entirely rather than leave unused params.
        self.depthnet.depth_head=DPTHead(in_channels=EMBEDS[cfg.encoder],
            features=FEATURES[cfg.encoder],use_bn=False,out_channels=OUTS[cfg.encoder],
            out_dim=3,use_clstoken=True)
        # Keep shape-head gradients when its final hidden features are negative.
        self.depthnet.depth_head.scratch.output_conv2[1]=nn.LeakyReLU(0.01,inplace=False)
        self.pose_aware_adapter=old.pose_aware_adapter if hasattr(old,'pose_aware_adapter') else None
        self.pose_depth_mode=cfg.pose_mode
        self.freeze_backbone_flag=True; self.stride=old.stride
        self.min_depth=cfg.min_depth; self.max_depth=cfg.max_depth
        self.metric_anchor=nn.Sequential(nn.LayerNorm(EMBEDS[cfg.encoder]),
            nn.Linear(EMBEDS[cfg.encoder],128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.metric_anchor[-1].weight)
        initial=(.5-cfg.min_depth)/(cfg.max_depth-cfg.min_depth)
        nn.init.constant_(self.metric_anchor[-1].bias,math.log(initial/(1-initial)))
        self.last_geometry=None
        self.depthnet.pretrained.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode); self.depthnet.pretrained.eval(); return self

    def extract_backbone_features(self,img):
        with torch.no_grad():
            return self.depthnet.pretrained.get_intermediate_layers(img,
                self.depthnet.intermediate_layer_idx[self.depthnet.encoder],return_class_token=True)

    def forward(self,img,camera_pose_vec=None,camera_gravity_vec=None,camera_K=None,
                return_feats=False,return_raw=False,return_pose_aux=False):
        B,_,H,W=img.shape
        if (H,W)!=(448,448): raise ValueError('Pinned main requires 448x448 input')
        if camera_K is None: raise ValueError('MoGe factorization requires known crop-adjusted K')
        feats=self.extract_backbone_features(img)
        feature,raw=self.depthnet.depth_head(feats,H//14,W//14)
        hw=(H//self.cfg.shape_stride,W//self.cfg.shape_stride)
        raw_small=F.interpolate(raw.float(),hw,mode='bilinear',align_corners=True)
        rays=image_rays(camera_K.float(),hw,(H,W))
        # Stable ray-plane prior prevents a collapsed pointmap at cold start.
        points=rays+.1*raw_small
        canonical,shift,gauge=canonicalize_points(points,rays)
        metric_feats=feats; pose_aux={}
        if self.pose_depth_mode=='global_film':
            if camera_pose_vec is None: raise ValueError('global_film needs camera_pose_vec')
            metric_feats,pose_aux=self.pose_aware_adapter(feats,camera_pose_vec)
        tokens=metric_feats[-1][0] if isinstance(metric_feats[-1],(tuple,list)) else metric_feats[-1]
        anchor_raw=self.metric_anchor(tokens.mean(dim=1)).view(B,1,1,1)
        anchor=self.min_depth+(self.max_depth-self.min_depth)*anchor_raw.sigmoid()
        small=metric_depth_from_shape(canonical,anchor)
        depth=F.interpolate(small,(H,W),mode='bilinear',align_corners=True)
        tok=F.interpolate(depth,(H//self.stride,W//self.stride),mode='nearest') if self.stride>1 else depth
        feature=F.interpolate(feature,(H,W),mode='bilinear',align_corners=False)
        raw_equiv=torch.logit((depth/self.max_depth).clamp(1e-6,1-1e-6))
        self.last_geometry={'points':points,'canonical':canonical,'rays':rays,
                            'shift':shift,'gauge':gauge,'anchor':anchor,'depth':depth}
        outputs=[depth,tok,feature]
        if return_raw: outputs.append(raw_equiv)
        if return_feats: outputs.append(feats)
        if return_pose_aux: outputs.append(pose_aux)
        return tuple(outputs)


class EconomicGraspMoGeRayRoPE(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        from models.economicgrasp_bip3d import economicgrasp_dpt
        self.base=economicgrasp_dpt(encoder=cfg.encoder,tok_feat_dim=128,
            min_depth=cfg.min_depth,max_depth=cfg.max_depth,bin_num=256,
            freeze_backbone=True,is_training=True,use_obs_depth=False,use_depth_comp=False,
            use_cdf=True,pose_depth_mode=cfg.pose_mode,seed_selection_mode='image_fps',
            geometry_depth_source='pred',vis_dir=None)
        if cfg.use_moge: self.base.depth_net=FactorizedMetricDepth(self.base.depth_net,cfg)
        if cfg.use_rayrope:
            decoder_channels=self.base.kview_grasp_module.decoder.input_proj.in_channels
            self.base.kview_grasp_module.group=GraspRayRoPEGrouping(cfg,128,decoder_channels)
        self.sigma_head=None
        if cfg.use_rayrope and cfg.ray_encoding=='expected' and cfg.uncertainty=='learned':
            self.sigma_head=nn.Sequential(nn.Conv2d(FEATURES[cfg.encoder],32,3,padding=1),
                nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,1,1))
            nn.init.zeros_(self.sigma_head[-1].weight)
            p=(cfg.fixed_halfwidth-cfg.sigma_min)/(cfg.sigma_max-cfg.sigma_min)
            nn.init.constant_(self.sigma_head[-1].bias,math.log(p/(1-p)))
        self.depth_bias_m=0.  # inference-only numeric-geometry intervention

    def train(self,mode=True):
        super().train(mode)
        for m in self.base.modules():
            if hasattr(m,'is_training'): m.is_training=bool(mode)
        self.base.depth_net.depthnet.pretrained.eval()
        return self

    def geometry_parameters(self):
        result=[p for p in self.base.depth_net.parameters() if p.requires_grad]
        if self.sigma_head is not None: result+=list(self.sigma_head.parameters())
        return result

    def forward(self,batch):
        if self.training and self.depth_bias_m!=0:
            raise ValueError('Depth interventions are inference diagnostics, not training augmentation')
        batch=filter_empty_objects(batch)
        # Explicit label presence, not train/eval, controls matching at validation.
        batch['cva_force_process_grasp_labels']='object_poses_list' in batch
        batch['cva_compute_diagnostics']=False
        captured={}; handles=[]
        def depth_hook(module,args,output):
            out=list(output); live=out[0]
            if self.depth_bias_m:
                out[0]=live+self.depth_bias_m
                out[1]=F.interpolate(out[0],output[1].shape[-2:],mode='nearest')
            captured['depth']=live
            if self.cfg.use_moge: captured.update(module.last_geometry)
            if self.cfg.use_rayrope:
                if self.sigma_head is None:
                    sigma=torch.full_like(out[0],self.cfg.fixed_halfwidth)
                else:
                    feat=F.interpolate(out[2].detach(),scale_factor=.25,mode='bilinear',align_corners=True)
                    sigma=self.cfg.sigma_min+(self.cfg.sigma_max-self.cfg.sigma_min)*self.sigma_head(feat).sigmoid()
                    sigma=F.interpolate(sigma,out[0].shape[-2:],mode='bilinear',align_corners=True)
                captured['sigma']=sigma
                self.base.kview_grasp_module.group.geometry_context={
                    'sigma':sigma.detach(),'shape':captured.get('canonical')}
            return tuple(out)
        def cva_detach(module,args,kwargs):
            if args: raise RuntimeError('Expected keyword-based main CVA API')
            kw=dict(kwargs); kw['depth_map']=kw['depth_map'].detach()
            return args,kw
        # No depth hook in the true off/off baseline unless diagnostic bias requested.
        if self.cfg.use_moge or self.cfg.use_rayrope or self.depth_bias_m:
            handles.append(self.base.depth_net.register_forward_hook(depth_hook))
        handles.append(self.base.kview_grasp_module.register_forward_pre_hook(cva_detach,with_kwargs=True))
        try:
            ep=self.base(batch)
            if captured: ep['mr_geometry']=captured
            return ep
        finally:
            for handle in handles: handle.remove()
            if self.cfg.use_rayrope: self.base.kview_grasp_module.group.geometry_context=None
            if self.cfg.use_moge: self.base.depth_net.last_geometry=None
