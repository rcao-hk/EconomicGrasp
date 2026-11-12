"""Strict current-main frozen runtime for extracting full candidate context.

A historical runtime may implement the same snapshot/label/filter interface,
but cannot bypass metadata, state-dict, physical-identity or replay checks.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from .contracts import sha256,digest,atomic_json,frame_seed
from .capture import FrozenScalarSelector,snapshot


def bind(template, base_checkpoint, selector_checkpoint, output):
    path=Path(output)
    if path.exists():raise FileExistsError(path)
    cfg=json.loads(Path(template).read_text())
    base=torch.load(base_checkpoint,map_location='cpu',weights_only=False)
    rank=torch.load(selector_checkpoint,map_location='cpu',weights_only=False)
    design=base.get('grasp_design',{})
    if not(design.get('enable_gvar') and design.get('enable_point_rayrope')):
        raise ValueError('Requires trusted Point RayRoPE+GVAR checkpoint metadata')
    if not base.get('use_cdf') or base.get('pose_depth_mode')!='global_film':
        raise ValueError('Only CDF/global_film source supported')
    if base.get('epoch')!=cfg['expected_base_epoch']:
        raise ValueError('Declared source epoch mismatch; do not silently change weights')
    mix=base.get('gntrans_mix_protocol',{})
    for key in ('original_train_count','gntrans_train_count'):
        if mix.get(key)!=cfg['expected_training_frames_each_domain']:
            raise ValueError(f'Missing/mismatched trusted source training metadata: {key}')
    if rank.get('schema')!='rare-executed-grasp-v1':raise ValueError('Unsupported source scorer schema')
    if rank['protocol']['training']['objective']!=cfg['selector_objective']:
        raise ValueError('Primary reference must be declared MSE (point); RARE is separate crosscheck')
    base_sha=sha256(base_checkpoint)
    if rank['protocol']['base_config']['checkpoint_sha256']!=base_sha:
        raise ValueError('MSE/RARE sidecar belongs to a different frozen GVAR')
    if rank.get('epoch')!=rank['protocol']['training']['epochs']:
        raise ValueError('Use preregistered final-epoch source scorer')
    args=list(cfg['model_args'])
    for flag,value in (
        ('rayrope_group_chunk',design['rayrope']['group_chunk']),
        ('rayrope_checkpoint_chunks',design['rayrope']['checkpoint_chunks']),
        ('gvar_action_chunk',design['gvar']['action_chunk']),
        ('gvar_activation_checkpoint',design['gvar']['activation_checkpoint'])):
        key='--'+flag
        if args.count(key)!=1:raise ValueError(f'Expected exactly one {key} in template')
        args[args.index(key)+1]=str(int(value))
    cfg.update(model_args=args,base_checkpoint=str(Path(base_checkpoint).resolve()),
               base_sha256=base_sha,selector_checkpoint=str(Path(selector_checkpoint).resolve()),
               selector_sha256=sha256(selector_checkpoint),source_grasp_design=design,
               source_mix_protocol=mix)
    atomic_json(output,cfg)


class NativeRuntime:
    def __init__(self,config,dataset_root,gntrans_root,device='cuda:0'):
        self.config=json.loads(Path(config).read_text())
        self.device=torch.device(device);self.root=dataset_root;self.trans_root=gntrans_root
        cfg=self.config
        for p,h in (('base_checkpoint','base_sha256'),('selector_checkpoint','selector_sha256')):
            if not cfg.get(h) or sha256(cfg[p])!=cfg[h]:raise ValueError('Bound checkpoint identity mismatch')
        if 'utils.arguments' in sys.modules:raise RuntimeError('Isolate each frozen runtime in its own process')
        argv=sys.argv[:];sys.argv=[argv[0],'--dataset_root',dataset_root,*cfg['model_args']]
        try:
            from utils.arguments import cfgs
        finally:sys.argv=argv
        self.cfg=cfgs
        if cfgs.use_obs_depth or cfgs.use_gt_depth or not (cfgs.multi_modal and cfgs.use_cdf and cfgs.enable_gvar and cfgs.enable_point_rayrope):
            raise ValueError('Frozen RGB/predicted-depth Point+GVAR only')
        if cfgs.kview_mode!='A1' or cfgs.use_top4_view_infer or cfgs.grasp_max_width!=.1:
            raise ValueError('Native A1 view, 0.10m width cap required')
        from models.grasp_designs import design_options,validate_checkpoint_design
        from models.economicgrasp_bip3d import economicgrasp_dpt,pred_decode_center_view_angle
        from utils.label_generation import batch_viewpoint_params_to_matrix
        self.decode=pred_decode_center_view_angle;self.rotation=batch_viewpoint_params_to_matrix
        self.model=economicgrasp_dpt(
            **design_options(cfgs),min_depth=cfgs.min_depth,max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,is_training=False,use_obs_depth=False,use_cdf=True,vis_dir=None,
            pose_depth_mode=cfgs.pose_depth_mode,
            camera_pose_key=getattr(cfgs,'camera_pose_key','camera_pose_vec'),
            camera_gravity_key=getattr(cfgs,'camera_gravity_key','camera_gravity_vec'),
            pose_hidden_dim=getattr(cfgs,'pose_hidden_dim',64),
            ray_gravity_hidden_dim=getattr(cfgs,'ray_gravity_hidden_dim',64),
            ray_gravity_mid_dim=getattr(cfgs,'ray_gravity_mid_dim',32)).to(self.device)
        base=torch.load(cfg['base_checkpoint'],map_location='cpu',weights_only=False)
        if base.get('grasp_design')!=cfg['source_grasp_design'] or base.get('gntrans_mix_protocol')!=cfg['source_mix_protocol']:
            raise ValueError('Bound provenance differs from source checkpoint')
        validate_checkpoint_design(self.model,base)
        self.model.load_state_dict(base['model_state_dict'],strict=True)
        self.model.eval().requires_grad_(False)
        reader=self.model.kview_grasp_module.decoder.reader
        if reader.cfg.variant!='volume_rel':raise ValueError('Primary stage1 uses native volume_rel evidence')
        rank=torch.load(cfg['selector_checkpoint'],map_location='cpu',weights_only=False)
        if rank['protocol']['base_config']['checkpoint_sha256']!=cfg['base_sha256']:
            raise ValueError('Selector/base mismatch')
        self.selector=FrozenScalarSelector(hidden=int(rank['hidden'])).to(self.device)
        self.selector.load_state_dict(rank['model'],strict=True)
        self.selector.eval().requires_grad_(False)
        self.datasets={};self.lookups={};self.evaluators={}
        torch.backends.cuda.matmul.allow_tf32=bool(cfg['matmul_allow_tf32'])
        torch.backends.cudnn.allow_tf32=bool(cfg['cudnn_allow_tf32'])
        self.provenance={
            'base_sha256':cfg['base_sha256'],'selector_sha256':cfg['selector_sha256'],
            'config_sha256':sha256(config),'config':cfg,'camera':'realsense',
            'dataset_root':str(Path(dataset_root).resolve()),
            'gntrans_root':str(Path(gntrans_root).resolve()),
            'frozen_geometry_source':'RGB_predicted_depth',
            'sensor_prefilter':'Original_sensor_voxel.01_threshold.01_approach.05',
            'context_stage':'all_1024_pre_sensor_filter_queries',
            'runtime_class':'interactive_grasp.native.NativeRuntime'}

    def dataset(self,domain,split):
        key=domain,split
        if key in self.datasets:return self.datasets[key]
        from dataset.graspnet_dataset import GraspNetMultiDataset,GraspNetTransDataset,collate_fn
        self.collate=collate_fn
        args=dict(camera='realsense',split=split,num_points=self.cfg.num_point,
                  remove_outlier=True,augment=False,load_label=False)
        if domain=='original':ds=GraspNetMultiDataset(self.root,**args)
        elif domain=='gntrans':
            if not self.trans_root:raise ValueError('GNTrans root required')
            ds=GraspNetTransDataset(self.root,self.trans_root,**args,use_gt_depth=True,
                min_depth=self.cfg.min_depth,max_depth=self.cfg.max_depth,
                bin_num=self.cfg.bin_num,depth_strides=1)
        else:raise ValueError(domain)
        lookup={(int(str(s)[-4:]),int(a)):i for i,(s,a) in enumerate(zip(ds.scenename,ds.frameid))}
        if len(lookup)!=len(ds):raise ValueError('Duplicate dataset frame mapping')
        self.datasets[key]=ds;self.lookups[key]=lookup
        return ds

    def sample(self,row,seed):
        ds=self.dataset(row['domain'],row['split']);idx=self.lookups[(row['domain'],row['split'])][(row['scene'],row['anno'])]
        value=frame_seed(row,seed)
        random.seed(value);np.random.seed(value);torch.manual_seed(value)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
        batch=self.collate([ds[idx]])
        return {k:v.to(self.device) if torch.is_tensor(v) else v for k,v in batch.items()}

    def true_valid(self,ctx,ids,Q,A,D):
        from rayrope_gvar.reader import canonical_probes,project_probes,support_probe_depths
        reader=self.model.kview_grasp_module.decoder.reader
        parent=ids//D;dep=(ids%D+1).to(ctx['depth'].dtype)*.01
        probes,_=canonical_probes(support_probe_depths(dep,reader.cfg,'rgb'),reader.cfg)
        _,_,_,valid=project_probes(ctx['centers'].reshape(-1,3)[parent],
                    ctx['rotations'].reshape(-1,3,3)[parent],probes,ctx['K'][0],ctx['image_hw'])
        return valid

    def snapshot(self,row,seed=0):
        return snapshot(self.model,self.sample(row,seed),self.decode,self.rotation,
                        self.true_valid,self.selector)

    def sensor_keep(self,row,grasps):
        from graspnetAPI import GraspGroup
        from utils.collision_detector import ModelFreeCollisionDetectorTorch
        ds=self.dataset('original',row['split'])
        idx=self.lookups[('original',row['split'])][(row['scene'],row['anno'])]
        cloud,_=ds.get_data(idx,return_raw_cloud=True)
        detector=ModelFreeCollisionDetectorTorch(cloud.reshape(-1,3),voxel_size=.01)
        mask=detector.detect(GraspGroup(grasps),approach_dist=.05,collision_thresh=.01)
        return ~mask.detach().cpu().numpy().astype(bool)

    def label(self,row,grasps):
        from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
        split=row['split']
        if split not in self.evaluators:
            self.evaluators[split]=ExactGraspNetActionEvaluator(self.root,'realsense',split=split,fc_mode='official',strict=True)
        ev=self.evaluators[split]
        if getattr(ev,'_interactive_scene',None)!=row['scene']:
            ev.scene_cache.clear();ev._interactive_scene=row['scene']
        label=ev.evaluate(row['scene'],row['anno'],grasps)
        # Compare to canonical friction grid after rounding floating threshold
        # representation; official labels use these six discrete coefficients.
        f=np.round(np.asarray(label.friction,dtype=float),2)
        if not np.isfinite(f).all():raise ValueError('Nonfinite exact labels')
        bits=(f[:,None]>0)&(f[:,None]<=np.asarray([.2,.4,.6,.8,1.,1.2])[None])&(~np.asarray(label.collision_or_empty,dtype=bool)[:,None])
        return bits.sum(-1).astype(np.int64)
