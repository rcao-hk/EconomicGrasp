"""Strict main-compatible frozen runtime. Imported only for mining/inference.

No mutation of production model classes. A temporary forward hook captures
native score-branch features at the original full forward length (TF32-safe).
"""
from __future__ import annotations
import importlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from .core import sha256, digest, frame_key, frame_seed, physical_ids, success_bits


def read_bound_config(path: str) -> dict:
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if not config.get('checkpoint_sha256') or sha256(config['checkpoint']) != config['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA missing/mismatch; bind a verified mix10 or mix50 checkpoint')
    return config


def check_mix_protocol(ck: dict, train_fraction: float = 0.1,
                       expected_epoch: int = 20) -> None:
    """Validate immutable MIX provenance, not just the filename/model weights.

    Existing historical 50% e15 checkpoints may be used ONLY with explicit
    epoch=16 approval and a complete true MIX protocol in the checkpoint.
    Do not reconstruct, invent or patch metadata to make an old weight pass.
    """
    from .core import mix_frame_stride
    stride = mix_frame_stride(train_fraction)
    expected_per_domain = 100 * len(range(0, 256, stride))
    p = ck.get('gntrans_mix_protocol', {})
    if not isinstance(p, dict):
        raise ValueError('No structured gntrans_mix_protocol in checkpoint')
    for key in ('original_train_count', 'gntrans_train_count'):
        if p.get(key) != expected_per_domain:
            raise ValueError(
                f'{key} mismatch: checkpoint={p.get(key)} expected={expected_per_domain} '
                f'for train_fraction={train_fraction}')
    if (p.get('mixed_train_count') != expected_per_domain * 2 or
            p.get('train_fraction_each_domain') != float(train_fraction)):
        raise ValueError('Mixed checkpoint train-fraction or total frame count mismatch')
    if float(train_fraction) == 0.5:
        if (p.get('paired_scene_frame_sampling') is not True or
                not p.get('original_train_index_sha256') or
                not p.get('gntrans_train_index_sha256')):
            raise ValueError('Mix50 requires paired sampling and recorded original/GNTrans frame hashes')
        if p['original_train_index_sha256'] != p['gntrans_train_index_sha256']:
            raise ValueError('Mix50 Original/GNTrans frame indices differ')
    if not ck.get('use_cdf'):
        raise ValueError('CDF-only checkpoint required')
    if int(expected_epoch) not in ((20, 16) if float(train_fraction) == 0.5 else (20,)):
        raise ValueError('Unsupported checkpoint epoch; use fixed e19 or explicit historical 50% e15')
    if ck.get('epoch') != int(expected_epoch):
        raise ValueError(
            f'Checkpoint epoch={ck.get("epoch")} does not match explicitly selected '
            f'checkpoint epoch={expected_epoch}')


def check_mix10(ck: dict) -> None:
    """Backwards-compatible original 10% contract and unit test."""
    check_mix_protocol(ck, 0.1, 20)


def bind_config(template: str, checkpoint: str, output: str,
                allow_historical_e15: bool = False) -> None:
    from .core import write_json
    config = json.loads(Path(template).read_text(encoding='utf-8'))
    fraction = float(config.get('mix_training_fraction', 0.1))
    from .core import mix_frame_stride
    mix_frame_stride(fraction)
    if allow_historical_e15 and fraction != 0.5:
        raise ValueError('--historical-e15 is only supported for verified 50%+50% weights')
    epoch = 16 if allow_historical_e15 else 20
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    check_mix_protocol(ck, fraction, epoch)
    expected_gvar = config['backbone'] == 'point_gvar'
    design = ck.get('grasp_design', {})
    if not design.get('enable_point_rayrope') or bool(design.get('enable_gvar')) != expected_gvar:
        raise ValueError('Point/GVAR architecture mismatch')
    if ck.get('pose_depth_mode') != 'global_film':
        raise ValueError('This primary protocol requires global_film')
    config.update(checkpoint=str(Path(checkpoint).resolve()),
                  checkpoint_sha256=sha256(checkpoint),
                  expected_checkpoint_epoch=epoch,
                  provenance_status=('historical_e15_seen_monitored' if allow_historical_e15
                                     else 'fixed_e19'),
                  source_mix_protocol=ck['gntrans_mix_protocol'], source_grasp_design=design)
    write_json(output, config)


class Runtime:
    def __init__(self, config_path: str, dataset_root: str, gntrans_root: str, device: str):
        self.config = read_bound_config(config_path)
        self.config_digest = digest(self.config)
        self.device = torch.device(device)
        self.root, self.trans_root = dataset_root, gntrans_root
        # Global parser is loaded once in this isolated process, never shared
        # concurrently across different checkpoint configurations.
        if 'utils.arguments' in sys.modules:
            raise RuntimeError('Construct one isolated runtime per process before importing production modules')
        old = sys.argv[:]
        sys.argv = [old[0], '--dataset_root', dataset_root, *self.config['model_args']]
        try:
            from utils.arguments import cfgs
        finally:
            sys.argv = old
        self.cfg = cfgs
        if cfgs.use_obs_depth or cfgs.use_gt_depth or not cfgs.use_cdf or not cfgs.multi_modal:
            raise ValueError('Strict RGB-predicted-depth CDF network required')
        if cfgs.kview_mode != 'A1' or cfgs.use_top4_view_infer:
            raise ValueError('Only native A1 Top-1 view is supported')
        from models.grasp_designs import design_options, validate_checkpoint_design
        from models.economicgrasp_bip3d import economicgrasp_dpt, pred_decode_center_view_angle
        from utils.label_generation import batch_viewpoint_params_to_matrix
        self.decode, self.rotation = pred_decode_center_view_angle, batch_viewpoint_params_to_matrix
        self.model = economicgrasp_dpt(
            **design_options(cfgs), min_depth=cfgs.min_depth, max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num, is_training=False, use_obs_depth=False,
            pose_depth_mode=cfgs.pose_depth_mode, use_cdf=True, vis_dir=None,
            camera_pose_key=getattr(cfgs,'camera_pose_key','camera_pose_vec'),
            camera_gravity_key=getattr(cfgs,'camera_gravity_key','camera_gravity_vec'),
            pose_hidden_dim=getattr(cfgs,'pose_hidden_dim',64),
            ray_gravity_hidden_dim=getattr(cfgs,'ray_gravity_hidden_dim',64),
            ray_gravity_mid_dim=getattr(cfgs,'ray_gravity_mid_dim',32)).to(self.device)
        ck = torch.load(self.config['checkpoint'], map_location='cpu', weights_only=False)
        check_mix_protocol(
            ck, float(self.config.get('mix_training_fraction', 0.1)),
            int(self.config.get('expected_checkpoint_epoch', 20))
        )
        if ck['gntrans_mix_protocol'] != self.config.get('source_mix_protocol'):
            raise ValueError('Bound checkpoint training protocol changed since binding')
        validate_checkpoint_design(self.model, ck)
        if bool(ck['grasp_design'].get('enable_gvar')) != (self.config['backbone']=='point_gvar'):
            raise ValueError('Config backbone name does not match actual model')
        if cfgs.grasp_max_width != .1:
            raise ValueError('Primary protocol uses executed gripper width cap 0.10m')
        self.model.load_state_dict(ck['model_state_dict'], strict=True)
        self.model.eval().requires_grad_(False)
        self.datasets, self.indices, self.evaluators = {}, {}, {}
        self.dataset_source_hash = None

    def dataset(self, domain: str, split: str):
        key = domain, split
        if key in self.datasets: return self.datasets[key]
        from dataset.graspnet_dataset import GraspNetMultiDataset, GraspNetTransDataset, collate_fn
        self.collate = collate_fn
        args = dict(camera='realsense', split=split, num_points=self.cfg.num_point,
                    remove_outlier=True, augment=False, load_label=False)
        if domain == 'original':
            ds = GraspNetMultiDataset(self.root, **args)
        elif domain == 'gntrans':
            if not self.trans_root: raise ValueError('GNTRANS_RGB_ROOT is required')
            ds = GraspNetTransDataset(self.root, self.trans_root, **args, use_gt_depth=True,
                                      min_depth=self.cfg.min_depth, max_depth=self.cfg.max_depth,
                                      bin_num=self.cfg.bin_num, depth_strides=1)
        else: raise ValueError(domain)
        lookup = {}
        for i, (s, a) in enumerate(zip(ds.scenename, ds.frameid)):
            scene = int(str(s)[-4:]); k = (scene, int(a))
            if k in lookup: raise ValueError('Duplicate scene/frame dataset index')
            lookup[k] = i
        self.datasets[key], self.indices[key] = ds, lookup
        return ds

    def sample(self, row: dict, seed: int):
        ds = self.dataset(row['domain'], row['split'])
        ix = self.indices[(row['domain'], row['split'])][(row['scene'], row['anno'])]
        value = frame_seed(row, seed)
        random.seed(value); np.random.seed(value); torch.manual_seed(value)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
        raw = self.collate([ds[ix]])
        if any(isinstance(v, (list, tuple)) for v in raw.values()):
            raise ValueError('load_label=False must not produce training object lists')
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k,v in raw.items()}
        return batch, ix

    def snapshot(self, row: dict, seed: int) -> dict:
        batch, ix = self.sample(row, seed)
        captured = []
        hook = self.model.kview_grasp_module.decoder.score_dropout.register_forward_hook(
            lambda _m, _i, out: captured.append(out.detach().clone()))
        try:
            with torch.inference_mode():
                ep = self.model(dict(batch))
                native = self.decode(ep, use_cdf=True)[0].detach().cpu().numpy()
        finally:
            hook.remove()
        if len(captured) != 1: raise RuntimeError('Expected exactly one native score feature capture')
        pred, used = ep.get('depth_net_pred'), ep.get('depth_map_used_for_geometry')
        if pred is None or used is None or pred.shape != used.shape or not torch.allclose(pred, used, atol=1e-6, rtol=0):
            raise RuntimeError('Network geometry must equal RGB predicted depth, not GT/sensor')
        logits = ep['grasp_cdf_pred_angle_depth'][0].detach().float()
        T,Q,A,D = logits.shape
        if (T,A,D,Q) != (6,12,4,1024):
            raise ValueError(f'Primary full native lattice must be 6x1024x12x4; got {logits.shape}')
        prob = logits.sigmoid().mean(0)  # Q,A,D
        joint = prob.reshape(Q,-1).argmax(1); angles = torch.div(joint,D,rounding_mode='floor')
        q = torch.arange(Q, device=prob.device)
        baseline = prob[q,angles]  # Q,D
        if not torch.isfinite(baseline).all(): raise ValueError('Nonfinite scores')
        feature = captured[0].float()
        if self.config['backbone'] == 'point_gvar':
            if feature.shape != (Q*A*D,64): raise ValueError(f'GVAR score feature {feature.shape}')
            feature = feature.reshape(Q,A,D,64)[q,angles]
        else:
            if feature.shape != (1,64,Q*A): raise ValueError(f'Point score feature {feature.shape}')
            feature = feature[0].transpose(0,1).reshape(Q,A,64)[q,angles]
            feature = feature[:,None,:].expand(Q,D,64)
        widths = (1.2*ep['grasp_width_pred_angle_depth'][0].permute(1,2,0).float()/10)
        widths = widths.clamp(0,self.cfg.grasp_max_width)[q,angles]
        centers = ep['xyz_graspable'][0].detach().float()
        view = ep['grasp_top_view_xyz'][0].detach().float()
        rot = self.rotation(-view, angles.float()*(np.pi/A))
        bank = torch.zeros((Q,D,17), device=prob.device)
        bank[...,0]=baseline; bank[...,1]=widths; bank[...,2]=.02
        bank[...,3]=torch.arange(1,D+1,device=prob.device)[None]*.01
        bank[...,4:13]=rot.reshape(Q,1,9); bank[...,13:16]=centers[:,None,:]; bank[...,16]=-1
        g = bank.detach().cpu().numpy()
        winner = g[np.arange(Q),baseline.argmax(1).cpu().numpy()]
        if native.shape != winner.shape or not np.allclose(native,winner,atol=1e-6,rtol=0):
            raise RuntimeError('Native 17D decoder parity failed; do not relax tolerance')
        base_logit = torch.logit(baseline.clamp(1e-7,1-1e-7))
        if (base_logit.sigmoid()-baseline).abs().max().item()>1e-6:
            raise RuntimeError('Epoch-0 scorer parity failed')
        return dict(features=feature.cpu().numpy().copy(), base_logit=base_logit.cpu().numpy(),
                    grasps=g, angles=angles.cpu().numpy(), data_index=ix,
                    native_parity_max_abs=float(np.max(np.abs(native-winner))))

    def label(self, row: dict, grasps: np.ndarray) -> dict:
        from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
        split = row['split']
        if split not in self.evaluators:
            self.evaluators[split] = ExactGraspNetActionEvaluator(
                self.root, 'realsense', split=split, fc_mode='official', strict=True)
        ev=self.evaluators[split]
        if getattr(ev,'_rare_last_scene',None)!=row['scene']:
            ev.scene_cache.clear()  # bounded host memory over 100 training scenes
            ev._rare_last_scene=row['scene']
        result = ev.evaluate(row['scene'],row['anno'],grasps)
        bits = success_bits(result.friction,result.collision_or_empty)
        return dict(count=bits.sum(-1).astype(np.int64), bits=bits,
                    friction=result.friction, collision=result.pure_collision,
                    empty=result.empty)

    def sensor_collision_filter(self, row: dict, grasps: np.ndarray) -> np.ndarray:
        # Explicit ORIGINAL sensor cloud for both domains. GNTrans preprocessing
        # still uses its legacy virtual-depth crop; not pure appearance causality.
        from graspnetAPI import GraspGroup
        from utils.collision_detector import ModelFreeCollisionDetectorTorch
        ds = self.dataset('original',row['split'])
        ix = self.indices[('original',row['split'])][(row['scene'],row['anno'])]
        cloud,_ = ds.get_data(ix,return_raw_cloud=True)
        detector=ModelFreeCollisionDetectorTorch(cloud.reshape(-1,3),voxel_size=.01)
        gg=GraspGroup(grasps)
        collision=detector.detect(gg,approach_dist=.05,collision_thresh=.01)
        return grasps[~collision.detach().cpu().numpy()]
