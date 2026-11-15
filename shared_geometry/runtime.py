"""Read-only native adapter and separately loaded offline visible-depth targets."""
from __future__ import annotations
import importlib
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from interactive_grasp.contracts import sha256, digest
from .geometry import align_depth, check_camera


class FieldRuntime:
    def __init__(self, base):
        self.base = base
        for name in ('model', 'snapshot', 'label', 'sensor_keep', 'provenance'):
            if not hasattr(base, name):
                raise TypeError(f'Historical provider must expose {name}')
        self.reader = base.model.kview_grasp_module.decoder.reader
        self.provenance = dict(base.provenance, field_source='full_native_pregeom_and_predicted_depth',
                               camera_frame='native_grasp_camera', field_runtime_sha256=sha256(__file__))

    def snapshot(self, row, seed=0):
        evidence = []
        def capture(_module, args, kwargs):
            ctx = args[1] if len(args) > 1 else kwargs.get('context')
            if ctx is None:
                raise ValueError('Missing native reader context')
            if not evidence:
                if ctx['pregeom'].shape[0] != 1 or ctx['depth'].shape[:2] != (1, 1):
                    raise ValueError('Single-frame native provider required')
                if tuple(ctx['depth'].shape[-2:]) != tuple(ctx['image_hw']):
                    raise ValueError('Depth and crop geometry disagree')
                evidence.append(dict(feature_map=ctx['pregeom'][0].detach().float().cpu().numpy().copy(),
                                     pred_depth=ctx['depth'][0, 0].detach().float().cpu().numpy().copy(),
                                     K=ctx['K'][0].detach().float().cpu().numpy().copy()))
        hook = self.reader.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            arrays, parity = self.base.snapshot(row, seed)
        finally:
            hook.remove()
        for name in ('native_decode_max_abs', 'native_reader_replay_max_abs'):
            if name not in parity or not np.isfinite(parity[name]) or parity[name] > 1e-6:
                raise RuntimeError('Frozen source parity failed: '+name)
        if len(evidence) != 1 or len(arrays['grasps']) != 1024:
            raise ValueError('Expected full 1024-query scene')
        keep = {k: arrays[k] for k in ('grasps', 'features', 'base_logit')}
        keep.update(evidence[0])
        for k, value in keep.items():
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite native '+k)
        check_camera(keep['K'], keep['pred_depth'].shape)
        return keep, parity

    def label(self, *args):
        return self.base.label(*args)

    def sensor_keep(self, *args):
        return self.base.sensor_keep(*args)


def create_runtime(args):
    module, cls = args.provider.split(':')
    mod = importlib.import_module(module)
    base = getattr(mod, cls)(args.config, args.dataset_root, args.gntrans_root, args.device)
    if not getattr(mod, '__file__', None):
        raise ValueError('Provider source file is required for provenance')
    base.provenance['field_provider_sha256'] = sha256(mod.__file__)
    for key in ('base_sha256', 'selector_sha256', 'config_sha256'):
        if not base.provenance.get(key):
            raise ValueError('Missing source identity: '+key)
    return FieldRuntime(base)


class TargetLoader:
    """Explicit rendered-depth source, no path guessing or hidden depth substitution.

    Config describes a SAME-camera raw depth image and raw intrinsics. A custom
    historical renderer must supply this contract; missing files fail closed.
    Target is used in export only, never in test inference.
    """
    def __init__(self, path):
        self.cfg = json.loads(Path(path).read_text())
        self.config_sha256 = sha256(path)
        if self.cfg.get('source') != 'rendered_visible':
            raise ValueError('Primary geometry teacher must be declared rendered_visible')
        if self.cfg.get('same_camera_as_native') is not True:
            raise ValueError('Explicit same-camera-frame assertion required')
        if not self.cfg.get('calibration_description') or 'REPLACE' in self.cfg['calibration_description']:
            raise ValueError('Document renderer camera/extrinsics and crop convention')
        if float(self.cfg.get('depth_scale_to_m', 0)) <= 0:
            raise ValueError('Explicit raw-unit-to-metre conversion required')

    def load(self, row, K, hw):
        cfg = self.cfg
        p = Path(cfg['depth_template'].format(**row))
        kp = Path(cfg['intrinsics_template'].format(**row))
        if not p.is_file() or not kp.is_file():
            raise FileNotFoundError(f'Clean visible geometry missing: {p}, {kp}')
        if p.suffix == '.npy':
            depth = np.load(p, allow_pickle=False)
        elif p.suffix == '.npz':
            with np.load(p, allow_pickle=False) as z:
                depth = z[cfg.get('depth_key', 'depth')].copy()
        else:
            from PIL import Image
            depth = np.asarray(Image.open(p))
        if kp.suffix == '.mat':
            from scipy.io import loadmat
            raw_K = loadmat(kp)[cfg.get('intrinsics_key', 'intrinsic_matrix')]
        elif kp.suffix == '.npz':
            with np.load(kp, allow_pickle=False) as z:
                raw_K = z[cfg.get('intrinsics_key', 'K')].copy()
        else:
            raw_K = np.load(kp, allow_pickle=False)
        depth = np.asarray(depth, np.float32)*float(cfg['depth_scale_to_m'])
        if depth.ndim != 2 or depth.max(initial=0) > float(cfg.get('max_depth_m', 10)):
            raise ValueError('Visible-depth units/shape failed sanity check')
        aligned = align_depth(depth, raw_K, K, hw)
        identity=hashlib.sha256(depth.astype('<f4').tobytes()+np.asarray(raw_K,dtype='<f4').tobytes()).hexdigest()
        return aligned, dict(source='rendered_visible', config_sha256=self.config_sha256, raw_geometry_identity=identity,
                             depth_path=str(p.resolve()), depth_sha256=sha256(p),
                             intrinsics_path=str(kp.resolve()), intrinsics_sha256=sha256(kp),
                             raw_K=np.asarray(raw_K).tolist(), target_K=np.asarray(K).tolist(),
                             alignment='same-camera K_raw @ inv(K_crop), nearest depth')
