"""Scene-memory wrapper around the strictly frozen Stage-1 runtime/provider."""
from __future__ import annotations
import importlib
import numpy as np
from .core import sample_scene,sha256


class SceneRuntime:
    def __init__(self, base, grid_size=32):
        self.base=base;self.grid_size=grid_size
        for name in ('model','snapshot','label','sensor_keep'):
            if not hasattr(base,name):raise TypeError(f'Historical provider must expose {name}')
        self.reader=base.model.kview_grasp_module.decoder.reader
        self.provenance={**base.provenance,'scene_source':'reader.context.pregeom_before_probe_sampling',
                         'scene_sampling':'uniform_cell_centers_align_corners_false',
                         'scene_grid':grid_size,'scene_geometry':'same_RGB_predicted_depth_and_camera_K',
                         'scene_scope':'full_model_input_crop_not_uncropped_camera_image'}
    def snapshot(self,row,seed=0):
        memories=[]
        def hook(module,args,kwargs):
            ctx=args[1] if len(args)>1 else kwargs.get('context')
            if ctx is None:raise ValueError('Missing reader context')
            if not memories:
                memories.append(sample_scene(ctx['pregeom'].detach(),ctx['depth'].detach(),
                                              ctx['K'].detach(),ctx['image_hw'],self.grid_size))
        h=self.reader.register_forward_pre_hook(hook,with_kwargs=True)
        try:a,p=self.base.snapshot(row,seed)
        finally:h.remove()
        if len(memories)!=1:raise RuntimeError('No native pregeometry scene map captured')
        for k in ('native_decode_max_abs','native_reader_replay_max_abs'):
            if k not in p or not np.isfinite(p[k]) or p[k]>1e-6:raise RuntimeError(f'Failed original parity: {k}')
        # No need to persist 1024*36 old probe tokens for this experiment.
        a={k:v for k,v in a.items() if k not in ('memory','memory_valid')}
        a.update(memories[0]);return a,p
    def label(self,*a,**kw):return self.base.label(*a,**kw)
    def sensor_keep(self,*a,**kw):return self.base.sensor_keep(*a,**kw)


def create(a):
    module,name=a.provider.split(':');m=importlib.import_module(module)
    base=getattr(m,name)(a.config,a.dataset_root,a.gntrans_root,a.device)
    if not getattr(m,'__file__',None):raise ValueError('Provider source must be auditable')
    base.provenance['query_scene_provider_sha256']=sha256(m.__file__)
    for k in ('base_sha256','selector_sha256','config_sha256'):
        if not base.provenance.get(k):raise ValueError('Missing source identity')
    return SceneRuntime(base,a.grid_size)
