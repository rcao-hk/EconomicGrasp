"""Depth-only models. Never import the production grasp graph or evaluator."""
from __future__ import annotations
import importlib
import sys
import types
from pathlib import Path
import torch
from torch import nn
from .geometry import pixel_grid, z_from_height
from .data import digest, sha256

ARMS = {
    'C0': (False, False, False, False),  # historical global FiLM, supervised two crops
    'C1': (True,  False, False, False),  # + identity-initialized ray/gravity FiLM
    'C2': (False, True,  False, False),
    'C3': (True,  True,  False, False),
    'T0': (True,  False, False, False),  # reset scalar head; zero plane-input capacity control
    'T1': (True,  False, True,  False),  # depth + calibrated plane
    'T2': (True,  False, True,  True),   # height + SAME calibrated plane
}


def production_modules():
    # Relative imports work, but models/__init__.py (MinkowskiEngine etc.) is not executed.
    name = '_camera_geometry_source_models'
    if name not in sys.modules:
        p = types.ModuleType(name)
        p.__path__ = [str(Path(__file__).resolve().parents[1]/'models')]
        sys.modules[name] = p
    return (importlib.import_module(name+'.dinov2_dpt'),
            importlib.import_module(name+'.pose_aware_depth'))


def tensor_fingerprint(named_tensors):
    import hashlib
    h = hashlib.sha256()
    for name, t in sorted(named_tensors):
        t = t.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class PlaneFiLM(nn.Module):
    """Identical capacity in T0/T1/T2. T0 receives zeros, not plane metadata."""
    def __init__(self, dim, levels=4):
        super().__init__(); self.dim, self.levels = dim, levels
        self.net = nn.Sequential(nn.Linear(4,32), nn.GELU(), nn.Linear(32,levels*2*dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats, plane):
        gb = self.net(plane).reshape(-1,self.levels,2,self.dim)
        out = []
        for i, (p,c) in enumerate(feats):
            gamma, beta = gb[:,i,0], gb[:,i,1]
            out.append((p*(1+gamma[:,None])+beta[:,None], c*(1+gamma)+beta))
        return out


class MetricModel(nn.Module):
    def __init__(self, cfg, arm, initialize=True):
        super().__init__()
        if arm not in ARMS: raise ValueError('Unknown arm: '+arm)
        self.cfg, self.arm = cfg, arm
        self.use_ray, self.use_consistency, self.use_plane, self.height_output = ARMS[arm]
        self.table_study = arm.startswith('T')
        self.tiny = cfg.get('backend','production') == 'tiny'
        self.load_audit = {}
        if self.tiny:
            # ONLY for CPU interface/smoke tests, never a production experimental result.
            self.backbone = nn.Conv2d(3,8,3,padding=1)
            self.depth_head = nn.Sequential(nn.Conv2d(8,8,3,padding=1),nn.GELU(),nn.Conv2d(8,1,1))
            self.global_adapter = PlaneFiLM(8,1)
            self.ray_adapter = nn.Conv2d(6,16,1) if self.use_ray else None
            if self.ray_adapter is not None:
                nn.init.zeros_(self.ray_adapter.weight); nn.init.zeros_(self.ray_adapter.bias)
            self.plane_adapter = PlaneFiLM(8,1) if self.table_study else None
            self.backbone.requires_grad_(False); self.global_adapter.requires_grad_(False)
            last = self.depth_head[-1]
        else:
            dpt, pose = production_modules()
            ec = {'vits':(64,[48,96,192,384],384), 'vitb':(128,[96,192,384,768],768),
                  'vitl':(256,[256,512,1024,1024],1024)}
            width, channels, dim = ec[cfg['encoder']]
            self.depthnet = dpt.DA2(encoder=cfg['encoder'], features=width,
                out_channels=channels, max_depth=cfg['max_depth_m'], out_dim=1)
            self.global_adapter = pose.PoseAwareDPTFiLM(feat_dim=dim, num_levels=4,
                pose_dim=3, hidden_dim=cfg.get('pose_hidden_dim',64))
            if initialize: self.load_historical(cfg)
            self.depthnet.pretrained.requires_grad_(False)
            self.global_adapter.requires_grad_(False)
            self.ray_adapter = (pose.RayGravityDenseDPTFiLM(feat_dim=dim, num_levels=4,
                geometry_hidden_dim=64, geometry_mid_dim=32) if self.use_ray else None)
            self.plane_adapter = PlaneFiLM(dim) if self.table_study else None
            last = self.depthnet.depth_head.scratch.output_conv2[2]
        if self.table_study:
            # All three table arms reset the SAME scalar layer; do not interpret warm z logits as h.
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.train(False)

    def load_historical(self, cfg):
        path = cfg['checkpoint']
        if not Path(path).is_file(): raise FileNotFoundError(path)
        ck = torch.load(path, map_location='cpu', weights_only=False)  # trusted user checkpoint only
        state = ck[cfg['checkpoint_key']] if cfg.get('checkpoint_key') else ck
        prefix = cfg['checkpoint_prefix']
        selected = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        core = {k[len('depthnet.'):]:v for k,v in selected.items() if k.startswith('depthnet.')}
        glob = {k[len('pose_aware_adapter.'):]:v for k,v in selected.items() if k.startswith('pose_aware_adapter.')}
        unknown = set(selected)-{'depthnet.'+k for k in core}-{'pose_aware_adapter.'+k for k in glob}
        if not core or not glob or unknown:
            raise ValueError(f'Expected exact historical global-FiLM depth subtree; unknown={sorted(unknown)}')
        self.depthnet.load_state_dict(core, strict=True)
        self.global_adapter.load_state_dict(glob, strict=True)
        self.load_audit = {'checkpoint_sha256':sha256(path), 'prefix':prefix,
                           'core_keys':len(core), 'global_keys':len(glob), 'strict':True}

    def frozen_fingerprint(self):
        module = self.backbone if self.tiny else self.depthnet.pretrained
        values = [('backbone.'+k,v) for k,v in module.state_dict().items()]
        values += [('global.'+k,v) for k,v in self.global_adapter.state_dict().items()]
        return tensor_fingerprint(values)

    def train(self, mode=True):
        super().train(mode)
        (self.backbone if self.tiny else self.depthnet.pretrained).eval()
        self.global_adapter.eval()
        return self

    def forward(self, rgb, K, pose, gravity, plane=None):
        B, _, H, W = rgb.shape
        uv = pixel_grid((H,W), device=rgb.device).reshape(1,-1,2).expand(B,-1,-1)
        if self.table_study and plane is None: raise ValueError('Every table arm needs calibrated plane for shared eligibility')
        if self.tiny:
            from .geometry import rays
            with torch.no_grad(): x = self.backbone(rgb)
            if self.ray_adapter is not None:
                r = rays(uv,K).transpose(1,2).reshape(B,3,H,W)
                r = r/r.norm(dim=1,keepdim=True)
                gamma,beta = self.ray_adapter(torch.cat((r,gravity[:,:,None,None].expand_as(r)),1)).chunk(2,1)
                x = x*(1+gamma)+beta
            if self.plane_adapter is not None:
                p = plane if self.use_plane else torch.zeros_like(plane)
                f = [(x.flatten(2).transpose(1,2),x.mean((-2,-1)))]
                x = self.plane_adapter(f,p)[0][0].transpose(1,2).reshape_as(x)
            raw = self.depth_head(x)[:,0]
        else:
            if (H,W) != (448,448): raise ValueError('Production experiment uses 448x448 crops')
            with torch.no_grad():
                feats = self.depthnet.pretrained.get_intermediate_layers(rgb,
                    self.depthnet.intermediate_layer_idx[self.cfg['encoder']], return_class_token=True)
                feats, _ = self.global_adapter(feats, pose)
            if self.ray_adapter is not None:
                feats, _ = self.ray_adapter(feats, camera_k=K, camera_gravity_vec=gravity,
                                            image_hw=(H,W), patch_hw=(H//14,W//14))
            if self.plane_adapter is not None:
                feats = self.plane_adapter(feats, plane if self.use_plane else torch.zeros_like(plane))
            _, logits = self.depthnet.depth_head(feats,H//14,W//14)
            raw = logits[:,0]
        h = self.cfg.get('height_min_m',-.05)+(self.cfg.get('height_max_m',.35)-self.cfg.get('height_min_m',-.05))*raw.sigmoid()
        if self.table_study:
            decoded, eligible = z_from_height(h.flatten(1),uv,K,plane,self.cfg.get('grazing',.1))
            eligible = eligible.reshape(B,H,W)
        else:
            eligible = torch.ones_like(raw,dtype=torch.bool)
        z = decoded.reshape(B,H,W) if self.height_output else raw.sigmoid()*self.cfg['max_depth_m']
        return {'z':z, 'height':h if self.height_output else None, 'eligible':eligible}
