"""P1 checks on copies only. Production Trainer owns the DDP update."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
import checkerboard_probe as q


def resolved(m):
    return [m.detach_depth_gse, m.detach_depth_seed_xyz, m.detach_depth_support]


def string_keys(v):
    if isinstance(v,dict):return {str(k):string_keys(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [string_keys(x) for x in v]
    return v


def set_cfg(c, arm):
    values = {'A': (0, 0, 0, 0), 'B': (0, 1, 0, 0), 'C': (1, 1, 1, 1)}[arm]
    for name, value in zip(('detach_depth', 'detach_depth_gse', 'detach_depth_seed_xyz', 'detach_depth_support'), values):
        setattr(c, name, value)


def compatibility(prod, out):
    c = prod.cfgs
    cases = [(True, None, None, [1, 1, 1]), (True, 0, None, [0, 0, 0]),
             (True, 1, None, [1, 1, 1]), (True, 1, [0, 1, 0], [0, 1, 0]),
             (True, 0, [1, 0, 0], [1, 0, 0]), (False, 0, [0, 0, 0], [1, 1, 1])]
    rr = []
    for cdf, master, overrides, expected in cases:
        kwargs = dict(use_cdf=cdf, vis_dir=None, pose_depth_mode=c.pose_depth_mode)
        if master is not None:
            kwargs['detach_depth'] = master
        if overrides is not None:
            kwargs.update(zip(('detach_depth_gse', 'detach_depth_seed_xyz', 'detach_depth_support'), overrides))
        m = prod.economicgrasp_dpt(**kwargs)
        actual = resolved(m)
        assert actual == [bool(v) for v in expected], (kwargs, actual)
        assert m.spatial_enhancer.detach_depth_grad == actual[0]
        assert m.kview_grasp_module.config.detach_depth == actual[2]
        rr.append(dict(arguments=kwargs, resolved=actual))
        del m
        gc.collect()
    q.dump(out/'constructor_compatibility.json', dict(status='pass', cases=rr))


def forward_gate(a):
    out=a.output;prod=q.imports(out);q.seed()
    compatibility(prod, out)
    m=q.make_model(prod);ds=q.bases(prod)
    frames=json.loads((out/'probe_manifest.json').read_text())['frames']
    selected=json.loads((out/'selected_checkpoints.json').read_text())
    rr=[];heads=[]
    for cp in (selected[0],selected[-1]):
        state=torch.load(cp['path'],map_location='cpu',weights_only=False)
        m.load_state_dict(state.get('model_state_dict',state),strict=True);del state
        state_hash=q.tree_hash(m.state_dict())
        for split in ('train','test_seen'):
            cpu=q.batch_for(prod,ds[split],[r for r in frames if r['split']==split][:2])
            for mode in ('train','eval'):
                baseline=None
                for route in ('all','none','noE','noQ','noC'):
                    grids=[];original=F.grid_sample
                    def capture(x,grid,*args,**kwargs):
                        grids.append(grid.detach().cpu().clone())
                        return original(x,grid,*args,**kwargs)
                    with q.Preserve(m):
                        m.train(mode=='train');q.route(m,route);q.seed(1901)
                        b=prod.move_batch_to_device(copy.deepcopy(cpu),torch.device('cuda'),True,False)
                        b.update(cva_compute_diagnostics=True,geometry_compute_diagnostics=False,cva_export_angle_feature=False)
                        with patch.object(F,'grid_sample',capture):
                            ep=m(b)
                        loss,ep=prod.get_loss_economicgrasp(ep,use_cdf=True)
                        # All discrete tensors/masks plus predictions, seed geometry and every actual grid.
                        values={k:v.detach().cpu().clone() for k,v in ep.items() if torch.is_tensor(v) and
                                (k.startswith('batch_') or any(s in k for s in ('pred','score','xyz','idx','inds','mask')))}
                        values.update({'grid_'+str(i):g for i,g in enumerate(grids)})
                        assert grids, 'No actual sampling grids captured'
                        if baseline is None:baseline=values
                        assert values.keys()==baseline.keys()
                        diffs={k:float((v.double()-baseline[k].double()).abs().max()) if v.numel() else 0. for k,v in values.items()}
                        bad={k:d for k,d in diffs.items() if d != 0}
                        rr.append(dict(checkpoint=cp['path'],split=split,mode=mode,route=route,
                                       grid_calls=len(grids),exact=not bad,differences=bad,checked_keys=list(values)))
                        q.dump(out/'forward_grid_gate.json',dict(status='running',records=rr))
                        assert not bad, rr[-1]
                        if cp==selected[-1] and split=='train' and mode=='train' and route in ('all','noE','none'):
                            own=[(n,p) for n,p in m.named_parameters() if p.requires_grad and not n.startswith('depth_net.')]
                            for lname,key in [('view','B: View Loss'),('cdf','B: CDF Loss'),('width','B: Width Loss')]:
                                gs=torch.autograd.grad(ep[key],[p for n,p in own],retain_graph=True,allow_unused=True)
                                groups={}
                                for (n,p),g in zip(own,gs):
                                    if g is not None:
                                        prefix='.'.join(n.split('.')[:2])
                                        groups[prefix]=groups.get(prefix,0.)+float(g.double().square().sum())
                                assert sum(groups.values())>0, (route,lname)
                                heads.append(dict(route=route,loss=lname,squared_norm_by_module=groups))
                        del ep,loss,b,values,grids
                    print(json.dumps(rr[-1] | {'checked_keys':len(rr[-1]['checked_keys'])}),flush=True)
        assert q.tree_hash(m.state_dict())==state_hash
    q.dump(out/'forward_grid_gate.json',dict(status='pass',records=rr,state_restored=True))
    q.dump(out/'head_gradient_gate.json',dict(status='pass',records=heads))


def ddp_step(a):
    out=a.output;prod=q.imports(out);c=prod.cfgs
    selected=json.loads((out/'selected_checkpoints.json').read_text())
    cp=selected[0]  # Actual Adam history exists here; target epoch16 is weights-only.
    c.log_dir=str(out/'ddp_step'/a.arm);c.vis_dir=None;c.seed=0
    c.checkpoint_path=cp['path'];c.resume=True;prod.CHECKPOINT_PATH=cp['path']
    set_cfg(c,a.arm)
    trainer=prod.Trainer();m=trainer.unwrap_model()
    assert trainer.world_size==3 and c.batch_size==3
    assert len(trainer.TRAIN_DATASET)==25600 and len(trainer.TRAIN_DATALOADER)==2845
    optimizer_before=q.tree_hash(string_keys(trainer.optimizer.state_dict()))
    model_before=q.tree_hash(m.state_dict())
    epoch=trainer.start_epoch
    trainer.train_sampler.set_epoch(epoch)
    indices=list(trainer.train_sampler)[:3]
    batch=next(iter(trainer.TRAIN_DATALOADER))
    batch_hash=q.tree_hash(batch)
    # The production transfer/model mutates its input dict in place. Preserve
    # the original CPU batch for before/after probes and the identity audit.
    trainer.TRAIN_DATALOADER=[copy.deepcopy(batch)]
    before={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
    records=[];clip=torch.nn.utils.clip_grad_norm_
    def record_clip(parameters,max_norm,**kw):
        norm=clip(parameters,max_norm,**kw)
        records.append(dict(global_grad_norm=float(norm),clip_coefficient=min(1.,max_norm/(float(norm)+1e-6))))
        return norm
    # Diagnostic depth-only probe is isolated from model/RNG and DDP collectives.
    def probe():
        with q.Preserve(m),torch.no_grad():
            m.eval()
            b=prod.move_batch_to_device(copy.deepcopy(batch),trainer.device,True,False)
            ep=m(b)
            return ep['depth_net_pred'].detach().cpu().clone()
    z0=probe()
    with patch.object(torch.nn.utils,'clip_grad_norm_',record_clip):
        trainer.train_one_epoch(epoch)
    z1=probe()
    import cv2
    from PIL import Image
    probe_metrics=[]
    for i,index in enumerate(indices):
        scene=int(batch['scene_idx'][i]);frame=int(batch['anno_idx'][i])
        path=Path(c.dataset_root)/'scenes'/f'scene_{scene:04d}'/c.camera/'label'/f'{frame:04d}.png'
        x0,y0,x1,y1=batch['crop_box'][i].tolist()
        inst=np.asarray(Image.open(path))[y0:y1,x0:x1].astype(np.int32)
        inst=cv2.resize(inst,(448,448),interpolation=cv2.INTER_NEAREST)
        gt=batch['gt_depth_m'][i].squeeze().numpy()
        roi=q.rectangle(np.isfinite(gt)&(gt>=.2)&(gt<=1.))
        probe_metrics.append(dict(index=index,scene=scene,frame=frame,
            before=q.metrics(z0[i,0].numpy(),gt,inst,roi),after=q.metrics(z1[i,0].numpy(),gt,inst,roi)))
    norms={}
    for n,p in m.named_parameters():
        if n in before:
            module='DPT_FiLM' if n.startswith('depth_net.') else 'grasp'
            norms[module]=norms.get(module,0.)+float((p.detach().cpu().double()-before[n].double()).square().sum())
    # Params must agree across ranks; buffers need not (production broadcast_buffers=False).
    after_hash=q.tree_hash(dict(m.named_parameters()))
    hashes=[None]*3;torch.distributed.all_gather_object(hashes,after_hash)
    assert len(set(hashes))==1, hashes
    assert len(records)==1
    result=dict(arm=a.arm,rank=trainer.rank,checkpoint=cp,epoch=epoch,indices=indices,
                input_hash=batch_hash,model_before=model_before,optimizer_before=optimizer_before,
                resolved=resolved(m),update_squared_norm=norms,parameters_after=after_hash,
                ranks_parameters_equal=True,depth_change_mean_abs_mm=float((z1-z0).abs().mean()*1000),
                lr=trainer.optimizer.param_groups[0]['lr'],probe_metrics=probe_metrics,**records[0])
    (out/'ddp_step').mkdir(exist_ok=True)
    torch.save(dict(before=z0,after=z1,indices=indices),out/'ddp_step'/f'{a.arm}_rank{trainer.rank}_depth.pt')
    q.dump(out/'ddp_step'/f'{a.arm}_rank{trainer.rank}.json',result)
    print(json.dumps(result),flush=True)
    trainer.close();prod.cleanup_distributed(trainer.distributed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['forward','ddp'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['A','B','C'])
    a=p.parse_args()
    forward_gate(a) if a.stage=='forward' else ddp_step(a)
