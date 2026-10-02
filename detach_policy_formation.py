"""Bounded experiment using production Trainer/loss/AdamW; no alternate train loop."""
import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

import checkerboard_probe as q
from detach_policy_gate import set_cfg, resolved, string_keys


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda'])


class PairedDataset(Dataset):
    """Sampling randomness is keyed to occurrence, independent of worker prefetch."""
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,key):
        epoch,rank,pos,index=key
        seed=int.from_bytes(hashlib.sha256(f'0/{epoch}/{rank}/{pos}/{index}'.encode()).digest()[:4],'little')
        r=random.getstate(),np.random.get_state(),torch.get_rng_state()
        try:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            s=self.base[index]
        finally:
            random.setstate(r[0]);np.random.set_state(r[1]);torch.set_rng_state(r[2])
        s['_policy_input_identity']=np.array([epoch,rank,pos,index,seed],dtype=np.int64)
        return s


class Occurrences(Sampler):
    def __init__(self,sampler,epoch,rank,start_batch):
        sampler.set_epoch(epoch)
        self.keys=[(epoch,rank,p,i) for p,i in enumerate(sampler)][start_batch*3:]
    def __iter__(self):return iter(self.keys)
    def __len__(self):return len(self.keys)


def prepare(a):
    out=a.output;prod=q.imports(out);c=prod.cfgs;q.seed(0)
    assert c.max_epoch==21 and c.batch_size==3
    assert json.loads((out/'p1_acceptance.json').read_text())['status']=='pass'
    target=out/'canonical_step0.pt'
    if target.exists():raise RuntimeError('Canonical exists: do not recreate the common initialization')
    set_cfg(c,'A');c.checkpoint_path=None;c.resume=False;prod.CHECKPOINT_PATH=None
    m=q.make_model(prod)
    dummy=object.__new__(prod.Trainer);dummy.net=m;dummy.main=False
    optimizer=dummy.build_optimizer();assert not optimizer.state
    state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    frozen=[n for n,p in m.named_parameters() if not p.requires_grad]
    assert frozen and all(n.startswith('depth_net.') for n in frozen)
    torch.save(dict(model_state_dict=state,optimizer_state_dict=optimizer.state_dict(),rng=rng_state(),
                    frozen_keys=frozen,seed=0,optimizer_updates=0),target)
    canonical=dict(path=str(target),sha256=q.sha(target),model_hash=q.tree_hash(state),
                   frozen_parameters=frozen,trainable_parameters=[n for n,p in m.named_parameters() if p.requires_grad],
                   seed=0,optimizer_history=0,constructor='production default frozen pretrained DINO, random remaining modules',
                   checkpoint_override=None)
    pretrained=Path('checkpoints/depth_anything_v2_vitb.pth').resolve()
    canonical['pretrained_source']=dict(path=str(pretrained),sha256=q.sha(pretrained),
        loading_filter='only keys containing pretrained; DPT/FiLM/proposal/task heads retain constructor initialization')
    q.dump(out/'canonical_manifest.json',canonical)
    del m,optimizer,state
    ds=q.bases(prod);assert len(ds['train'])==25600 and len(ds['test_seen'])==7680
    old=json.loads((out/'probe_manifest.json').read_text())['frames']
    seen=[r['index'] for r in old if r['split']=='test_seen']
    # Predeclared deterministic uniform indices; no output has been inspected.
    for i in np.linspace(0,7679,32,dtype=int).tolist():
        if i not in seen:seen.append(i)
        if len(seen)==32:break
    frames=[dict(split='train',index=r['index']) for r in old if r['split']=='train']
    frames += [dict(split='test_seen',index=i) for i in sorted(seen)]
    manifest=[]
    for fr in frames:
        i=fr['index'];fr.update(scene=f"scene_{i//256+(100 if fr['split']=='test_seen' else 0):04d}",frame=i%256)
        s,inst,meta=q.get_sample(ds[fr['split']],fr,out)
        gt=np.asarray(s['gt_depth_m']).squeeze();fg=(inst>0)&np.isfinite(gt)&(gt>=.2)&(gt<=1.)
        vu=np.argwhere(fg)
        uv=vu[np.linspace(0,len(vu)-1,min(1024,len(vu)),dtype=int)][:,::-1] if len(vu) else np.empty((0,2),int)
        np.save(out/'inputs'/(meta['stem']+'_uv.npy'),uv)
        meta.update(common_uv_sha256=q.tree_hash(uv),common_uv_count=len(uv),continuity=any(r['split']==fr['split'] and r['index']==i for r in old))
        manifest.append(meta)
    q.dump(out/'development_manifest.json',dict(seed=0,selection='old four seen plus uniform integer linspace candidates until32; fixed before formation',
        frames=manifest,AP_scenes=[100,110,119,129],AP_frames=list(range(256)),similar_novel_not_used=True))
    contract=json.loads((out/'previous_run_contract.json').read_text())
    contract.update(run_id=out.name,stage='P4_prepared',seed=0,canonical=canonical,
        source_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        matcher_sha256=q.sha(Path('utils/label_generation.py')),arms={'A_all_open':[0,0,0],'B_detach_E':[1,0,0],'C_detach_all':[1,1,1]},
        P4_updates_per_arm=17070,P4_epochs=6,cosine_total_epochs=21,original_updates_per_arm=59745,
        loader_semantics='DistributedSampler original seed0, shuffle and padding; occurrence-keyed RNG independent of prefetch; sample payload hashes logged per rank',
        resume_claim='Input stream reconstructible; numerical restart equivalence must be measured, not claimed bitwise',
        initial_state_source='constructor once; no Stage1 or trained DPT weights')
    q.dump(out/'run_contract.json',contract)
    print(json.dumps(dict(stage='prepared',canonical_sha256=canonical['sha256'],frames=len(manifest))),flush=True)


class StopAtBudget(Exception):pass


def train(a):
    out=a.output;prod=q.imports(out);c=prod.cfgs
    assert json.loads((out/'p1_acceptance.json').read_text())['status']=='pass'
    set_cfg(c,a.arm);c.log_dir=str(a.logs/a.name);c.vis_dir=None;c.resume=False;c.checkpoint_path=None;c.seed=0
    prod.CHECKPOINT_PATH=None
    trainer=prod.Trainer();m=trainer.unwrap_model();rank=trainer.rank
    assert trainer.world_size==3 and c.batch_size==3 and c.max_epoch==21
    assert len(trainer.TRAIN_DATASET)==25600 and len(trainer.train_sampler)==8534 and len(trainer.TRAIN_DATALOADER)==2845
    canonical_path=out/'canonical_step0.pt';manifest=json.loads((out/'canonical_manifest.json').read_text())
    assert q.sha(canonical_path)==manifest['sha256']
    initial=torch.load(canonical_path,map_location='cpu',weights_only=False)
    m.load_state_dict(initial['model_state_dict'],strict=True);trainer.optimizer.load_state_dict(initial['optimizer_state_dict'])
    frozen=set(initial['frozen_keys']);del initial
    q.seed(rank)
    position=dict(step=0,epoch=0,batch=0)
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['canonical_sha256']==manifest['sha256'] and state['arm']==a.arm
        missing=m.load_state_dict(state['model_delta'],strict=False)
        assert set(missing.missing_keys)==frozen and not missing.unexpected_keys
        trainer.optimizer.load_state_dict(state['optimizer_state_dict'])
        position.update(state['position']);restore_rng(state['rank_rng'][rank]);del state
    start_step=position['step'];armout=out/a.name;armout.mkdir(exist_ok=True)
    logroot=a.logs/a.name;logroot.mkdir(parents=True,exist_ok=True)
    record=(armout/f'train_rank{rank}.jsonl').open('a',buffering=1)
    dev=json.loads((out/'development_manifest.json').read_text())['frames']
    started=time.time();input_record={};global_step=position['step'];epoch_start_batch=0

    def checkpoint(label):
        states=[None]*3;dist.all_gather_object(states,rng_state())
        if trainer.main:
            path=logroot/(label+'.pt')
            if shutil.disk_usage(logroot).free<1_500_000_000:raise RuntimeError('Less than1.5GB free: checkpoint storage needs attention')
            state=dict(version=1,canonical_path=str(canonical_path),canonical_sha256=manifest['sha256'],
                arm=a.arm,resolved=resolved(m),position=position.copy(),rank_rng=states,
                sampler_epoch=position['epoch'],loader_next_batch=position['batch'],
                scheduler=dict(kind='epoch_cosine',max_epoch=21,next_epoch=position['epoch'],base_lr=c.learning_rate),
                model_delta={n:v.detach().cpu() for n,v in m.state_dict().items() if n not in frozen},
                optimizer_state_dict=trainer.optimizer.state_dict(),source_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
            torch.save(state,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path)
            digest=q.sha(path)
            with (armout/'checkpoint_manifest.jsonl').open('a') as f:f.write(json.dumps(dict(path=str(path),sha256=digest,bytes=path.stat().st_size,position=position.copy()))+'\n')
            event_path=armout/'formation_event.json'
            if event_path.exists():
                event=json.loads(event_path.read_text())
                if global_step>event['step'] and 'post_state' not in event:
                    keep=logroot/f'event_post_{global_step:06d}.pt'
                    if not keep.exists():os.link(path,keep)
                    event['post_state']=dict(path=str(keep),sha256=digest)
                    q.dump(event_path,event)
            # Own rolling states only; epoch states and event evidence are retained.
            rolling=sorted(logroot.glob('rolling_*.pt'))
            for prior in rolling[:-6]:
                with (armout/'retired_rolling.jsonl').open('a') as f:f.write(json.dumps(dict(path=str(prior),sha256=q.sha(prior),reason='six newer complete rolling states retained'))+'\n')
                prior.unlink()
        dist.barrier()

    def probes(full=False):
        if trainer.main:
            rows=[]
            with q.Preserve(m),torch.no_grad():
                m.eval()
                for fr in dev:
                    if not full and not fr['continuity']:continue
                    s=torch.load(out/'inputs'/(fr['stem']+'.pt'),weights_only=False)
                    data=np.load(out/'inputs'/(fr['stem']+'.npz'))
                    pred=q.depth_forward(m,s)[0][0,0].cpu().numpy().copy()
                    metrics=q.metrics(pred,data['gt'],data['instance'],fr['roi'])
                    uv=np.load(out/'inputs'/(fr['stem']+'_uv.npy'))
                    metrics['common_uv_depth_mae_mm']=float(abs(pred[uv[:,1],uv[:,0]]-data['gt'][uv[:,1],uv[:,0]]).mean()*1000) if len(uv) else None
                    fg=(data['instance']>0)&np.isfinite(data['gt'])&(data['gt']>=.2)&(data['gt']<=1.)
                    metrics.update(pred_fg_std_mm=float(pred[fg].std()*1000) if fg.any() else None,
                        pred_fg_min_m=float(pred[fg].min()) if fg.any() else None,pred_fg_max_m=float(pred[fg].max()) if fg.any() else None)
                    rows.append(dict(arm=a.arm,seed=0,global_step=global_step,epoch=position['epoch'],**fr,**metrics))
                    if full and fr['continuity']:
                        (armout/'predictions').mkdir(exist_ok=True)
                        np.savez_compressed(armout/'predictions'/f"step{global_step:06d}_{fr['stem']}.npz",pred=pred)
            with (armout/'formation_curves.jsonl').open('a') as f:
                for r in rows:f.write(json.dumps(r)+'\n')
            seen=[r.get('A_B_mm') for r in rows if r['split']=='test_seen' and r['continuity']]
            if seen and all(v is not None and np.isfinite(v) for v in seen):
                value=float(np.mean(seen));history_path=armout/'formation_event_monitor.json'
                history=json.loads(history_path.read_text()) if history_path.exists() else []
                if not history or history[-1]['step']!=global_step:
                    threshold=max(2.,3*history[0]['A_B_mm']) if history else max(2.,3*value)
                    high=value>threshold;history.append(dict(step=global_step,A_B_mm=value,threshold_mm=threshold,high=high))
                    q.dump(history_path,history)
                    event_path=armout/'formation_event.json'
                    if len(history)>1 and high and history[-2]['high'] and not event_path.exists():
                        retained=[]
                        for prior in sorted(logroot.glob('rolling_*.pt'))[-3:]:
                            keep=logroot/('event_'+prior.name)
                            if not keep.exists():os.link(prior,keep)
                            retained.append(dict(path=str(keep),sha256=q.sha(keep)))
                        q.dump(event_path,dict(step=global_step,rule='two consecutive continuity seen mean A_B > max(2mm,3xstep0); diagnostic interval only',
                            before_and_current=retained,step0=str(logroot/'step0.pt')))
        dist.barrier()

    def before(epoch,batch_index,b):
        nonlocal input_record
        effective=batch_index+epoch_start_batch
        # Dense depth distributions are not consumed by the regression model/loss.
        used={k:v for k,v in b.items() if k not in ('depth_prob_gt','depth_prob_weight')}
        input_record=dict(epoch=epoch,batch=effective,input_hash=q.tree_hash(used),
            identity=b['_policy_input_identity'].tolist(),rng_hash=q.tree_hash(string_keys(rng_state())))

    def after(epoch,batch_index,ep,norm):
        nonlocal global_step
        global_step+=1;next_batch=batch_index+epoch_start_batch+1
        position.update(step=global_step,epoch=epoch+(next_batch==2845),batch=0 if next_batch==2845 else next_batch)
        losses=trainer.extract_scalar_metrics(ep)
        forward={k:v for k,v in ep.items() if torch.is_tensor(v) and
            (k.startswith('batch_') or k in ('depth_net_pred','depth_head_raw_pred','view_score','grasp_cdf_pred_angle_depth','grasp_width_pred_angle_depth','token_sel_idx','xyz_graspable'))}
        gradient_groups={}
        if global_step%20==1:
            for n,p in m.named_parameters():
                if p.grad is not None:
                    group='FiLM' if '.pose_aware_adapter.' in n else 'DPT' if n.startswith('depth_net.') else n.split('.')[0]
                    gradient_groups[group]=gradient_groups.get(group,0.)+float(p.grad.detach().double().square().sum())
        record.write(json.dumps(dict(arm=a.arm,rank=rank,global_step=global_step,**input_record,
            global_grad_norm=float(norm),clip=min(1.,1./(float(norm)+1e-6)),lr=trainer.optimizer.param_groups[0]['lr'],
            forward_hash=q.tree_hash(forward),postclip_gradient_squared_norm=gradient_groups,
            losses=losses,coverage=q.loss_coverage(ep),elapsed_sec=time.time()-started))+'\n')
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient; do not continue')
        if global_step%250==0:
            checkpoint(f'rolling_{global_step:06d}');probes()
        if a.stop_step and global_step>=a.stop_step:
            checkpoint(f'stop_{global_step:06d}');raise StopAtBudget()
    trainer.on_batch_start=before;trainer.on_optimizer_step=after
    try:
        if not a.resume:
            assert q.tree_hash(m.state_dict())==manifest['model_hash']
            checkpoint('step0');probes(full=True)
        for epoch in range(position['epoch'],a.stop_epoch):
            epoch_start_batch=position['batch'] if epoch==position['epoch'] else 0
            sampler=Occurrences(trainer.train_sampler,epoch,rank,epoch_start_batch)
            generator=torch.Generator().manual_seed(1701+rank+epoch*3)
            trainer.TRAIN_DATALOADER=DataLoader(PairedDataset(trainer.TRAIN_DATASET),batch_size=3,
                sampler=sampler,num_workers=c.num_workers,collate_fn=prod.collate_fn,drop_last=False,
                pin_memory=False,persistent_workers=c.num_workers>0,generator=generator)
            assert len(trainer.TRAIN_DATALOADER)==2845-epoch_start_batch
            trainer.train_one_epoch(epoch)
            assert global_step==(epoch+1)*2845
            checkpoint(f'epoch_{epoch:02d}');probes(full=True)
        if trainer.main:q.dump(armout/'status.json',dict(status='complete',step=global_step,epoch=position['epoch'],stop_epoch=a.stop_epoch))
    except StopAtBudget:
        if trainer.main:q.dump(armout/'status.json',dict(status='bounded_stop',step=global_step,position=position))
    finally:
        record.close();trainer.close();prod.cleanup_distributed(trainer.distributed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--logs',type=Path)
    p.add_argument('--arm',choices=['A','B','C']);p.add_argument('--name')
    p.add_argument('--resume',type=Path);p.add_argument('--stop-epoch',type=int,default=6);p.add_argument('--stop-step',type=int)
    a=p.parse_args();prepare(a) if a.stage=='prepare' else train(a)
