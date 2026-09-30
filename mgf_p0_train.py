#!/usr/bin/env python3
"""P0-1: matched online scoring fine-tuning with a truly frozen MGF source."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from mgf_p0_core import VERSION, Metrics, assert_frozen, fingerprint, score_loss


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-checkpoint', required=True)
    p.add_argument('--dataset-root', default='/data/robotarm/dataset/graspnet')
    p.add_argument('--output-root', required=True)
    p.add_argument('--variant', choices=['feature_only','full','cva'], required=True)
    p.add_argument('--epochs', type=int, default=5, help='Additional fine-tuning epochs, not total pretraining')
    p.add_argument('--batch-size', type=int, default=3, help='Per GPU; no gradient accumulation')
    p.add_argument('--train-fraction', type=float, default=.2)
    p.add_argument('--eval-fraction', type=float, default=.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--ranking-weight', type=float, default=.1)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--eval-workers', type=int, default=1)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--max-train-frames', type=int, default=0)
    p.add_argument('--max-val-frames', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    return p


@torch.no_grad()
def validate(source, control, loader, device):
    from mgf_p0_online import safe_batch
    was = control.training
    control.eval()
    own, base = Metrics(), Metrics()
    for raw in loader:
        c = source(safe_batch(raw, device, True))
        e = c['ep']; y, m = e['batch_grasp_cdf_bins_angle_depth'], e['batch_grasp_cdf_valid_mask']
        own.update(control(c), y, m)
        base.update(e['grasp_cdf_pred_angle_depth'], y, m)
        del c
    # Validation must be collective-safe. Every rank evaluates a disjoint
    # validation shard, then pools sufficient statistics. The old rank0-only
    # validation left the other ranks waiting in a NCCL barrier for >600 s.
    own.synchronize(device)
    base.synchronize(device)
    control.train(was)
    return {'score': own.report(), 'base': base.report()}


def preflight(source, control, dataset, device):
    from dataset.graspnet_dataset import collate_fn
    from mgf_p0_online import safe_batch
    control.eval()
    for i in range(min(32,len(dataset))):
        ctx = source(safe_batch(collate_fn([dataset[i]]),device,True))
        e = ctx['ep']; y, v = e['batch_grasp_cdf_bins_angle_depth'], e['batch_grasp_cdf_valid_mask']
        logits = control(ctx)
        initial_diff = float((logits-e['grasp_cdf_pred_angle_depth']).abs().max())
        if initial_diff > 2e-5:
            raise RuntimeError(f'Initial control does not reconstruct source Base: {initial_diff}')
        if not bool(v.any()):
            continue
        loss, _ = score_loss(logits, y, v, distributed=False)
        loss.backward()
        assert_frozen(source.model)
        norm = math.sqrt(sum(float(p.grad.detach().square().sum()) for p in control.parameters() if p.grad is not None))
        control.zero_grad(set_to_none=True)
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError('No finite control gradient with valid labels')
        return {'sample_index': i, 'valid_candidates': int(v.sum()),
                'initial_base_max_abs_diff': initial_diff, 'control_gradient_norm': norm,
                'frozen_state_sha256': fingerprint(source.model),
                'trainable_names': [n for n,p in control.named_parameters() if p.requires_grad],
                'trainable_parameters': sum(p.numel() for p in control.parameters() if p.requires_grad)}
    raise RuntimeError('Preflight found no labeled query in first 32 samples')


def main():
    a = parser().parse_args()
    if min(a.epochs,a.batch_size,a.log_every) < 1 or min(a.workers,a.eval_workers,a.max_steps,a.max_train_frames,a.max_val_frames)<0:
        raise ValueError('Invalid count/worker setting')
    if any(not math.isfinite(x) for x in (a.lr,a.weight_decay,a.grad_clip,a.ranking_weight,a.temperature)) or min(a.lr,a.grad_clip,a.temperature)<=0 or min(a.weight_decay,a.ranking_weight)<0:
        raise ValueError('Invalid optimizer/loss settings')
    if not torch.cuda.is_available(): raise RuntimeError('Training requires the repository CUDA environment')
    world, rank, local = (int(os.getenv(k,d)) for k,d in [('WORLD_SIZE','1'),('RANK','0'),('LOCAL_RANK','0')])
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl')
    from mgf_p0_online import FrozenSource, make_control, safe_batch, code_digest
    from metric_field_runtime import (make_dataset, seed_all, digest, atomic_json, atomic_torch,
                                      rng_state, restore_rng, worker_init)
    from dataset.graspnet_dataset import collate_fn
    seed_all(a.seed)
    source = FrozenSource(a.source_checkpoint, device)
    control = make_control(source,a.variant)
    _, train, ti = make_dataset(a.dataset_root,'train',a.train_fraction,labels=True,
                                use_fuse_depth=source.protocol.get('use_fuse_depth',False),max_frames=a.max_train_frames)
    _, val, vi = make_dataset(a.dataset_root,'test_seen',a.eval_fraction,labels=True,
                              use_fuse_depth=source.protocol.get('use_fuse_depth',False),max_frames=a.max_val_frames)
    protocol = dict(version=VERSION, source_checkpoint=source.checkpoint,source_sha256=source.checkpoint_sha,
                    code_sha256=code_digest(),variant=a.variant, epochs=a.epochs, seed=a.seed,
                    train_fraction=a.train_fraction,eval_fraction=a.eval_fraction,
                    train_frames=len(ti),val_frames=len(vi),sampling_sha256=digest({'train':ti,'val':vi}),
                    world_size=world,batch_per_gpu=a.batch_size,effective_batch=world*a.batch_size,
                    lr=a.lr,weight_decay=a.weight_decay,grad_clip=a.grad_clip,lr_schedule='cosine_epoch',
                    ranking_weight=a.ranking_weight,temperature=a.temperature,
                    dataset_root=str(Path(a.dataset_root).resolve()),workers=a.workers,eval_workers=a.eval_workers,
                    max_steps=a.max_steps,partial_run=bool(a.max_steps or a.max_train_frames or a.max_val_frames),
                    prediction_source='frozen eval RGB+metadata; no sensor-depth network input',
                    labels='existing canonical CDF annotations, online matching; NOT exact predicted-action labels',
                    initialization='same source adapter/readout hidden weights; residual last layer zero; CVA private decoder copy',
                    validation_execution='disjoint strided validation shards on every DDP rank; pooled sufficient statistics; no padding duplicates',
                    selection='fixed additional budget; latest only; test_seen is monitoring, not model selection')
    signature=digest(protocol); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    if rank==0:
        import fcntl
        lock=open(out/'.train.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    latest=out/'checkpoint_latest.pt'; ck=None
    if a.resume:
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['signature']!=signature: raise RuntimeError('Resume protocol/source/code changed; use a new output root')
        control.load_state_dict(ck['control'],strict=True)
    elif (out/'protocol.json').exists() or latest.exists():
        raise FileExistsError('Output already used; --resume or a new directory required')
    frozen_before=fingerprint(source.model)
    if not a.resume:
        rng=rng_state(); report=preflight(source,control,train,device); restore_rng(rng)
        if report['frozen_state_sha256']!=frozen_before: raise RuntimeError('Preflight modified frozen state')
        if rank==0:
            atomic_json(out/'protocol.json',protocol); atomic_json(out/'gradient_contract.json',report)
    params=[p for p in control.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=a.weight_decay)
    start,history=0,[]
    if ck:
        opt.load_state_dict(ck['optimizer']); start=ck['epoch']+1; history=ck['history']
        if frozen_before != ck['frozen_state_sha256']: raise RuntimeError('Frozen source bytes changed')
    gen=torch.Generator().manual_seed(a.seed+rank)
    sampler=DistributedSampler(train,world,rank,shuffle=True,seed=a.seed) if world>1 else None
    loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,shuffle=sampler is None,
                      collate_fn=collate_fn,num_workers=a.workers,worker_init_fn=worker_init,
                      generator=gen,persistent_workers=False,pin_memory=False)
    # No DistributedSampler here: it pads when len(val) is not divisible by
    # world size, which would duplicate validation frames. A deterministic
    # strided Subset gives each frame to exactly one rank.
    val_rank = (
        Subset(val, list(range(rank, len(val), world)))
        if world > 1 else val
    )
    vl=DataLoader(val_rank,batch_size=a.batch_size,shuffle=False,collate_fn=collate_fn,
                  num_workers=a.eval_workers,worker_init_fn=worker_init,
                  persistent_workers=False,pin_memory=False)
    if ck:
        restore_rng(ck['rng'][rank]); gen.set_state(ck['loader_rng'][rank]); del ck
    wrapped=DDP(control,device_ids=None,broadcast_buffers=False,find_unused_parameters=False) if world>1 else control
    for epoch in range(start,a.epochs):
        t0=time.monotonic(); control.train(); assert_frozen(source.model)
        if sampler is not None: sampler.set_epoch(epoch)
        lr=a.lr*.5*(1+math.cos(math.pi*epoch/a.epochs))
        for group in opt.param_groups: group['lr']=lr
        metrics=Metrics(); steps=0
        for step,raw in enumerate(loader):
            if a.max_steps and step>=a.max_steps: break
            ctx=source(safe_batch(raw,device,True)); e=ctx['ep']
            y,m=e['batch_grasp_cdf_bins_angle_depth'],e['batch_grasp_cdf_valid_mask']
            opt.zero_grad(set_to_none=True)
            logits=wrapped(ctx)
            loss,diagnostic=score_loss(logits,y,m,a.ranking_weight,a.temperature)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite P0 scoring loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,a.grad_clip,error_if_nonfinite=True)
            assert_frozen(source.model); opt.step(); metrics.update(logits,y,m); steps+=1
            if rank==0 and (step+1)%a.log_every==0:
                print(f'[P0-1] {a.variant} epoch={epoch} step={step+1}/{len(loader)} lr={lr:.6g} '
                      f'loss={float(loss):.5f} valid={int(diagnostic["valid_candidates"])} '
                      f'rank_q={int(diagnostic["informative_queries"])}',flush=True)
            del ctx,logits,loss,e
        metrics.synchronize(device)
        # All ranks participate in validation and its metric all-reduces.
        # This both avoids the NCCL watchdog timeout and reduces wall time by
        # roughly world_size for the expensive frozen-source validation.
        v=validate(source,control,vl,device)
        current=fingerprint(source.model)
        if current!=frozen_before: raise RuntimeError('Frozen parameters/buffers changed during fine-tuning')
        rs,gs=[None]*world,[None]*world
        if world>1:
            dist.all_gather_object(rs,rng_state()); dist.all_gather_object(gs,gen.get_state())
        else: rs[0],gs[0]=rng_state(),gen.get_state()
        if rank==0:
            row=dict(epoch=epoch,optimizer_steps=steps,lr=lr,train=metrics.report(),validation=v,
                     frozen_state_unchanged=True,seconds=time.monotonic()-t0)
            history.append(row)
            atomic_json(out/'metrics.json',history)
            atomic_torch(latest,dict(version=VERSION,signature=signature,protocol=protocol,epoch=epoch,
                          control=control.state_dict(),optimizer=opt.state_dict(),history=history,
                          frozen_state_sha256=frozen_before,rng=rs,loader_rng=gs))
            print('[P0-1 EPOCH] '+json.dumps(row),flush=True)


if __name__=='__main__':
    try: main()
    finally:
        if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()
