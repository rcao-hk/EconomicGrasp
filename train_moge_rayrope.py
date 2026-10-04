#!/usr/bin/env python3
"""Online 20% GraspNet training: main baseline / MoGe / grasp-frame RayRoPE.

No feature/action-label mining or gradient accumulation. All task/geometry
modules train online; DAV2 encoder and numeric-geometry-to-grasp paths stay
frozen/detached. Existing main compact CDF annotations are required.
"""
from __future__ import annotations
import argparse,json,math,os,sys,time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,Subset
from moge_rayrope.config import ModelConfig,LossConfig,VERSION,MAIN_COMMIT


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',default='/data/robotarm/dataset/graspnet')
    p.add_argument('--label-folder',default='economic_grasp_label_300views_extend_angle_cdf_depth')
    p.add_argument('--output-root',required=True)
    p.add_argument('--encoder',choices=['vits','vitb','vitl'],default='vitb')
    p.add_argument('--use-moge',type=int,choices=[0,1],default=0)
    p.add_argument('--use-rayrope',type=int,choices=[0,1],default=0)
    p.add_argument('--ray-encoding',choices=['none','point','expected'],default='expected')
    p.add_argument('--uncertainty',choices=['fixed','learned'],default='fixed')
    p.add_argument('--ray-apply-vo',type=int,choices=[0,1],default=1)
    p.add_argument('--shape-tokens',type=int,choices=[0,1],default=0)
    p.add_argument('--pose-mode',choices=['none','global_film'],default='global_film')
    p.add_argument('--fixed-halfwidth',type=float,default=.02)
    p.add_argument('--ray-radius-px',type=float,default=40.)
    p.add_argument('--ray-grid',type=int,default=7)
    p.add_argument('--group-chunk',type=int,default=64)
    p.add_argument('--seeds',type=int,default=1024)
    p.add_argument('--shape-global-weight',type=float,default=1.)
    p.add_argument('--shape-local-weight',type=float,default=.5)
    p.add_argument('--reprojection-weight',type=float,default=.05)
    p.add_argument('--interval-weight',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=3)
    p.add_argument('--train-fraction',type=float,default=.2)
    p.add_argument('--eval-fraction',type=float,default=.1)
    p.add_argument('--use-fuse-depth',type=int,choices=[0,1],default=1)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--weight-decay',type=float,default=1e-3)
    p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--eval-workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--max-train-frames',type=int,default=0)
    p.add_argument('--max-val-frames',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    return p


def gradient_contract(model,dataset,device,mc,lc):
    from moge_rayrope.runtime import move_batch,rng_state,restore_rng
    from moge_rayrope.objective import objective
    from dataset.graspnet_dataset import collate_fn
    state=rng_state(); buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    report=None
    try:
        model.train()
        for idx in range(min(len(dataset),16)):
            ep=model(move_batch(collate_fn([dataset[idx]]),device,labels=True))
            if not bool(ep['batch_grasp_cdf_valid_mask'].any()): continue
            total,task,_=objective(ep,mc,lc)
            geom=model.geometry_parameters()
            grads=torch.autograd.grad(task,geom,retain_graph=True,allow_unused=True)
            leak=max([float(g.abs().max()) for g in grads if g is not None]+[0.])
            if leak>1e-10: raise RuntimeError(f'Grasp loss leaks into geometry: {leak}')
            geo_grads=torch.autograd.grad(total-task,geom,retain_graph=True,allow_unused=True)
            geo_norm=math.sqrt(sum(float(g.double().square().sum()) for g in geo_grads if g is not None))
            if not math.isfinite(geo_norm) or geo_norm<=0: raise RuntimeError('No finite geometry gradient')
            total.backward()
            branch_grads={}
            for label,prefix in (('shape_decoder','base.depth_net.depthnet.depth_head.'),
                                 ('metric_anchor','base.depth_net.metric_anchor.'),
                                 ('sigma_head','sigma_head.'),
                                 ('ray_group','base.kview_grasp_module.group.')):
                branch_grads[label]=math.sqrt(sum(float(p.grad.double().square().sum())
                    for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None))
            if mc.use_moge and (branch_grads['shape_decoder']<=0 or branch_grads['metric_anchor']<=0):
                raise RuntimeError(f'MoGe shape/metric branch is not being trained: {branch_grads}')
            if mc.uncertainty=='learned' and branch_grads['sigma_head']<=0:
                raise RuntimeError('Learned interval head has no supervision gradient')
            task_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.depth_net.') and not n.startswith('sigma_head.')]
            task_norm=math.sqrt(sum(float(p.grad.double().square().sum()) for p in task_params if p.grad is not None))
            if task_norm<=0 or not math.isfinite(task_norm): raise RuntimeError('No finite task gradient')
            if any(p.grad is not None for p in model.base.depth_net.depthnet.pretrained.parameters()):
                raise RuntimeError('Frozen DAV2 encoder accumulated gradients')
            report=dict(sample_index=idx,grasp_to_geometry_max_abs=leak,geometry_grad_norm=geo_norm,
                        task_grad_norm=task_norm,branch_gradient_norms=branch_grads,dav2_frozen=True,
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            break
        if report is None: raise RuntimeError('No valid CDF example found for gradient preflight')
        return report
    finally:
        model.zero_grad(set_to_none=True)
        for n,b in model.named_buffers(): b.copy_(buffers[n])
        restore_rng(state)


@torch.no_grad()
def validate(model,loader,device,mc,lc):
    from moge_rayrope.runtime import move_batch,rng_state,restore_rng,seed_all
    from moge_rayrope.objective import objective
    from moge_rayrope.metrics import EpochMetrics
    state=rng_state(); was=model.training; model.eval(); seed_all(137)
    stat=EpochMetrics()
    try:
        for raw in loader:
            ep=model(move_batch(raw,device,labels=True)); _,_,logs=objective(ep,mc,lc)
            stat.update(ep,logs)
        stat.synchronize(device)
        try:
            return stat.report()
        except FloatingPointError:
            failure_dir=os.getenv('MR_DIAGNOSTIC_FAILURE_DIR')
            if failure_dir and int(os.getenv('RANK','0'))==0:
                torch.save(dict(diagnostic_only=True,model=model.state_dict(),
                    model_config=asdict(mc),loss_config=asdict(lc),loss_sums=stat.loss_sums,
                    metrics_vector=stat.v),Path(failure_dir)/'failure_model.pt')
            raise
    finally: model.train(was); restore_rng(state)


def main():
    a=parser().parse_args(); sys.argv=[sys.argv[0]]
    if min(a.epochs,a.batch_size,a.log_every)<1 or min(a.workers,a.eval_workers,a.max_steps,a.max_train_frames,a.max_val_frames)<0:
        raise ValueError('Invalid counts')
    if not all(math.isfinite(v) for v in (a.lr,a.weight_decay,a.grad_clip)) or min(a.lr,a.grad_clip)<=0 or a.weight_decay<0:
        raise ValueError('Invalid optimizer')
    mc=ModelConfig(encoder=a.encoder,use_moge=bool(a.use_moge),use_rayrope=bool(a.use_rayrope),
        ray_encoding=a.ray_encoding,uncertainty=a.uncertainty,ray_apply_vo=bool(a.ray_apply_vo),
        use_shape_tokens=bool(a.shape_tokens),pose_mode=a.pose_mode,fixed_halfwidth=a.fixed_halfwidth,
        ray_radius_px=a.ray_radius_px,ray_grid=a.ray_grid,group_chunk=a.group_chunk,seeds=a.seeds)
    lc=LossConfig(global_shape=a.shape_global_weight,local_shape=a.shape_local_weight,
                  reprojection=a.reprojection_weight,interval=a.interval_weight)
    if mc.use_moge and lc.global_shape<=0: raise ValueError('MoGe pointmap requires global shape supervision')
    if mc.uncertainty=='learned' and lc.interval<=0: raise ValueError('Learned uncertainty requires interval supervision')
    if not torch.cuda.is_available(): raise RuntimeError('Use the CUDA/GraspNet server for integration training')
    rank=int(os.getenv('RANK','0')); world=int(os.getenv('WORLD_SIZE','1')); local=int(os.getenv('LOCAL_RANK','0'))
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(seconds=3600))
    from moge_rayrope.runtime import (verify_main_contract,configure_main,make_dataset,move_batch,ROOT,
         seed_all,worker_init,rng_state,restore_rng,sha256_file,code_fingerprint,digest,atomic_json,atomic_torch)
    from moge_rayrope.model import EconomicGraspMoGeRayRoPE
    from moge_rayrope.objective import objective
    from moge_rayrope.metrics import EpochMetrics
    source_contract=verify_main_contract()
    main_options=configure_main(mc,a.dataset_root,a.label_folder,bool(a.use_fuse_depth))
    from dataset.graspnet_dataset import collate_fn
    seed_all(a.seed)
    model=EconomicGraspMoGeRayRoPE(mc).to(device)
    _,train,_,ts=make_dataset(a.dataset_root,'train',a.train_fraction,True,mc,a.label_folder,bool(a.use_fuse_depth),a.max_train_frames)
    _,val,_,vs=make_dataset(a.dataset_root,'test_seen',a.eval_fraction,True,mc,a.label_folder,bool(a.use_fuse_depth),a.max_val_frames)
    protocol=dict(version=VERSION,main_commit=MAIN_COMMIT,main_files=source_contract,model=asdict(mc),loss=asdict(lc),
        main_options=main_options,dataset_root=str(Path(a.dataset_root).resolve()),label_folder=a.label_folder,
        train_fraction=a.train_fraction,eval_fraction=a.eval_fraction,train_frames=len(train),val_frames=len(val),
        sampling_sha256=digest({'train':ts,'val':vs}),use_fuse_depth=bool(a.use_fuse_depth),
        epochs=a.epochs,seed=a.seed,batch_per_gpu=a.batch_size,world_size=world,effective_batch=a.batch_size*world,
        optimizer='AdamW',torch_version=str(torch.__version__),cuda_version=torch.version.cuda,lr=a.lr,weight_decay=a.weight_decay,grad_clip=a.grad_clip,lr_schedule='cosine_epoch',
        workers=a.workers,eval_workers=a.eval_workers,max_steps=a.max_steps,
        partial_run=bool(a.max_steps or a.max_train_frames or a.max_val_frames),
        initialization='fresh task/geometry modules; frozen pretrained DAV2 encoder; no MoGe pretrained weights',
        annotation_contract='main compact CDF dataset annotations matched online; no feature/action/teacher cache',
        depth_contract='grasp gradients blocked from metric depth, shape pointmap and interval head',
        validation='all DDP ranks, disjoint strided subsets, pooled metrics; latest fixed-budget checkpoint',
        code_sha256=code_fingerprint(),dav2_sha256=sha256_file(ROOT/'checkpoints'/f'depth_anything_v2_{mc.encoder}.pth'))
    signature=digest(protocol); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    lock=None
    if rank==0:
        import fcntl
        lock=open(out/'.train.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    latest=out/'checkpoint_latest.pt'; ck=None; start=0; history=[]
    if a.resume:
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['signature']!=signature: raise RuntimeError('Resume configuration/source/code changed')
        model.load_state_dict(ck['model'],strict=True); start=ck['epoch']+1; history=ck['history']
    else:
        # Only rank 0 checks creation; otherwise another rank may observe rank 0's
        # newly written protocol while still completing its own initialization.
        occupied=[bool((out/'protocol.json').exists() or latest.exists()) if rank==0 else None]
        if world>1: dist.broadcast_object_list(occupied,src=0)
        if occupied[0]:
            raise FileExistsError('Output already used: use --resume or a fresh output root')
        contract=gradient_contract(model,train,device,mc,lc)
        if rank==0:
            atomic_json(out/'protocol.json',protocol); atomic_json(out/'gradient_contract.json',contract)
            atomic_json(out/'sampling.json',{'train':ts,'validation':vs})
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=a.weight_decay)
    if ck: opt.load_state_dict(ck['optimizer'])
    sampler=DistributedSampler(train,world,rank,shuffle=True,seed=a.seed) if world>1 else None
    gen=torch.Generator().manual_seed(a.seed+rank)
    loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,shuffle=sampler is None,
        num_workers=a.workers,collate_fn=collate_fn,generator=gen,worker_init_fn=worker_init,
        persistent_workers=False,pin_memory=False)
    vsub=Subset(val,list(range(rank,len(val),world))) if world>1 else val
    vloader=DataLoader(vsub,batch_size=a.batch_size,num_workers=a.eval_workers,collate_fn=collate_fn,
        shuffle=False,worker_init_fn=worker_init,persistent_workers=False,pin_memory=False)
    if ck:
        restore_rng(ck['rng'][rank]); gen.set_state(ck['loader_rng'][rank]); del ck
    else: seed_all(a.seed+rank)
    # device_ids=None keeps variable-size canonical labels on CPU.
    wrapped=DDP(model,device_ids=None,broadcast_buffers=False,find_unused_parameters=True) if world>1 else model
    for epoch in range(start,a.epochs):
        torch.cuda.reset_peak_memory_stats(device)
        t0=time.monotonic(); model.train(); stat=EpochMetrics()
        if sampler is not None: sampler.set_epoch(epoch)
        lr=a.lr*.5*(1+math.cos(math.pi*epoch/a.epochs))
        for g in opt.param_groups: g['lr']=lr
        steps=0
        for step,raw in enumerate(loader):
            if a.max_steps and step>=a.max_steps: break
            opt.zero_grad(set_to_none=True)
            ep=wrapped(move_batch(raw,device,True))
            loss,_,logs=objective(ep,mc,lc)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,a.grad_clip,error_if_nonfinite=True)
            opt.step(); stat.update(ep,logs); steps+=1
            if rank==0 and (step+1)%a.log_every==0:
                print(f'[MR TRAIN] epoch={epoch} step={step+1}/{len(loader)} lr={lr:.7g} '
                      f'loss={float(loss):.5f} shape={float(logs["global_shape"]):.5f} '
                      f'local={float(logs["local_shape"]):.5f} elapsed_s={time.monotonic()-t0:.1f}',flush=True)
            del ep,loss
        stat.synchronize(device)
        if rank==0: print(f'[MR VAL] epoch={epoch} distributed validation begins',flush=True)
        validation=validate(model,vloader,device,mc,lc)
        states=[None]*world; gens=[None]*world
        if world>1: dist.all_gather_object(states,rng_state()); dist.all_gather_object(gens,gen.get_state())
        else: states[0]=rng_state(); gens[0]=gen.get_state()
        peak=dict(rank=rank,allocated_bytes=torch.cuda.max_memory_allocated(device),
                  reserved_bytes=torch.cuda.max_memory_reserved(device))
        peaks=[None]*world
        if world>1: dist.all_gather_object(peaks,peak)
        else: peaks[0]=peak
        if rank==0:
            row=dict(epoch=epoch,lr=lr,steps=steps,train=stat.report(),validation=validation,
                     seconds=time.monotonic()-t0,peak_cuda_memory=peaks)
            history.append(row); atomic_json(out/'metrics.json',history)
            atomic_torch(latest,dict(version=VERSION,protocol=protocol,signature=signature,epoch=epoch,
                model=model.state_dict(),optimizer=opt.state_dict(),history=history,rng=states,loader_rng=gens))
            print('[MR EPOCH] '+json.dumps(row),flush=True)
        if world>1: dist.barrier()  # only bounded checkpoint write, NOT rank0-only validation

if __name__=='__main__':
    try: main()
    finally:
        if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()
