"""Offline probes of the archived production code. Does not construct a trainer."""
import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, binary_dilation


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False, default=lambda v:v.item()), encoding='utf8')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def rows(p, rr):
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with Path(p).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rr)


def ply(p):
    types={'double':'<f8','float':'<f4','uchar':'u1','uint8':'u1'};props=[];count=None
    with Path(p).open('rb') as f:
        while True:
            line=f.readline().decode('ascii').strip()
            if line.startswith('format'):assert line=='format binary_little_endian 1.0'
            if line.startswith('element vertex'):count=int(line.split()[-1])
            if line.startswith('property'):
                _,kind,name=line.split();props.append((name,types[kind]))
            if line=='end_header':break
        a=np.fromfile(f,dtype=props,count=count)
    red=a['red']>a['blue'];blue=a['blue']>a['red'];out=[]
    for mask in (red,blue):
        v=a[mask];assert len(v)==448*448
        out.append(np.stack([v[k] for k in ('x','y','z')],axis=-1).reshape(448,448,3))
    assert np.max(np.abs(out[0][...,:2]/out[0][...,2:]-out[1][...,:2]/out[1][...,2:]))<1e-5
    return out


def spectrum(x, scale=1., highpass=True):
    """scale=native pixel / output pixel: match cycles/image, not native cpp."""
    x=np.asarray(x,np.float64)
    if x.ndim==2:x=x[None]
    c,h,w=x.shape
    if min(h,w)<16 or not np.isfinite(x).all():return {'spectrum_status':'invalid_ROI'},None,None
    win=np.outer(np.hanning(h),np.hanning(w));a=x-gaussian_filter(x,(0,3*scale,3*scale)) if highpass else x
    f=np.fft.fft2(a*win);power=np.abs(f)**2
    fy,fx=np.meshgrid(np.fft.fftfreq(h)*scale,np.fft.fftfreq(w)*scale,indexing='ij')
    band=((abs(fx-.21205)<=.04)&(abs(fy+.21205)<=.04))|((abs(fx+.21205)<=.04)&(abs(fy-.21205)<=.04))
    hi=np.hypot(fx,fy)>.08;p=power.sum(axis=0);den=float(p[hi].sum());bp=float(p[band].sum())
    full=p.copy();full[0,0]=0;i,j=np.unravel_index(full.argmax(),full.shape);px,py=float(fx[i,j]),float(fy[i,j]);mag=math.hypot(px,py)
    return dict(spectrum_status='ok' if band.any() else 'reference_band_above_native_nyquist',R_B=bp/den if den>0 and band.any() else None,
                A_B=math.sqrt(bp/c)/(h*w*math.sqrt(float((win**2).mean()))) if band.any() else None,
                band_squared_norm=bp,high_squared_norm=den,peak_fx_output_cpp=px,peak_fy_output_cpp=py,
                peak_normal_period_output_px=1/mag if mag else None,normal_direction_deg=math.degrees(math.atan2(py,px)),channels=c,height=h,width=w),f,band


def rectangle(mask):
    # Largest fully valid rectangle, selected from GT only, shared across states.
    heights=np.zeros(mask.shape[1],int);best=(0,(0,0,0,0))
    for y,line in enumerate(mask):
        heights=np.where(line,heights+1,0);stack=[]
        for x,v in enumerate(list(heights)+[0]):
            start=x
            while stack and stack[-1][1]>v:
                left,h=stack.pop();area=(x-left)*h
                if area>best[0]:best=(area,(left,y-h+1,x,y+1))
                start=left
            if not stack or stack[-1][1]<v:stack.append((start,v))
    return best[1]


def metrics(pred,gt,inst,roi):
    valid=np.isfinite(gt)&(gt>=.2)&(gt<=1.);fg=valid&(inst>0);n=int(fg.sum());finite=int((fg&np.isfinite(pred)).sum())
    e=pred-gt;ok=n>0 and finite==n
    m=dict(fg_count=n,fg_finite_count=finite,fg_status='valid' if ok else 'empty' if not n else 'nonfinite',
           fg_mae_mm=float(abs(e[fg]).mean()*1000) if ok else None,fg_bias_mm=float(e[fg].mean()*1000) if ok else None)
    edge=np.zeros_like(fg)
    for axis in (0,1):
        sl0=[slice(None)]*2;sl1=sl0.copy();sl0[axis]=slice(None,-1);sl1[axis]=slice(1,None);s0,s1=tuple(sl0),tuple(sl1)
        bad=(inst[s0]!=inst[s1])|(abs(gt[s0]-gt[s1])>.01);edge[s0]|=bad;edge[s1]|=bad
    near=binary_dilation(edge,iterations=2)
    for name,region in [('interior',fg&~near),('edge_near',fg&near)]:
        dif=[]
        for axis in (0,1):
            p=np.diff(pred,axis=axis);g=np.diff(gt,axis=axis)
            sl0=[slice(None)]*2;sl1=sl0.copy();sl0[axis]=slice(None,-1);sl1[axis]=slice(1,None);s0,s1=tuple(sl0),tuple(sl1)
            sel=region[s0]&region[s1]&(inst[s0]==inst[s1]);dif.extend(abs(p-g)[sel].tolist())
        m[name+'_adjacent_error_mm']=float(np.mean(dif)*1000) if dif and np.isfinite(dif).all() else None;m[name+'_pairs']=len(dif)
    x0,y0,x1,y1=roi;sp,_,_=spectrum(e[y0:y1,x0:x1]);m.update(sp)
    if m.get('A_B') is not None:m['A_B_mm']=m.pop('A_B')*1000
    return m


def config(out):
    expr=ast.parse((out/'production_log_train.txt').read_text().splitlines()[0],mode='eval').body
    assert isinstance(expr,ast.Call) and expr.func.id=='Namespace'
    return {k.arg:ast.literal_eval(k.value) for k in expr.keywords}


def imports(out):
    sys.argv=[sys.argv[0]]
    import train_cva_ddp as prod
    for k,v in config(out).items():setattr(prod.cfgs,k,v)
    return prod


def seed(n=0):
    random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)


def tree_hash(x):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            v=v.detach().cpu().contiguous();h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v):h.update(k.encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            for z in v:add(z)
        else:h.update(repr(v).encode())
    add(x);return h.hexdigest()


def make_model(prod):
    c=prod.cfgs
    m=prod.economicgrasp_dpt(min_depth=c.min_depth,max_depth=c.max_depth,bin_num=c.bin_num,is_training=True,
        use_obs_depth=c.use_obs_depth,pose_depth_mode=c.pose_depth_mode,use_depth_comp=False,use_cdf=c.use_cdf,
        detach_depth=bool(c.detach_depth),vis_dir=None,vis_every=c.vis_every).cuda()
    return m


def bases(prod):
    c=prod.cfgs
    return {split:prod.GraspNetMultiDataset(c.dataset_root,camera=c.camera,split=split,voxel_size=c.voxel_size,
        num_points=c.num_point,remove_outlier=True,augment=False,use_gt_depth=False,use_fuse_depth=c.use_fuse_depth,
        graspness_mode=c.graspness_mode,min_depth=c.min_depth,max_depth=c.max_depth,bin_num=c.bin_num,
        depth_strides=1,extend_angle=True,load_grasp_payload=False) for split in ('train','test_seen')}


def inventory(a):
    out=a.output;c=config(out);prod=imports(out);rr=[]
    log=Path(c['log_dir'])
    for p in sorted(log.glob('*.tar')):
        s=torch.load(p,map_location='cpu',weights_only=False);full='model_state_dict' in s
        op=s.get('optimizer_state_dict') if full else None
        rr.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime,
            fields=list(s) if full else ['model_weights_only'],epoch=s.get('epoch') if full else None,
            optimizer_steps=sorted({int(v['step']) for v in op['state'].values()}) if op else None,
            optimizer_groups=[{k:v for k,v in g.items() if k!='params'}|{'parameter_tensors':len(g['params'])} for g in op['param_groups']] if op else None,
            rng=False,loader=False,scaler=False,scheduler=False));del s
    dump(out/'checkpoint_manifest.json',rr)
    with torch.no_grad():m=make_model(prod)
    d=bases(prod);manifest=[]
    # Fix identities before evaluating any model result. Replace one evenly spaced test endpoint with target.
    chosen={'train':np.linspace(0,len(d['train'])-1,4,dtype=int).tolist(),
            'test_seen':[0,2559,4599,7679]}
    for split,ids in chosen.items():
        for i in ids:manifest.append(dict(split=split,index=i,scene=d[split].scenename[i],frame=d[split].frameid[i]))
    assert any(r['scene']=='scene_0117' and r['frame']==247 for r in manifest)
    dump(out/'probe_manifest.json',dict(selection='four uniform train indices; test uniform endpoints/one-third plus predeclared target replacing two-thirds',frames=manifest,dataset_lengths={k:len(v) for k,v in d.items()}))
    cp0=next(r for r in rr if Path(r['path']).name=='checkpoint_0.tar');steps=cp0['optimizer_steps'];assert len(steps)==1
    possible=[w for w in range(1,7) if math.ceil(math.ceil(len(d['train'])/w)/c['batch_size'])==steps[0]]
    nodes={n:dict(type=type(x).__name__,kernel=getattr(x,'kernel_size',None),stride=getattr(x,'stride',None),padding=getattr(x,'padding',None)) for n,x in m.named_modules() if n.startswith('depth_net') and isinstance(x,(torch.nn.Conv2d,torch.nn.ConvTranspose2d))}
    contract=dict(status='P0 inventory; runtime forward checks pending',original_config=c,source_git='19e563db6afd0f139225b8b40809ff37d1c5f5f2',
        source_manifest='source_manifest.json',original_script_sha256=sha(out/'launch.sh'),production_script_syntax='bash -n passed',
        actual_imports={k:str(sys.modules[k].__file__) for k in ['train_cva_ddp','models.economicgrasp_bip3d','models.economicgrasp_depth','dataset.graspnet_dataset']},
        interpreter=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),
        backend=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,deterministic=torch.are_deterministic_algorithms_enabled(),
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(),math_sdp=torch.backends.cuda.math_sdp_enabled(),mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled()),
        routes=dict(E_detach=m.spatial_enhancer.detach_depth_grad,Q_detach=m.detach_depth,C_detach=m.kview_grasp_module.config.detach_depth),
        model_class=type(m).__name__,geometry_depth_source=m.geometry_depth_source,pose_depth_mode=m.pose_depth_mode,seed_selection_mode=m.seed_selection_mode,
        constructor_initialization='frozen pretrained DINO weights only; DPT/FiLM and grasp heads constructor initialization; no external trainer checkpoint per logged Namespace',
        constructor_weight_path=str(Path('checkpoints/depth_anything_v2_vitb.pth').resolve()),constructor_weight_sha256=sha('checkpoints/depth_anything_v2_vitb.pth'),
        trainable_depth_parameters=[n for n,p in m.named_parameters() if n.startswith('depth_net') and p.requires_grad],
        frozen_backbone=all(not p.requires_grad for p in m.depth_net.depthnet.pretrained.parameters()),
        world_size_evidence=dict(script_default=3,compatible_with_checkpoint0_steps_and_actual_dataset_lengths=possible),
        batch_per_rank=c['batch_size'],accumulation=1,amp=False,scaler=None,global_clip=1.,optimizer='AdamW',schedule='cosine by epoch: base_lr*(1+cos(epoch/max_epoch*pi))/2',
        data=dict(lengths={k:len(v) for k,v in d.items()},subset='full scene lists; no 10% subset in actual entry/data code',augment=False,depth_stride=1,
            rgb_resize=str(d['train'].img_transforms),gt_instance_resize='nearest',GT='fused GT construction from production dataset'),
        pose=dict(mode='global_film',camera_pose='unit -R_table_from_cam[:,2] in table coordinates',gravity='unit R_table_from_cam.T @ [0,0,1], world-up in camera coordinates',K='crop-adjusted and rescaled 448x448 intrinsics'),
        checkpoint_exact_resume=False,PLY=dict(path=str(out/'original_target.ply'),sha256=sha(out/'original_target.ply'),iteration=54000,optimizer_update='pending mapping',mode='test_seen therefore eval; production is_training remains True'),
        operator_metadata=nodes,notes=['No running trainer found in owned-process snapshot; source snapshot is current files, historical immutability not independently recorded.','No production numeric operator replaced.','Source flags checked at model construction; autograd route checks pending.'])
    dump(out/'run_contract.json',contract)
    pred,gt=ply(out/'original_target.ply');sp,_,_=spectrum(pred[...,2]-gt[...,2]);dump(out/'original_ply_spectrum.json',sp)
    np.savez_compressed(out/'original_ply_depth.npz',pred=pred[...,2],gt=gt[...,2],ray=gt[...,:2]/gt[...,2:])
    history=[]
    for p in sorted(Path('/home/robotarm/EconomicGrasp/vis/dpt_cva_cdf_detach_depth').glob('dpt_pred_gt_xyz_*.ply')):
        try:
            pr,gg=ply(p);ss,_,_=spectrum(pr[...,2]-gg[...,2]);history.append(dict(path=str(p),mtime=p.stat().st_mtime,sha256=sha(p),**ss))
        except AssertionError:history.append(dict(path=str(p),status='filtered_or_incomplete_not_reshaped'))
    rows(out/'original_ply_history.csv',history)
    print(json.dumps(dict(stage='inventory',checkpoints=len(rr),world_size_candidates=possible,probes=len(manifest),original_spectrum=sp)),flush=True)


def selftest():
    y,x=np.mgrid[:448,:448];a=.01*np.sin(2*np.pi*(95*x/448-95*y/448));s,_,_=spectrum(a)
    assert abs(abs(s['peak_fx_output_cpp'])-95/448)<1e-8 and s['R_B']>.99
    assert abs(s['A_B']-.01/np.sqrt(2))<1e-4
    mask=np.ones((8,9),bool);mask[:2]=False;assert rectangle(mask)==(0,2,9,8)
    gt=np.ones((32,32))*.5;pred=gt.copy();pred[5,5]=np.nan
    assert metrics(pred,gt,np.ones_like(gt),(0,0,32,32))['fg_status']=='nonfinite'
    print('SELFTEST PASS: known-frequency amplitude and band, valid rectangle, nonfinite foreground')


def get_sample(base, frame, out):
    import cv2
    from PIL import Image
    seed(1000+frame['index'])
    s=base[frame['index']];crop=s['crop_box'].tolist();x0,y0,x1,y1=crop
    path=Path(base.labelpath[frame['index']]);inst=np.asarray(Image.open(path))
    inst=cv2.resize(inst[y0:y1,x0:x1].astype(np.int32),(448,448),interpolation=cv2.INTER_NEAREST)
    gt=s['gt_depth_m'].squeeze();valid=np.isfinite(gt)&(gt>=.2)&(gt<=1.)
    roi=rectangle(valid)
    # Unused dense GT-distribution tensors are omitted from the saved diagnostic input only.
    small={k:v for k,v in s.items() if k not in ('depth_prob_gt','depth_prob_weight')}
    stem=frame['split']+'_'+frame['scene']+f"_{frame['frame']:04d}"
    (out/'inputs').mkdir(exist_ok=True);torch.save(small,out/'inputs'/(stem+'.pt'))
    np.savez_compressed(out/'inputs'/(stem+'.npz'),instance=inst,gt=gt,rgb=s['img'].numpy(),K=s['K'],crop_box=crop,
                        camera_pose_vec=s['camera_pose_vec'],camera_gravity_vec=s['camera_gravity_vec'])
    meta=frame|dict(stem=stem,crop_box=crop,roi=list(roi),input_sha256=tree_hash(small),instance_sha256=tree_hash(inst),label_file_sha256=sha(path))
    return small,inst,meta


class Preserve:
    def __init__(self,m):self.m=m
    def __enter__(self):
        self.rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        self.buffers={n:b.detach().clone() for n,b in self.m.named_buffers()}
        self.flags=[(x,{k:getattr(x,k) for k in ('training','_vis_iter','is_training','detach_depth','detach_depth_grad') if hasattr(x,k)}) for x in self.m.modules()]
        self.c_detach=self.m.kview_grasp_module.config.detach_depth
        return self
    def __exit__(self,*exc):
        with torch.no_grad():
            for n,b in self.m.named_buffers():b.copy_(self.buffers[n])
        for x,flags in self.flags:
            for k,v in flags.items():setattr(x,k,v)
        self.m.kview_grasp_module.config.detach_depth=self.c_detach
        random.setstate(self.rng[0]);np.random.set_state(self.rng[1]);torch.set_rng_state(self.rng[2]);torch.cuda.set_rng_state_all(self.rng[3])


def depth_forward(m,s):
    return m.depth_net(s['img'][None].cuda(),camera_pose_vec=torch.as_tensor(s['camera_pose_vec'])[None].cuda(),
        camera_gravity_vec=torch.as_tensor(s['camera_gravity_vec'])[None].cuda(),camera_K=torch.as_tensor(s['K'])[None].cuda(),
        return_raw=True,return_feats=True,return_pose_aux=True)


def hooks(m,captured):
    handles=[];head=m.depth_net.depthnet.depth_head
    def put(name,t):
        if t.ndim==3 and t.shape[1]==1024:t=t.transpose(1,2).reshape(t.shape[0],t.shape[2],32,32)
        if t.ndim==4:
            arr=t[0].detach().float().cpu().numpy().copy();captured[name]=arr[:min(4,len(arr))]
            energy=None;bandpower=highpower=0.
            for start in range(0,len(arr),8):
                sp,f,band=spectrum(arr[start:start+8],scale=arr.shape[-1]/448)
                if f is None:continue
                p=(abs(f)**2).sum(axis=0);energy=p if energy is None else energy+p
                bandpower+=sp['band_squared_norm'];highpower+=sp['high_squared_norm']
            if energy is not None:
                h,w=arr.shape[-2:];energy[0,0]=0;iy,ix=np.unravel_index(energy.argmax(),energy.shape);scale=w/448
                sp.update(channels=len(arr),R_B=bandpower/highpower if highpower and band.any() else None,
                    A_B=math.sqrt(bandpower/len(arr))/(h*w*math.sqrt(float((np.outer(np.hanning(h),np.hanning(w))**2).mean()))) if band.any() else None,
                    band_squared_norm=bandpower,high_squared_norm=highpower,peak_fx_output_cpp=float(np.fft.fftfreq(w)[ix]*scale),peak_fy_output_cpp=float(np.fft.fftfreq(h)[iy]*scale))
                px,py=sp['peak_fx_output_cpp'],sp['peak_fy_output_cpp'];mag=math.hypot(px,py)
                sp.update(peak_normal_period_output_px=1/mag if mag else None,normal_direction_deg=math.degrees(math.atan2(py,px)))
                captured[name+'__all_channel_spectrum']=sp
    def hook(name):return lambda module,args,output:put(name,output)
    for i,x in enumerate(head.resize_layers):handles.append(x.register_forward_hook(hook(f'resize_{i}')))
    for n in ('refinenet1','output_conv1','output_conv2'):
        handles.append(getattr(head.scratch,n).register_forward_hook(hook(n)))
    for i,x in enumerate(head.scratch.output_conv2):handles.append(x.register_forward_hook(hook(f'output_conv2_part_{i}')))
    handles.append(head.scratch.output_conv2.register_forward_pre_hook(lambda m,args:put('bilinear_448',args[0])))
    def film_pre(module,args):
        for i,t in enumerate(args[0]):put(f'DINO_before_FiLM_{i}',t[0])
    def film_post(module,args,output):
        for i,t in enumerate(output[0]):put(f'DINO_after_FiLM_{i}',t[0])
    handles.extend([m.depth_net.pose_aware_adapter.register_forward_pre_hook(film_pre),m.depth_net.pose_aware_adapter.register_forward_hook(film_post)])
    for i,x in enumerate(m.depth_net.pose_aware_adapter.level_film):
        handles.append(x.register_forward_hook(lambda module,args,output,i=i:captured.__setitem__(f'gamma_beta_{i}',output[0].detach().cpu().numpy().copy())))
    return handles


def probe(a):
    out=a.output;prod=imports(out);seed();m=make_model(prod);ds=bases(prod)
    saved=json.loads((out/'probe_manifest.json').read_text());samples=[]
    for fr in saved['frames']:samples.append(get_sample(ds[fr['split']],fr,out))
    saved['frames']=[x[2] for x in samples];dump(out/'probe_manifest.json',saved)
    manifest=json.loads((out/'checkpoint_manifest.json').read_text())
    selected=[next(r for r in manifest if Path(r['path']).name==name) for name in ('checkpoint_0.tar','checkpoint_5.tar','checkpoint_15.tar')]
    selected.append(next(r for r in manifest if Path(r['path']).name.startswith('epoch_16_')))
    dump(out/'selected_checkpoints.json',selected)
    results=[];layer_rows=[];checks=[]
    for si,cp in enumerate(selected):
        label=['epoch0','epoch5','epoch15','epoch16_target'][si];state=torch.load(cp['path'],map_location='cpu',weights_only=False)
        state=state.get('model_state_dict',state);m.load_state_dict(state,strict=True);del state
        before=tree_hash(m.state_dict());t0=time.time()
        for s,inst,fr in samples:
            with Preserve(m),torch.no_grad():
                m.eval();base=depth_forward(m,s)[0][0,0].cpu().numpy().copy()
            with Preserve(m),torch.no_grad():
                m.eval();repeat=depth_forward(m,s)[0][0,0].cpu().numpy().copy()
            noise=float(abs(base-repeat).max());captured={}
            target=fr['scene']=='scene_0117' and fr['frame']==247
            if target:
                with Preserve(m),torch.no_grad():
                    m.eval();hs=hooks(m,captured)
                    try:res=depth_forward(m,s);withhooks=res[0][0,0].cpu().numpy().copy()
                    finally:
                        for h in hs:h.remove()
                error=float(abs(base-withhooks).max());assert error<=max(noise*2,1e-7)
                checks.append(dict(checkpoint=label,noise_m=noise,hooks_difference_m=error))
                captured.update(raw=res[3][0].cpu().numpy(),metric=base[None])
                (out/'activations').mkdir(exist_ok=True);np.savez_compressed(out/'activations'/(label+'.npz'),**{k:v for k,v in captured.items() if isinstance(v,np.ndarray)})
                for name,v in captured.items():
                    if isinstance(v,dict):continue
                    if v.ndim!=3:continue
                    if name+'__all_channel_spectrum' in captured:sp=captured[name+'__all_channel_spectrum']
                    else:sp,_,_=spectrum(v,scale=v.shape[-1]/448)
                    layer_rows.append(dict(checkpoint=label,node=name,scope='all channels, per-channel power summed; saved maps first min(4,C)',**sp))
            gt=s['gt_depth_m'].squeeze();rr=metrics(base,gt,inst,fr['roi'])
            results.append(dict(checkpoint=label,checkpoint_sha256=cp['sha256'],**fr,mode='eval',**rr))
            (out/'predictions').mkdir(exist_ok=True)
            np.savez_compressed(out/'predictions'/(label+'_'+fr['stem']+'.npz'),pred=base,gt=gt,instance=inst,roi=fr['roi'])
            if target and si==3:
                original=np.load(out/'original_ply_depth.npz');
                raydiff=float(abs(original['gt']-gt).max());pdiff=float(abs(original['pred']-base).max())
                dump(out/'target_match.json',dict(checkpoint=cp['path'],gt_max_error_m=raydiff,pred_max_error_m=pdiff,
                    original_PLY_sha256=sha(out/'original_target.ply'),actual_mode='eval, is_training=True',native_depth_only_probe=True,
                    exact_tensor_match=pdiff<=max(noise*2,1e-6),matching_note='Production PLY epoch/rank mapping must also be checked; native depth path independent of task selection.'))
                rows(out/'target_profile.csv',[dict(u=u,v=224,GT_mm=float(gt[224,u]*1000),pred_mm=float(base[224,u]*1000),original_PLY_pred_mm=float(original['pred'][224,u]*1000)) for u in range(50,65)])
        assert tree_hash(m.state_dict())==before,'Diagnostic changed model state'
        rows(out/'depth_spectrum.csv',results);rows(out/'layer_spectrum.csv',layer_rows);dump(out/'probe_checks.json',checks)
        print(json.dumps(dict(stage='probe',checkpoint=label,elapsed=time.time()-t0,frames=8)),flush=True)
    inputs=[]
    for s,inst,fr in samples:
        x0,y0,x1,y1=fr['roi']
        for key,x in [('GT',s['gt_depth_m'].squeeze()),('RGB_channelwise',s['img'].numpy())]:
            sp,_,_=spectrum(x[...,y0:y1,x0:x1]);inputs.append(dict(**fr,input=key,**sp))
    rows(out/'input_spectrum.csv',inputs)


def route(m,name):
    m.spatial_enhancer.detach_depth_grad=name in ('none','noE')
    m.detach_depth=name in ('none','noQ')
    m.kview_grasp_module.config.detach_depth=name in ('none','noC')


def target_batch(a):
    out=a.output;prod=imports(out);seed();m=make_model(prod);base=bases(prod)['test_seen']
    cp=next(r for r in json.loads((out/'selected_checkpoints.json').read_text()) if Path(r['path']).name.startswith('epoch_16_'))
    state=torch.load(cp['path'],map_location='cpu',weights_only=False);m.load_state_dict(state.get('model_state_dict',state),strict=True);del state
    indices=[4599,4602,4605];ss=[]
    for i in indices:
        seed(1000+i);ss.append(base[i])
    inp={k:torch.stack([torch.as_tensor(s[k]) for s in ss]).cuda() for k in ('img','camera_pose_vec','camera_gravity_vec','K')}
    before=tree_hash(m.state_dict());rr=[]
    for _ in range(2):
        with Preserve(m),torch.no_grad():
            m.eval();r=m.depth_net(inp['img'],camera_pose_vec=inp['camera_pose_vec'],camera_gravity_vec=inp['camera_gravity_vec'],camera_K=inp['K'],return_raw=True,return_feats=True,return_pose_aux=True)
            rr.append(r[0][0,0].cpu().numpy().copy())
    original=np.load(out/'original_ply_depth.npz');pred=rr[0];noise=float(abs(rr[0]-rr[1]).max());pdiff=float(abs(pred-original['pred']).max())
    folder=out/'export_roundtrip';folder.mkdir(exist_ok=True)
    with Preserve(m),torch.no_grad():
        m.vis_dir=str(folder);m._vis_iter=54000
        yy,xx=torch.meshgrid(torch.arange(448,device='cuda'),torch.arange(448,device='cuda'),indexing='ij')
        uv=torch.stack([xx,yy],-1).reshape(1,-1,2).float()
        pc=m._backproject_uvz(uv,torch.as_tensor(pred,device='cuda').reshape(1,-1,1),inp['K'][:1])
        gc=m._backproject_uvz(uv,torch.as_tensor(original['gt'],device='cuda',dtype=pc.dtype).reshape(1,-1,1),inp['K'][:1])
        m._save_pred_gt_cloud_ply(pc,gc,{'scene_idx':117,'anno_idx':247})
        m.vis_dir=None
    rp,rg=ply(next(folder.glob('*.ply')))
    roundtrip=float(abs(rp[...,2]-pred).max());assert roundtrip==0
    assert tree_hash(m.state_dict())==before
    dump(out/'target_batch_match.json',dict(checkpoint=cp['path'],checkpoint_sha256=cp['sha256'],test_seen_indices=indices,
        scenes=[base.scenename[i] for i in indices],frames=[base.frameid[i] for i in indices],per_rank_batch=3,rank=0,world_size=3,
        export_forward_index=54000,epoch_zero_based=16,optimizer_updates=17*2845,eval_batch_zero_based=511,
        derivation='epochs0..9:10*2845; epochs10..15:6*(2845+854); epoch16 train:2845; 54000-(10*2845+6*(2845+854)+2845)=511; rank0 sample 511*3*3=4599',
        mode='eval with model.is_training=True; native depth path',gt_max_error_m=float(abs(original['gt']-ss[0]['gt_depth_m'].squeeze()).max()),
        pred_max_error_m=pdiff,replay_noise_m=noise,export_readback_error_m=roundtrip,exact_tensor_match=pdiff<=max(2*noise,1e-6)))
    np.savez_compressed(out/'target_batch_prediction.npz',pred=pred,gt=original['gt'],original=original['pred'])
    print(json.dumps(dict(stage='target_batch',max_difference_m=pdiff,noise_m=noise)),flush=True)


def batch_for(prod,base,frames):
    adapter=prod.CVAExtendedLabelAdapter(base,dataset_root=prod.cfgs.dataset_root,use_cdf=True,
        label_folder=prod.cfgs.cdf_label_folder,num_angle=prod.cfgs.num_angle,num_depth=prod.cfgs.num_depth)
    ss=[]
    for fr in frames:
        seed(1000+fr['index']);s=adapter[fr['index']]
        # Neither field is read by the production regression objective or model.
        for key in ('depth_prob_gt','depth_prob_weight'):s.pop(key,None)
        ss.append(s)
    b=prod.collate_fn(ss);prod.validate_batch_label_contract(b,True)
    return b


def loss_coverage(ep):
    c=dict(depth_denominator=int(ep['depth_map_pred'].numel()),view_denominator=int(ep['view_score'].numel()))
    gt=ep['gt_depth_m'];c['depth_GT_valid']=int(((gt>=.2)&(gt<=1.)&torch.isfinite(gt)).sum())
    for k in ('batch_grasp_cdf_valid_mask','batch_grasp_width_valid_mask_angle_depth','batch_valid_mask','batch_grasp_view_valid_mask'):
        if k in ep:c[k]=dict(count=int(ep[k].bool().sum()),total=ep[k].numel(),empty=not bool(ep[k].bool().any()))
    if 'batch_grasp_cdf_bins_angle_depth' in ep:
        c['cdf_positive']=int(((ep['batch_grasp_cdf_bins_angle_depth']>0)&ep['batch_grasp_cdf_valid_mask']).sum())
    c['view_reduction']='unmasked SmoothL1 mean over all B*M*V (production code)'
    c['cdf_reduction']='mean over valid Q*A*D separately per threshold, then threshold mean'
    c['width_reduction']='mean over actual width-valid Q*A*D; target meters multiplied by10'
    return c


def grad_stats(gs,ref):
    present=[g for g in gs if g is not None]
    if not present:return dict(state='disconnected',norm=None,cosine_vs_depth=None)
    if not all(torch.isfinite(g).all() for g in present):return dict(state='nonfinite',norm=None,cosine_vs_depth=None)
    n=math.sqrt(sum(float(g.double().square().sum()) for g in present));dn=math.sqrt(sum(float(g.double().square().sum()) for g in ref if g is not None))
    dot=sum(float((g.double()*d.double()).sum()) for g,d in zip(gs,ref) if g is not None and d is not None)
    return dict(state='connected_nonzero' if n else 'connected_zero',norm=n,cosine_vs_depth=dot/(n*dn) if n*dn>1e-20 else None,
                depth_norm=dn,ratio_vs_depth=n/dn if dn>1e-12 else None,unused_tensors=sum(g is None for g in gs))


class PathCapture:
    """Read-only observations; restore methods and Python tracer afterwards."""
    def __init__(self,m):self.m=m;self.arrays={};self.old=[];self.collisions=[];self.refs={}
    def keep(self,name,t):
        if torch.is_tensor(t):self.arrays[name]=t[0].detach().cpu().numpy().copy()
    def __enter__(self):
        import inspect
        def wrap(obj,name,callback):
            old=getattr(obj,name);self.old.append((obj,name,old));sig=inspect.signature(old)
            def call(*args,**kwargs):
                bound=sig.bind(*args,**kwargs).arguments;r=old(*args,**kwargs);callback(bound,r);return r
            setattr(obj,name,call)
        def gse(a,r):
            self.keep('GSE_depth_input',a['depth_map']);self.keep('GSE_resized_mean',r[0]);self.refs['GSE_resized_mean']=r[0]
        def seed(a,r):self.keep('seed_xyz',r[1]);self.keep('seed_uv_index',r[2])
        def aux(a,r):
            for key in ('grid','depth_map','seed_xyz'):self.keep('CVA_'+key,a[key])
            self.keep('CVA_depth_delta',r[0])
        wrap(self.m.spatial_enhancer,'_depth_map_moments',gse)
        wrap(self.m,'_select_graspable_seed_queries',seed)
        for mod in self.m.modules():
            if hasattr(mod,'_sample_aux_maps'):wrap(mod,'_sample_aux_maps',aux)
        self.tracer=sys.gettrace()
        def trace(frame,event,arg):
            if event=='line' and frame.f_code.co_filename.endswith('utils/label_generation.py') and frame.f_lineno==1580:
                v=frame.f_locals;row=v['row_id'].detach().cpu().numpy();slot=v['slot_id'].detach().cpu().numpy();scene=v['scene_view_id'].detach().cpu().numpy()
                keys=row*v['top_view_scene'].shape[1]+slot;u,c=np.unique(keys,return_counts=True)
                self.collisions.append(dict(batch_i=v['batch_i'],object_i=v['obj_i'],writes=len(keys),duplicate_destinations=int((c>1).sum()),
                    max_writes_per_destination=int(c.max()) if len(c) else 0,distinct_scene_views_per_duplicate='torch.where indices are distinct; duplicate assignments have differing values'))
            return trace if frame.f_code.co_filename.endswith('utils/label_generation.py') else None
        sys.settrace(trace);return self
    def __exit__(self,*exc):
        sys.settrace(self.tracer)
        for obj,name,old in reversed(self.old):setattr(obj,name,old)


def audit(a):
    out=a.output;prod=imports(out);seed();m=make_model(prod);ds=bases(prod)
    frames=json.loads((out/'probe_manifest.json').read_text())['frames']
    selected=json.loads((out/'selected_checkpoints.json').read_text());selected=[selected[0],selected[-1]]
    batches={}
    for split in ('train','test_seen'):
        choices=[r for r in frames if r['split']==split]
        if split=='test_seen':choices=sorted(choices,key=lambda r:r['scene']!='scene_0117')
        fr=choices[:a.audit_batch_size];batches[split]=(batch_for(prod,ds[split],fr),fr)
    dump(out/'audit_batch_manifest.json',{sp:dict(frames=fr,batch_hash=tree_hash(b),batch_size=len(fr),gradient_scope='local train-mode probe, production final-train-batch size 8534%3=2, not DDP global gradient') for sp,(b,fr) in batches.items()})
    params=[(n,p) for n,p in m.named_parameters() if p.requires_grad and n.startswith('depth_net.')]
    dpt=[i+2 for i,(n,p) in enumerate(params) if '.depth_head.' in n];film=[i+2 for i,(n,p) in enumerate(params) if '.pose_aware_adapter.' in n]
    assert len(dpt)+len(film)==len(params) and not set(dpt)&set(film)
    dump(out/'gradient_parameter_scope.json',{'DPT':[params[i-2][0] for i in dpt],'FiLM':[params[i-2][0] for i in film]})
    records=[];equivalence=[];lossrows=[];pathrows=[]
    (out/'gradients').mkdir(exist_ok=True)
    for cp in selected:
        label='epoch0' if 'checkpoint_0.' in cp['path'] else 'epoch16_target'
        s=torch.load(cp['path'],map_location='cpu',weights_only=False);m.load_state_dict(s.get('model_state_dict',s),strict=True);del s
        statehash=tree_hash(m.state_dict())
        for sp,(cpu,fr) in batches.items():
            inputhash=tree_hash(cpu);baseline=None;noise={};all_maps={};all_vectors={};map_vectors={}
            for condition in ('all','all_repeat','none','noE','noQ','noC'):
                t0=time.time();real='all' if condition=='all_repeat' else condition
                with Preserve(m):
                    m.train();seed(1901);route(m,real)
                    b=prod.move_batch_to_device(copy.deepcopy(cpu),torch.device('cuda'),True,False)
                    b['cva_compute_diagnostics']=True;b['geometry_compute_diagnostics']=False;b['cva_export_angle_feature']=False
                    prod.assert_cpu_resident_label_lists(b,True)
                    if condition=='all' and sp=='test_seen':
                        with PathCapture(m) as pc:ep=m(b)
                        np.savez_compressed(out/'activations'/f'{label}_task_paths_train_mode.npz',**pc.arrays)
                        dump(out/f'{label}_label_assignment_collisions.json',pc.collisions)
                        del pc
                    else:ep=m(b)
                    total,ep=prod.get_loss_economicgrasp(ep,use_cdf=True)
                    compare={k:v.detach().cpu().clone() for k,v in ep.items() if torch.is_tensor(v) and (k in ('depth_net_pred','depth_head_raw_pred','view_score','grasp_cdf_pred_angle_depth','grasp_width_pred_angle_depth','token_sel_idx','xyz_graspable') or k.startswith('batch_') or ('view' in k and ('ind' in k or 'idx' in k)))}
                    compare['overall_loss']=total.detach().cpu()
                    if baseline is None:baseline=compare
                    else:
                        assert set(compare)==set(baseline)
                        diffs={k:float((v.double()-baseline[k].double()).abs().max()) for k,v in compare.items()}
                        if condition=='all_repeat':noise=diffs
                        bad={k:v for k,v in diffs.items() if v>(0 if k.startswith('batch_') or 'idx' in k or 'ind' in k else max(noise.get(k,0)*2,1e-7))}
                        equivalence.append(dict(checkpoint=label,split=sp,condition=condition,max_difference=max(diffs.values()),per_key=diffs,forward_equivalent=not bad,failed_keys=bad,
                            interpretation='invalid for between-forward causal path differences when labels or discrete choices differ'))
                    losses={'depth':ep['B: DepthReg Loss']*prod.cfgs.depth_prob_loss_weight,'view':ep['B: View Loss']*prod.cfgs.view_loss_weight,
                        'cdf':ep['B: CDF Loss']*prod.cfgs.score_loss_weight,'width':ep['B: Width Loss']*prod.cfgs.width_loss_weight}
                    losses['task_sum']=losses['view']+losses['cdf']+losses['width']
                    if condition=='all' and sp=='test_seen' and label=='epoch16_target':
                        losses.update(objectness=ep['B: Objectness Loss']*prod.cfgs.objectness_loss_weight,graspness=ep['B: Graspness Loss']*prod.cfgs.graspness_loss_weight)
                    targets=[ep['depth_net_pred'],ep['depth_head_raw_pred']]+[p for n,p in params]
                    ref=None;vecs={};maps={}
                    coverage=loss_coverage(ep)
                    for loss,lv in losses.items():
                        gs=torch.autograd.grad(lv,targets,retain_graph=True,allow_unused=True)
                        gs=[g.detach().cpu() if g is not None else None for g in gs]
                        if loss=='depth':ref=gs
                        vecs[loss]=gs
                        weights={'depth':prod.cfgs.depth_prob_loss_weight,'view':prod.cfgs.view_loss_weight,'cdf':prod.cfgs.score_loss_weight,'width':prod.cfgs.width_loss_weight,'objectness':prod.cfgs.objectness_loss_weight,'graspness':prod.cfgs.graspness_loss_weight}
                        for scale in ('weighted','raw') if loss!='task_sum' else ('weighted',):
                            weight=weights.get(loss,1.) if scale=='raw' else 1.
                            cur=[g/weight if g is not None else None for g in gs]
                            depthref=[g/(prod.cfgs.depth_prob_loss_weight if scale=='raw' else 1.) if g is not None else None for g in ref]
                            for scope,indices in [('z',[0]),('raw_logits',[1]),('DPT',dpt),('FiLM',film),('G',dpt+film)]:
                                records.append(dict(checkpoint=label,checkpoint_sha256=cp['sha256'],input_split=sp,audit_mode='train',batch_size=len(fr),condition=condition,
                                    route_kind='actual' if condition=='all' else 'repeat_baseline' if condition=='all_repeat' else 'forced',loss=loss,scale=scale,scope=scope,
                                    **grad_stats([cur[i] for i in indices],[depthref[i] for i in indices])))
                        lossrows.append(dict(checkpoint=label,input_split=sp,condition=condition,loss=loss,weighted_value=float(lv.detach()),raw_value=float(lv.detach())/weights[loss] if loss in weights else None,coverage=json.dumps(coverage)))
                        if gs[0] is not None:
                            maps[loss]=gs[0].numpy()
                            for bi,frame in enumerate(fr):
                                arr=gs[0][bi,0].numpy();spec,f,band=spectrum(arr)
                                dspec,df,_=spectrum(ref[0][bi,0].numpy())
                                records.append(dict(checkpoint=label,input_split=sp,audit_mode='train',batch_size=len(fr),condition=condition,route_kind='actual' if condition=='all' else 'forced',loss=loss,scale='weighted',scope='z_frequency',scene=frame['scene'],frame=frame['frame'],
                                    band_dot_vs_depth=float(np.real((np.conj(df)*f)[...,band].sum())),**spec))
                    if condition=='all':all_maps=maps;all_vectors=vecs
                    elif condition in ('noE','noQ','noC'):
                        for loss in ('view','cdf','width','task_sum'):
                            base=all_vectors[loss];other=vecs[loss]
                            delta=[None if g is None and v is None else (g if g is not None else torch.zeros_like(v))-(v if v is not None else torch.zeros_like(g)) for g,v in zip(base,other)]
                            pathrows.append(dict(checkpoint=label,input_split=sp,path=condition[2:],loss=loss,scope='G',**grad_stats([delta[i] for i in dpt+film],[all_vectors['depth'][i] for i in dpt+film])))
                            if delta[0] is not None:maps['all_minus_'+condition+'_'+loss]=delta[0].numpy()
                    np.savez_compressed(out/'gradients'/f'{label}_{sp}_{condition}.npz',**maps)
                    del ep,lv,total,targets,gs,ref,vecs,losses,compare,b
                assert tree_hash(cpu)==inputhash
                rows(out/'gradient_spectrum.csv',records);rows(out/'audit_losses.csv',lossrows);rows(out/'path_contributions.csv',pathrows);dump(out/'route_equivalence.json',equivalence)
                print(json.dumps(dict(stage='audit',checkpoint=label,split=sp,condition=condition,seconds=time.time()-t0)),flush=True)
            del all_vectors,all_maps
        assert tree_hash(m.state_dict())==statehash,'Audit polluted model parameters/buffers'
    route(m,'all')
    good=all(r['forward_equivalent'] for r in equivalence)
    dump(out/'audit_acceptance.json',dict(status='pass' if good else 'failed_forward_equivalence',state_preserved=True,input_preserved=True,numeric_backend='unchanged production default',world_scope='local probe gradients, not DDP updates',forward_equivalence=good,P3_eligible=good,
        route_differences_interpretable=good,actual_single_forward_loss_gradients_valid=True))


def view_backward(a):
    out=a.output;prod=imports(out);seed();m=make_model(prod);ds=bases(prod)
    frames=json.loads((out/'audit_batch_manifest.json').read_text())['test_seen']['frames'];cpu=batch_for(prod,ds['test_seen'],frames)
    cps=json.loads((out/'selected_checkpoints.json').read_text());rr=[]
    for cp,label in [(cps[0],'epoch0'),(cps[-1],'epoch16_target')]:
        s=torch.load(cp['path'],map_location='cpu',weights_only=False);m.load_state_dict(s.get('model_state_dict',s),strict=True);del s
        before=tree_hash(m.state_dict())
        with Preserve(m):
            m.train();seed(1901);route(m,'all');b=prod.move_batch_to_device(copy.deepcopy(cpu),torch.device('cuda'),True,False)
            b['cva_compute_diagnostics']=True;b['geometry_compute_diagnostics']=False;b['cva_export_angle_feature']=False
            with PathCapture(m) as pc:ep=m(b)
            total,ep=prod.get_loss_economicgrasp(ep,use_cdf=True)
            gs=torch.autograd.grad(ep['B: View Loss']*prod.cfgs.view_loss_weight,[pc.refs['GSE_resized_mean'],ep['depth_net_pred']],allow_unused=True)
            saved={}
            for name,g in zip(['resized_mean_gradient','depth_gradient'],gs):
                assert g is not None
                x=g[0].detach().cpu().numpy();saved[name]=x;sp,_,_=spectrum(x,scale=x.shape[-1]/448)
                rr.append(dict(checkpoint=label,node=name,loss='weighted view',mode='train',scope='local batch2 target first image',**sp))
            np.savez_compressed(out/'gradients'/f'{label}_view_E_backward.npz',**saved)
            del pc,ep,total,b,gs
        assert tree_hash(m.state_dict())==before
    rows(out/'view_E_backward.csv',rr)
    print('VIEW E BACKWARD COMPLETE; original bilinear align_corners=False, no operator replacement')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','selftest','probe','audit','target_batch','view_backward']);p.add_argument('--output',type=Path,required=True);p.add_argument('--audit-batch-size',type=int,choices=[2,3],default=2);a=p.parse_args()
    if a.stage=='selftest':selftest()
    elif a.stage=='inventory':inventory(a)
    elif a.stage=='probe':probe(a)
    elif a.stage=='target_batch':target_batch(a)
    elif a.stage=='view_backward':view_backward(a)
    else:audit(a)
