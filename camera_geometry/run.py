"""CLI for C0-C3 camera consistency and T0-T2 calibrated-plane studies."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from .data import (prepare,load_manifest,read_json,atomic_json,sha256,digest,inputs_for_views,
                   common_queries,stack_inputs)
from .geometry import transform_uv,rays,sample,losses
from .model import MetricModel,ARMS
from .evaluate import evaluate_model,VIEWS,grouped_mean_ci

DEFAULT = dict(backend='production',encoder='vitb',image_hw=[448,448],max_depth_m=1.,
    height_min_m=-.05,height_max_m=.35,grazing=.1,pose_hidden_dim=64,
    seed=0,steps=600,batch_size=4,train_rays=256,eval_rays=256,
    learning_rate=2e-5,weight_decay=1e-4,consistency_weight=1.,loss_scale_m=.01,
    save_every=50,bootstrap=1000,track_tolerance_m=.005,min_track_span_m=.02)


def config(path):
    given = read_json(path)
    unknown = set(given)-set(DEFAULT)-{'manifest','checkpoint','checkpoint_key','checkpoint_prefix'}
    if unknown: raise ValueError('Unknown config fields: '+str(sorted(unknown)))
    c = {**DEFAULT,**given}
    for key in ('steps','batch_size','train_rays','eval_rays','save_every','bootstrap'):
        if not isinstance(c[key],int) or c[key]<1: raise ValueError('Positive integer required: '+key)
    if c['backend'] not in ('production','tiny'): raise ValueError('Unknown backend')
    if c['backend']=='production' and c['image_hw']!=[448,448]: raise ValueError('Production requires 448x448')
    if (c['learning_rate']<=0 or c['weight_decay']<0 or c['consistency_weight']<0 or
        c['loss_scale_m']<=0 or c['max_depth_m']<=0 or not 0<c['grazing']<1 or
        c['height_min_m']>=c['height_max_m']): raise ValueError('Invalid physical/optimizer parameters')
    for key in ('manifest','checkpoint'):
        if c.get(key):
            p = Path(c[key]); c[key] = str((p if p.is_absolute() else Path(path).parent/p).resolve())
    return c


def setup(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    # grid_sample backward may be nondeterministic on CUDA; do not promise bitwise CUDA replay.


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'camera_geometry').glob('*.py'))
    for sub in ('models/dinov2_dpt.py','models/dinov2.py','models/pose_aware_depth.py'):
        p = root/sub
        if p.is_file(): paths.append(p)
    if (root/'models/dinov2_layers').exists(): paths += list((root/'models/dinov2_layers').rglob('*.py'))
    return {str(p.relative_to(root)):sha256(p) for p in sorted(paths)}


def select(m,partition):
    out = [e for e in m['entries'] if e['partition']==partition]
    if not out: raise ValueError('No entries in '+partition)
    return out


def batch(entries,cfg,rng,device):
    a=[]; b=[]; uv_a=[]; uv_b=[]; truth=[]; raw_rays=[]
    ids = rng.choice(len(entries),cfg['batch_size'],replace=len(entries)<cfg['batch_size'])
    for i in ids:
        e = entries[int(i)]; names = rng.choice(VIEWS,2,replace=False)
        views = inputs_for_views(e,names,cfg['image_hw'])
        uv,z,_ = common_queries(e,views,cfg['image_hw'],cfg['train_rays'],rng)
        a.append(views[0]); b.append(views[1]); truth.append(z)
        uv_a.append(transform_uv(uv,views[0]['A'])); uv_b.append(transform_uv(uv,views[1]['A']))
        raw_rays.append(rays(uv,torch.tensor(e['K'],dtype=torch.float32)))
    ts = [torch.stack(x).to(device) for x in (uv_a,uv_b,truth,raw_rays)]
    return stack_inputs(a,device),stack_inputs(b,device),*ts


def atomic_checkpoint(path,value):
    tmp = Path(str(path)+'.tmp'); torch.save(value,tmp); os.replace(tmp,path)


def train(config_path,arm,output,device='cuda:0',resume=False,smoke=False,stop_after=None):
    c = config(config_path)
    if c['backend']=='tiny' and not smoke: raise ValueError('tiny backend requires --smoke; not a science run')
    m = load_manifest(c['manifest']); tr=select(m,'train'); dev=select(m,'dev')
    if arm.startswith('T') and any('T_table_from_cam' not in e for e in tr+dev):
        raise ValueError('All table-study train/dev frames require calibrated plane')
    setup(c['seed']); model = MetricModel(c,arm).to(device)
    protocol = dict(config=c,arm=arm,manifest=m['fingerprint'],source_hashes=code_hashes(),
        initialization=model.load_audit,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        selection='fixed last step; initial/final dev reporting only',smoke=smoke,
        torch_version=str(torch.__version__),device=str(device),
        cuda_bitwise_replay_guaranteed=False,task_gradients_into_depth=False,
        conditioning='historical global FiLM frozen; optional identity ray/gravity increment',
        table_scalar_head_reset=arm.startswith('T'))
    root = Path(output); root.mkdir(parents=True,exist_ok=True)
    lock = root/'protocol.json'
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=c['learning_rate'],weight_decay=c['weight_decay'])
    rng = np.random.default_rng(c['seed']); step=0; history=[]; frozen=model.frozen_fingerprint()
    if resume:
        if read_json(lock)!=protocol: raise ValueError('Resume protocol/data/code differs')
        ck = torch.load(root/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model'],strict=True); opt.load_state_dict(ck['optimizer'])
        step,history,frozen = ck['step'],ck['history'],ck['frozen_fingerprint']
        rng.bit_generator.state=ck['numpy_rng']; torch.set_rng_state(ck['torch_rng'].cpu())
        if torch.cuda.is_available() and ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']])
    else:
        if any(root.iterdir()): raise FileExistsError('Output is populated; use a new directory or exact --resume')
        atomic_json(lock,protocol)
        initial = evaluate_model(model,dev,c,device)
        initial.update(arm=arm,step=0,manifest=m['fingerprint'],partition='dev',smoke=smoke)
        atomic_json(root/'initial_dev.json',initial)
    while step<c['steps']:
        model.train(); a,b,ua,ub,target,r = batch(tr,c,rng,device)
        oa,ob = model(**a),model(**b)
        za=sample(oa['z'][:,None],ua)[...,0]; zb=sample(ob['z'][:,None],ub)[...,0]
        mask=(sample(oa['eligible'][:,None].float(),ua,'nearest')[...,0]>.5)&(
              sample(ob['eligible'][:,None].float(),ub,'nearest')[...,0]>.5)
        weight=c['consistency_weight'] if model.use_consistency else 0.
        loss,terms=losses(za,zb,target,r,mask,weight,c['loss_scale_m'])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step(); step+=1
        history.append(dict(step=step,loss=float(loss.detach()),**{k:float(v) for k,v in terms.items()}))
        if step%c['save_every']==0 or step==c['steps'] or step==stop_after:
            if model.frozen_fingerprint()!=frozen: raise RuntimeError('Frozen backbone/global FiLM changed')
            atomic_checkpoint(root/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),
                step=step,history=history,protocol=protocol,frozen_fingerprint=frozen,numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
            atomic_json(root/'history.json',history)
            print(f'{arm} step {step}/{c["steps"]} loss={float(loss):.5f}',flush=True)
        if step==stop_after and step<c['steps']: return root/'last.pt'  # unit-test interruption seam, not CLI
    result=evaluate_model(model,dev,c,device)
    result.update(arm=arm,step=step,manifest=m['fingerprint'],partition='dev',smoke=smoke,
                  checkpoint_sha256=sha256(root/'last.pt'),
                  comparison_lock=digest({'config':c,'manifest':m['fingerprint'],'code':protocol['source_hashes'],'smoke':smoke}))
    atomic_json(root/'final_dev.json',result)
    atomic_json(root/'status.json',{'training_complete':True,'official_ap_not_run':True,
                                  'frozen_modules_unchanged':True,'smoke':smoke})
    return root/'last.pt'


def load_run(checkpoint,device):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    p=ck['protocol']
    if p['source_hashes'] != code_hashes(): raise ValueError('Evaluation source differs from locked training source')
    model=MetricModel(p['config'],p['arm'],initialize=False).to(device)
    model.load_state_dict(ck['model'],strict=True); model.eval()
    return model,ck


def evaluate(checkpoint,manifest,partition,output,device):
    if Path(output).exists(): raise FileExistsError(output)
    model,ck=load_run(checkpoint,device); m=load_manifest(manifest)
    if ck['step']!=ck['protocol']['config']['steps']: raise ValueError('Only fixed-final model may evaluate new partitions')
    result=evaluate_model(model,select(m,partition),ck['protocol']['config'],device)
    result.update(arm=ck['protocol']['arm'],partition=partition,manifest=m['fingerprint'],
        checkpoint_sha256=sha256(checkpoint),smoke=ck['protocol']['smoke'],step=ck['step'],
        comparison_lock=digest({'config':ck['protocol']['config'],'manifest':m['fingerprint'],
            'code':ck['protocol']['source_hashes'],'smoke':ck['protocol']['smoke']}))
    atomic_json(output,result)


@torch.no_grad()
def predict(checkpoint,manifest,partition,output,device):
    """No reference depth/segmentation files are opened, even if absent on deployment host."""
    root=Path(output)
    if root.exists(): raise FileExistsError(root)
    model,ck=load_run(checkpoint,device); m=load_manifest(manifest,verify=False)
    root.mkdir(parents=True); entries=[]
    for e in select(m,partition):
        if sha256(e['rgb_path'])!=e['source_sha256']['rgb_path']: raise ValueError('RGB source changed')
        views=inputs_for_views(e,VIEWS,ck['protocol']['config']['image_hw'])
        out=model(**stack_inputs(views,device)); path=root/(digest(e['id'])[:20]+'.npz')
        np.savez_compressed(path,z=out['z'].cpu().numpy(),eligible=out['eligible'].cpu().numpy(),
            positive=(out['z']>0).cpu().numpy(),K=np.stack([v['K'].numpy() for v in views]),views=np.asarray(VIEWS))
        entries.append(dict(id=e['id'],path=str(path),sha256=sha256(path)))
    atomic_json(root/'index.json',dict(entries=entries,checkpoint_sha256=sha256(checkpoint),
        teacher_files_read=False,scope='RGB + supplied calibration/crop metadata; crop provenance retained',
        input_manifest=m['fingerprint']))


def summarize(paths,output):
    if Path(output).exists(): raise FileExistsError(output)
    runs=[read_json(p) for p in paths]
    if len({r['comparison_lock'] for r in runs})!=1: raise ValueError('Different seeds, budgets, sources or smoke modes')
    if len({r['manifest'] for r in runs})!=1 or len({r['partition'] for r in runs})!=1:
        raise ValueError('Cannot compare different manifests/partitions')
    if any(r['step']==0 for r in runs): raise ValueError('Use final evaluations, not mixed initial/final')
    by={r['arm']:r for r in runs}
    if len(by)!=len(runs): raise ValueError('One fixed run per arm; summarize seeds separately')
    comparisons={}
    contrasts={'C1-C0':{'C1':1,'C0':-1},'C2-C0':{'C2':1,'C0':-1},
        'C3-C1':{'C3':1,'C1':-1},'C3-C2':{'C3':1,'C2':-1},
        'camera_x_consistency':{'C3':1,'C1':-1,'C2':-1,'C0':1},
        'T1-T0_plane_information':{'T1':1,'T0':-1},'T2-T1_parameterization':{'T2':1,'T1':-1}}
    for name,coef in contrasts.items():
        if not set(coef)<=set(by): continue
        maps={a:{(r['id'],r['crop']):r for r in by[a]['rows']} for a in coef}
        keys=set(next(iter(maps.values())))
        if any(set(v)!=keys for v in maps.values()): raise ValueError('Paired evaluation frame identities differ')
        ds=[]
        for key in sorted(keys):
            row=next(iter(maps.values()))[key].copy()
            metric='range_mm' if row['crop']=='cross_crop' else 'mae_mm'
            row['delta']=sum(coef[a]*maps[a][key][metric] for a in coef); ds.append(row)
        comparisons[name]={v:grouped_mean_ci([r for r in ds if r['crop']==v],'delta') for v in VIEWS+('cross_crop',)}
    result={'comparisons':comparisons,'negative_delta_is_better':True,'official_ap_not_run':True,
            'source_hashes':{str(p):sha256(p) for p in paths}}
    atomic_json(output,result)
    lines=['# Camera / Crop / Plane 实验汇总','',
        '单位：mm；差值小于零为误差降低。C与T两组不得作为同初始化的跨组对照。','',
        '| Arm | Full MAE | ROI MAE | Workspace MAE | 同射线跨crop极差 |','|---|---:|---:|---:|---:|']
    for arm,r in sorted(by.items()):
        vals=[r['metrics'][v]['mae_mm']['mean'] for v in VIEWS]+[r['metrics']['cross_crop']['range_mm']['mean']]
        lines.append('| '+arm+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    lines+=['','配对差值与场景/CAD聚类区间见同名JSON。小样本区间仅作探索性描述。',
            '这些是几何实验，不是候选覆盖、抓取Q、官方AP或机器人结果。']
    Path(output).with_suffix('.md').write_text('\n'.join(lines)+'\n')


def preflight(config_path,output,device):
    if Path(output).exists(): raise FileExistsError(output)
    c=config(config_path); m=load_manifest(c['manifest'])
    checks=[]
    for e in m['entries']:
        views=inputs_for_views(e,VIEWS,c['image_hw'])
        _,_,frac=common_queries(e,views,c['image_hw'],max(c['train_rays'],c['eval_rays']),np.random.default_rng(0))
        checks.append({'id':e['id'],'common_fraction':frac,'plane_available':'T_table_from_cam' in e})
    e=select(m,'train')[0]; inp=stack_inputs(inputs_for_views(e,['full'],c['image_hw']),device)
    setup(c['seed']); a=MetricModel(c,'C0').to(device).eval()
    with torch.no_grad(): za=a(**inp)['z'].cpu()
    del a
    setup(c['seed']); b=MetricModel(c,'C1').to(device).eval()
    with torch.no_grad(): zb=b(**inp)['z'].cpu()
    error=float((za-zb).abs().max())
    if error>1e-6: raise ValueError('C0/C1 initialization parity failed: '+str(error))
    atomic_json(output,dict(config=c,manifest=m['fingerprint'],frames=checks,C0_C1_max_error_m=error,
                           initialization=b.load_audit,code=code_hashes(),production=c['backend']=='production'))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--spec',required=True); a.add_argument('--output',required=True)
    a=sub.add_parser('render'); a.add_argument('--spec',required=True); a.add_argument('--output',required=True)
    a=sub.add_parser('import-gc6d'); a.add_argument('--spec',required=True); a.add_argument('--output',required=True)
    for name in ('preflight','train'):
        a=sub.add_parser(name); a.add_argument('--config',required=True); a.add_argument('--output',required=True)
        a.add_argument('--device',default='cuda:0')
        if name=='train':
            a.add_argument('--arm',choices=ARMS,required=True); a.add_argument('--resume',action='store_true')
            a.add_argument('--smoke',action='store_true')
    for name in ('evaluate','predict'):
        a=sub.add_parser(name); a.add_argument('--checkpoint',required=True); a.add_argument('--manifest',required=True)
        a.add_argument('--partition',choices=('dev','holdout','exploratory'),required=True)
        a.add_argument('--output',required=True); a.add_argument('--device',default='cuda:0')
    a=sub.add_parser('summarize'); a.add_argument('--inputs',nargs='+',required=True); a.add_argument('--output',required=True)
    a=sub.add_parser('inspect-checkpoint'); a.add_argument('--checkpoint',required=True)
    a=p.parse_args()
    if a.command=='prepare': prepare(a.spec,a.output)
    elif a.command=='render':
        from .render import render
        render(a.spec,a.output)
    elif a.command=='import-gc6d':
        from .data import import_gc6d
        import_gc6d(a.spec,a.output)
    elif a.command=='preflight': preflight(a.config,a.output,a.device)
    elif a.command=='train': train(a.config,a.arm,a.output,a.device,a.resume,a.smoke)
    elif a.command=='evaluate': evaluate(a.checkpoint,a.manifest,a.partition,a.output,a.device)
    elif a.command=='predict': predict(a.checkpoint,a.manifest,a.partition,a.output,a.device)
    elif a.command=='summarize': summarize(a.inputs,a.output)
    else:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        print('top-level keys:',list(ck)[:40])
        for k,v in ck.items():
            if isinstance(v,dict): print(k,':',list(v)[:25])

if __name__=='__main__': main()
