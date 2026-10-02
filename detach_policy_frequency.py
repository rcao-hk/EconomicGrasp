"""No-update, full-channel frequency measurements on historical snapshots."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion
import checkerboard_probe as q


def capture(m, arrays):
    head=m.depth_net.depthnet.depth_head;handles=[]
    def put(name,x):
        # Independent CPU copy is mandatory before the next inplace ReLU.
        arrays[name]=x[0].detach().float().cpu().numpy().copy()
    for name in ('refinenet1','output_conv1'):
        handles.append(getattr(head.scratch,name).register_forward_hook(lambda mod,args,x,n=name:put(n,x)))
    seq=head.scratch.output_conv2
    handles.append(seq.register_forward_pre_hook(lambda mod,args:put('resize448',args[0])))
    for i,mod in enumerate(seq):
        handles.append(mod.register_forward_hook(lambda mod,args,x,i=i:put(('pre_relu','post_relu','raw')[i],x)))
    return handles,seq[2]


def channels(arr,node,tag,rr):
    scale=arr.shape[-1]/448
    sp,f,band=q.spectrum(arr,scale=scale)
    h,w=arr.shape[-2:];win=np.outer(np.hanning(h),np.hanning(w))
    norm=h*w*np.sqrt((win**2).mean())
    fy=np.fft.fftfreq(h)*scale;fx=np.fft.fftfreq(w)*scale
    for name,target in [('f0',1/14),('2f0',2/14),('3f0',3/14),('observed',95/448)]:
        if target>=scale/2:
            rr.append(tag|dict(node=node,frequency=name,status='NA_above_native_nyquist'))
            continue
        iy=int(np.argmin(abs(fy+target)));ix=int(np.argmin(abs(fx-target)))
        region=((abs(fy[:,None]+target)<=1.5/448)&(abs(fx[None]-target)<=1.5/448))
        for ch in range(len(arr)):
            val=f[ch,iy,ix];p=abs(f[ch])**2
            candidates=np.where(region,p,-1);py,px=np.unravel_index(candidates.argmax(),candidates.shape)
            rr.append(tag|dict(node=node,channel=ch,frequency=name,status='ok',
                center_fx=float(fx[ix]),center_fy=float(fy[iy]),native_hw=[h,w],
                amplitude=float(abs(val)/norm),phase_rad=float(np.angle(val)),
                real=float(val.real/norm),imag=float(val.imag/norm),
                local_squared_amplitude=float(p[region].sum()/norm**2),
                local_peak_fx=float(fx[px]),local_peak_fy=float(fy[py])))
    return sp,f,band,norm


def main(a):
    out=a.output;old=a.previous;prod=q.imports(out);q.seed();m=q.make_model(prod)
    frames=json.loads((old/'probe_manifest.json').read_text())['frames']
    selected=json.loads((out/'selected_checkpoints.json').read_text())
    rr=[];projection=[];phase=[];layer=[]
    (out/'frequency_snapshots').mkdir(exist_ok=True)
    for cp in (selected[0],selected[-1]):
        label='early' if cp==selected[0] else 'late'
        state=torch.load(cp['path'],map_location='cpu',weights_only=False)
        m.load_state_dict(state.get('model_state_dict',state),strict=True);del state
        state_hash=q.tree_hash(m.state_dict())
        for fr in frames:
            s=torch.load(old/'inputs'/(fr['stem']+'.pt'),weights_only=False)
            input_arrays=np.load(old/'inputs'/(fr['stem']+'.npz'))
            inst=input_arrays['instance'];gt=input_arrays['gt'];rgb=input_arrays['rgb']
            arrays={};tag=dict(checkpoint=label,split=fr['split'],scene=fr['scene'],frame=fr['frame'])
            with q.Preserve(m),torch.no_grad():
                m.eval();hs,conv=capture(m,arrays)
                try:res=q.depth_forward(m,s)
                finally:
                    for h in hs:h.remove()
                pred=res[0][0,0].cpu().numpy().copy()
                weights=conv.weight.detach().cpu().numpy().reshape(-1)
                bias=float(conv.bias.detach().cpu())
            assert not np.shares_memory(arrays['pre_relu'],arrays['post_relu'])
            assert np.array_equal(np.maximum(arrays['pre_relu'],0),arrays['post_relu'])
            reconstruction=np.einsum('c,chw->hw',weights.astype(float),arrays['post_relu'].astype(float))+bias
            maxerr=float(abs(reconstruction-arrays['raw'][0]).max());assert maxerr<2e-4,maxerr
            spectra={}
            arrays.update(metric=pred[None],GT=gt[None],RGB=rgb)
            for node,arr in arrays.items():
                sp,f,band,norm=channels(arr,node,tag,rr)
                layer.append(tag|dict(node=node,**sp))
                if node in ('post_relu','raw'):spectra[node]=(f,band,norm)
            hf,band,norm=spectra['post_relu'];rf=spectra['raw'][0][0]
            contrib=weights[:,None,None]*hf
            summed=contrib.sum(0);linear_error=float(abs(summed-rf).max()/norm)
            assert linear_error<2e-5,linear_error
            fy,fx=np.meshgrid(np.fft.fftfreq(448),np.fft.fftfreq(448),indexing='ij')
            regions={'target_B':band,'H_outside_B':(np.hypot(fx,fy)>.08)&~band}
            regions.update({name:((abs(fx-t)<=1.5/448)&(abs(fy+t)<=1.5/448))|
                                 ((abs(fx+t)<=1.5/448)&(abs(fy-t)<=1.5/448))
                            for name,t in [('f0',1/14),('2f0',2/14),('3f0',3/14)]})
            for name,mask in regions.items():
                incoherent=float((abs(contrib[:,mask])**2).sum()/norm**2)
                coherent=float((abs(summed[mask])**2).sum()/norm**2)
                projection.append(tag|dict(region=name,pixel_max_error=maxerr,complex_fft_max_error=linear_error,
                    sum_channel_power=incoherent,projected_power=coherent,cross_term=coherent-incoherent,
                    ratio_coherent_to_incoherent=coherent/incoherent if incoherent else None,
                    channel_contribution_power=[float((abs(v[mask])**2).sum()/norm**2) for v in contrib]))
            fg=(inst>0)&np.isfinite(gt)&(gt>=.2)&(gt<=1.)
            interior=np.zeros_like(fg)
            for obj in np.unique(inst[fg]):interior|=binary_erosion((inst==obj)&fg,iterations=3)
            yy,xx=np.indices(gt.shape)
            for v in range(14):
                for u in range(14):
                    mask=interior&(xx%14==u)&(yy%14==v);n=int(mask.sum())
                    phase.append(tag|dict(u_mod14=u,v_mod14=v,count=n,
                        error_mean_mm=float((pred-gt)[mask].mean()*1000) if n else None,
                        error_mae_mm=float(abs(pred-gt)[mask].mean()*1000) if n else None,
                        GT_mean_m=float(gt[mask].mean()) if n else None,
                        RGB_mean=float(rgb[:,mask].mean()) if n else None))
            if fr['scene']=='scene_0117' and fr['frame']==247:
                np.savez_compressed(out/'frequency_snapshots'/f'{label}_full32.npz',
                    pre_relu=arrays['pre_relu'],post_relu=arrays['post_relu'],raw=arrays['raw'],weights=weights,bias=bias)
            q.rows(out/'frequency_channels.csv',rr);q.rows(out/'frequency_layers.csv',layer)
            q.rows(out/'frequency_projection.csv',projection);q.rows(out/'grid_phase14.csv',phase)
            print(json.dumps(tag|dict(status='done',reconstruction_max_error=maxerr)),flush=True)
        assert q.tree_hash(m.state_dict())==state_hash
    q.dump(out/'frequency_mechanism.json',dict(status='measurements_complete',updates=0,
        snapshots='independent copies before inplace ReLU; all channels measured',
        f0=[1/14,-1/14],observed=[95/448,-95/448],frequency_resolution_cpp=1/448,
        align_corners=True,DINO_32x32='NA: f0 exceeds native Nyquist in output-pixel units',
        projection=projection,
        boundaries=['Single-frame probes are separate from original three-frame reproduction.',
                    'Nonlinear harmonic proximity is not proof of common origin.',
                    '1x1 spatial projection does not create nonzero frequencies.',
                    'Phase statistics include counts and controls; position and crop remain confounders.']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);main(p.parse_args())
