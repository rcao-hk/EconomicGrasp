"""Plots of saved arrays only; no training or model mutation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def run(out):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    labels=['epoch0','epoch5','epoch15','epoch16_target'];stem='test_seen_scene_0117_0247'
    data=[np.load(out/'predictions'/f'{l}_{stem}.npz') for l in labels]
    inputs=np.load(out/'inputs'/f'{stem}.npz');rgb=np.clip(inputs['rgb'].transpose(1,2,0)*[.229,.224,.225]+[.485,.456,.406],0,1)
    fig,axs=plt.subplots(3,4,figsize=(13,10),constrained_layout=True)
    for i,(l,d) in enumerate(zip(labels,data)):
        axs[0,i].imshow(d['pred']*1000,vmin=200,vmax=1000,cmap='viridis');axs[0,i].set_title(l+' depth [mm]')
        e=(d['pred']-d['gt'])*1000
        im=axs[1,i].imshow(e,vmin=-100,vmax=100,cmap='RdBu_r');axs[1,i].set_title('Error [mm]; shared ±100')
        h=e-gaussian_filter(e,3);w=np.outer(np.hanning(448),np.hanning(448));f=np.fft.fftshift(np.fft.fft2(h*w));p=abs(f)**2/(448**4*(w*w).mean())
        sp=axs[2,i].imshow(np.log10(p+1e-12),origin='lower',extent=(-.5,.5,-.5,.5),vmin=-8,vmax=3,cmap='magma')
        axs[2,i].set_title('log10 power [mm²], shared scale');axs[2,i].set_xlabel('fx [cycles/pixel]')
        for sx,sy in [(1,-1),(-1,1)]:
            from matplotlib.patches import Rectangle
            axs[2,i].add_patch(Rectangle((sx*.21205-.04,sy*.21205-.04),.08,.08,fill=False,edgecolor='cyan',linewidth=.8))
        for j in (0,1):axs[j,i].axis('off')
    fig.colorbar(im,ax=axs[1,:],shrink=.7);fig.colorbar(sp,ax=axs[2,:],shrink=.7)
    fig.savefig(figdir/'target_evolution.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
    axs[0].imshow(rgb);axs[0].set_title('Actual preprocessed RGB')
    im=axs[1].imshow(inputs['gt']*1000,vmin=200,vmax=1000);axs[1].set_title('GT [mm]')
    axs[2].imshow(inputs['instance']>0);axs[2].set_title('GT foreground')
    for l,d in zip(labels,data):axs[3].plot(np.arange(50,65),d['pred'][224,50:65]*1000,label=l)
    axs[3].plot(np.arange(50,65),inputs['gt'][224,50:65]*1000,'k--',label='GT');axs[3].set(xlabel='u at v=224',ylabel='depth [mm]');axs[3].legend(fontsize=7)
    for ax in axs[:3]:ax.axis('off')
    fig.savefig(figdir/'input_and_profile.png',dpi=160);plt.close(fig)
    rr=list(csv.DictReader(open(out/'depth_spectrum.csv')))
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for split,marker in [('train','o'),('test_seen','s')]:
        for ax,key,title in zip(axs,['fg_mae_mm','A_B_mm','interior_adjacent_error_mm'],['Foreground MAE [mm]','Fixed-band amplitude [mm]','Same-instance adjacent error [mm]']):
            vals=[[float(r[key]) for r in rr if r['checkpoint']==l and r['split']==split] for l in labels]
            for j,v in enumerate(vals):ax.scatter([j]*len(v),v,s=9,alpha=.3)
            ax.plot(range(4),[np.mean(v) for v in vals],marker=marker,label=split+' equal-frame mean');ax.set_title(title);ax.set_xticks(range(4),['e0','e5','e15','e16']);ax.legend(fontsize=7)
    fig.savefig(figdir/'cross_frame_metrics.png',dpi=160);plt.close(fig)
    grad=out/'gradients'/'epoch16_target_test_seen_all.npz'
    if grad.exists():
        g=np.load(grad);names=['depth','view','cdf','width'];arr=[g[n][0,0] for n in names]
        cap=max(np.quantile(abs(a),.999) for a in arr);fig,axs=plt.subplots(2,4,figsize=(13,7),constrained_layout=True)
        powers=[]
        for a in arr:
            h=a-gaussian_filter(a,3);powers.append(abs(np.fft.fftshift(np.fft.fft2(h*np.outer(np.hanning(448),np.hanning(448)))))**2)
        vmax=max(np.log10(p+1e-30).max() for p in powers)
        for i,(n,a,p) in enumerate(zip(names,arr,powers)):
            im=axs[0,i].imshow(a,vmin=-cap,vmax=cap,cmap='RdBu_r');axs[0,i].set_title(n+' weighted dL/dz [1/m]')
            sp=axs[1,i].imshow(np.log10(p+1e-30),origin='lower',extent=(-.5,.5,-.5,.5),vmin=vmax-12,vmax=vmax,cmap='magma');axs[1,i].set_title('log10 FFT power, shared scale')
            axs[0,i].axis('off')
        fig.colorbar(im,ax=axs[0,:],shrink=.7);fig.colorbar(sp,ax=axs[1,:],shrink=.7)
        fig.suptitle('One actual train-mode forward; local batch=2; gradient display clips at shared 99.9% cap')
        fig.savefig(figdir/'actual_loss_gradients.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(figures=[p.name for p in figdir.glob('*.png')],status='complete')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
