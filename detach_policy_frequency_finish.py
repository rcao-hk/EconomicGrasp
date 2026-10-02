"""Original batch3 replay and compact figure for the measured P2 evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import checkerboard_probe as q
from detach_policy_frequency import capture,channels


def main(a):
    out=a.output;old=a.previous;prod=q.imports(out);q.seed();m=q.make_model(prod)
    base=q.bases(prod)['test_seen'];indices=[4599,4602,4605];samples=[]
    for i in indices:q.seed(1000+i);samples.append(base[i])
    inp={k:torch.stack([torch.as_tensor(s[k]) for s in samples]).cuda() for k in ('img','camera_pose_vec','camera_gravity_vec','K')}
    selected=json.loads((out/'selected_checkpoints.json').read_text());records=[];rr=[]
    for cp in (selected[0],selected[-1]):
        label='early' if cp==selected[0] else 'late'
        state=torch.load(cp['path'],map_location='cpu',weights_only=False)
        m.load_state_dict(state.get('model_state_dict',state),strict=True);del state
        arrays={}
        with q.Preserve(m),torch.no_grad():
            m.eval();hs,conv=capture(m,arrays)
            try:
                r=m.depth_net(inp['img'],camera_pose_vec=inp['camera_pose_vec'],camera_gravity_vec=inp['camera_gravity_vec'],camera_K=inp['K'],return_raw=True,return_feats=True,return_pose_aux=True)
            finally:
                for h in hs:h.remove()
            pred=r[0][0,0].cpu().numpy().copy()
        for node,arr in arrays.items():
            sp,*_=channels(arr,node,dict(checkpoint=label,batch=3,frame=247),rr)
            records.append(dict(checkpoint=label,node=node,**sp))
        if label=='late':
            original=np.load(old/'original_ply_depth.npz')['pred']
            error=float(abs(pred-original).max());assert error<=1e-6,error
    q.rows(out/'frequency_original_batch_channels.csv',rr)
    q.dump(out/'frequency_original_batch.json',dict(indices=indices,mode='eval is_training=True',original_PLY_pred_max_error_m=error,records=records))
    def read(name):return list(csv.DictReader((out/name).open()))
    layers=[r for r in read('frequency_layers.csv') if r['scene']=='scene_0117' and r['frame']=='247']
    projection=[r for r in read('frequency_projection.csv') if r['scene']=='scene_0117' and r['frame']=='247']
    phase=[r for r in read('grid_phase14.csv') if r['scene']=='scene_0117' and r['frame']=='247']
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for row,label in enumerate(('early','late')):
        nodes=['pre_relu','post_relu','raw']
        values=[float(next(r for r in layers if r['checkpoint']==label and r['node']==n)['R_B']) for n in nodes]
        axes[row,0].bar(nodes,values);axes[row,0].set_yscale('log');axes[row,0].set_ylim(1e-7,1)
        axes[row,0].set_title(f'{label}: target band / high-frequency power')
        rr2=[next(r for r in projection if r['checkpoint']==label and r['region']==region) for region in ('target_B','H_outside_B')]
        x=np.arange(2)
        axes[row,1].bar(x-.18,[float(r['sum_channel_power']) for r in rr2],.36,label='sum of weighted channel powers')
        axes[row,1].bar(x+.18,[float(r['projected_power']) for r in rr2],.36,label='power after complex sum')
        axes[row,1].set_xticks(x,['target B','other high frequencies']);axes[row,1].set_yscale('log');axes[row,1].legend(fontsize=7)
        axes[row,1].set_title(f'{label}: projection energy (logit units squared)')
        grid=np.full((14,14),np.nan)
        for r in phase:
            if r['checkpoint']==label and r['error_mean_mm']:grid[int(r['v_mod14']),int(r['u_mod14'])]=float(r['error_mean_mm'])
        im=axes[row,2].imshow(grid,cmap='coolwarm',vmin=-25,vmax=25)
        axes[row,2].set_title(f'{label}: FG interior signed error (mm)')
        axes[row,2].set_xlabel('u mod14');axes[row,2].set_ylabel('v mod14')
    fig.colorbar(im,ax=axes[:,2],shrink=.75,label='mm')
    (out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/frequency_mechanism.png',dpi=180);plt.close(fig)
    summary=json.loads((out/'frequency_mechanism.json').read_text())
    summary.update(status='pass',original_batch_replay='frequency_original_batch.json',
        mechanism_figure='figures/frequency_mechanism.png',interpretation={
            'late_target_band_ratio':float(next(r['ratio_coherent_to_incoherent'] for r in projection if r['checkpoint']=='late' and r['region']=='target_B')),
            'late_other_high_frequency_ratio':float(next(r['ratio_coherent_to_incoherent'] for r in projection if r['checkpoint']=='late' and r['region']=='H_outside_B')),
            'conclusion':'Both target-frequency reinforcement and cancellation outside the band increase relative concentration; ReLU participates in expressing higher-frequency power.',
            'causal_limit':'Forward computation evidence does not identify why training learned these representations; no decoder or pose training ablation was performed.'})
    q.dump(out/'frequency_mechanism.json',summary)
    print(json.dumps(dict(status='pass',original_batch_max_error=error)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    main(p.parse_args())
