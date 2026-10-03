"""Pooled diagnostic statistics; never use a batch-averaged AUPRC as global AP."""
from __future__ import annotations
import numpy as np
import torch


class EpochMetrics:
    def __init__(self):
        self.loss_sums={}; self.images=0
        # candidates,pos,pred_sum,target_sum,query_n,informative_n,regret,hit,
        # depth_abs,depth_n,fg_abs,fg_n,interval_hit,interval_n,width_sum
        self.v=np.zeros(15,dtype=np.float64); self.pos=np.zeros(64); self.neg=np.zeros(64)

    @torch.no_grad()
    def update(self,ep,loss_stats):
        logits=ep['grasp_cdf_pred_angle_depth'].float(); b=logits.shape[0]
        self.images+=b
        for k,val in loss_stats.items(): self.loss_sums[k]=self.loss_sums.get(k,0.)+float(val)*b
        bins=ep['batch_grasp_cdf_bins_angle_depth'].long(); mask=ep['batch_grasp_cdf_valid_mask'].bool()
        th=torch.arange(1,logits.shape[1]+1,device=logits.device)
        target=((bins[...,None]>0)&(bins[...,None]<=th)).float().mean(-1)
        u=logits.sigmoid().mean(1); p=u[mask]; y=target[mask]; positive=bins[mask]>0
        idx=(p.clamp(0,1)*64).long().clamp_max(63)
        self.pos+=torch.bincount(idx[positive],minlength=64).cpu().numpy()
        self.neg+=torch.bincount(idx[~positive],minlength=64).cpu().numpy()
        self.v[:4]+=np.array([p.numel(),int(positive.sum()),float(p.sum()),float(y.sum())])
        uf=u.flatten(-2); yf=target.flatten(-2); m=mask.flatten(-2)
        oracle=yf.masked_fill(~m,-1).max(-1).values
        informative=(m.sum(-1)>1)&((oracle-yf.masked_fill(~m,2).min(-1).values)>1e-6)
        choice=uf.masked_fill(~m,-1).argmax(-1)
        chosen=yf.gather(-1,choice[...,None])[...,0]
        diff=(oracle-chosen)[informative]
        self.v[4:8]+=np.array([m.shape[0]*m.shape[1],int(informative.sum()),float(diff.sum()),int((diff.abs()<1e-6).sum())])
        d=ep.get('depth_map_pred'); gt=ep.get('gt_depth_m')
        if d is not None and gt is not None:
            d=d.float(); gt=gt.float()
            if d.ndim==3: d=d[:,None]
            if gt.ndim==3: gt=gt[:,None]
            if d.shape!=gt.shape: gt=torch.nn.functional.interpolate(gt,d.shape[-2:],mode='nearest')
            valid=torch.isfinite(gt)&(gt>.2)&(gt<1.)
            err=(d-torch.nan_to_num(gt)).abs()
            self.v[8:10]+=[float(err[valid].sum()),int(valid.sum())]
            fg=ep.get('objectness_label_tok')
            if fg is not None and fg.numel()==gt.numel():
                vm=valid&(fg.reshape_as(gt)>0)
                self.v[10:12]+=[float(err[vm].sum()),int(vm.sum())]
            sigma=ep.get('mr_geometry',{}).get('sigma')
            if sigma is not None:
                self.v[12:15]+=[int((err[valid]<=sigma[valid]).sum()),int(valid.sum()),float((2*sigma[valid]).sum())]

    def synchronize(self,device):
        if not torch.distributed.is_initialized(): return
        keys=[None]*torch.distributed.get_world_size()
        torch.distributed.all_gather_object(keys,sorted(self.loss_sums))
        allkeys=sorted(set(k for group in keys for k in group))
        data=np.concatenate((self.v,self.pos,self.neg,[self.images],[self.loss_sums.get(k,0.) for k in allkeys]))
        t=torch.tensor(data,dtype=torch.float64,device=device); torch.distributed.all_reduce(t)
        a=t.cpu().numpy(); self.v=a[:15]; self.pos=a[15:79]; self.neg=a[79:143]; self.images=int(a[143])
        self.loss_sums={k:float(v) for k,v in zip(allkeys,a[144:])}

    def report(self):
        def div(x,y): return float(x/y) if y else None
        v=self.v; tp=np.cumsum(self.pos[::-1]); fp=np.cumsum(self.neg[::-1]); pn=self.pos.sum(); nn=self.neg.sum()
        auc=div((self.pos*(np.cumsum(self.neg)-.5*self.neg)).sum(),pn*nn)
        auprc=div((self.pos[::-1]*tp/np.maximum(tp+fp,1)).sum(),pn)
        return dict(images=self.images,scalars={k:div(v,self.images) for k,v in self.loss_sums.items()},
            cdf_candidates=int(v[0]),cdf_positive_fraction=div(v[1],v[0]),
            utility_mean=div(v[2],v[0]),target_mean=div(v[3],v[0]),
            auroc64=auc,auprc64=auprc,informative_query_fraction=div(v[5],v[4]),
            ranking_regret=div(v[6],v[5]),top1_best_hit=div(v[7],v[5]),
            depth_mae_m=div(v[8],v[9]),foreground_depth_mae_m=div(v[10],v[11]),
            interval_coverage=div(v[12],v[13]),interval_width_m=div(v[14],v[13]))
