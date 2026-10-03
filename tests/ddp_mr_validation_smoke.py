"""Two-rank CPU test of the trainer's actual distributed validation function.

A tiny model/loss replace unavailable CUDA main modules. Seven samples are
strided without padding: rank0 has four, rank1 three. One rank also sees no
positive CDF labels in its local shard. The pooled image count must be seven.
"""
from pathlib import Path
import os,sys,types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn,distributed as dist
from torch.utils.data import Dataset,DataLoader,Subset
from moge_rayrope.config import ModelConfig,LossConfig
from train_moge_rayrope import validate

class Data(Dataset):
    def __len__(self): return 7
    def __getitem__(self,i):
        return {'img':torch.tensor([float(i)]),'gt_depth_m':torch.full((1,2,2),.5),
                'labels':torch.ones(2,2,2,dtype=torch.long)*(1 if i%2==0 else 0)}
class Model(nn.Module):
    def __init__(self): super().__init__(); self.p=nn.Parameter(torch.tensor(.1))
    def forward(self,batch):
        b=batch['img'].shape[0]
        x=dict(batch)
        x['depth_map_pred']=torch.ones_like(x['gt_depth_m'])*(.5+self.p)
        x['grasp_cdf_pred_angle_depth']=self.p.expand(b,6,2,2,2)
        x['batch_grasp_cdf_bins_angle_depth']=x['labels']
        x['batch_grasp_cdf_valid_mask']=torch.ones_like(x['labels'],dtype=torch.bool)
        return x

def fake_loss(ep,use_cdf=True):
    depth=(ep['depth_map_pred']-ep['gt_depth_m']).square().mean()
    task=ep['grasp_cdf_pred_angle_depth'].square().mean()
    ep['A: DepthReg Loss']=depth
    return depth+task,ep

if __name__=='__main__':
    dist.init_process_group('gloo')
    try:
        fake=types.ModuleType('models.loss_economicgrasp_depth_kview_transformer'); fake.get_loss=fake_loss
        sys.modules[fake.__name__]=fake
        rank=dist.get_rank(); world=dist.get_world_size()
        loader=DataLoader(Subset(Data(),list(range(rank,7,world))),batch_size=2)
        result=validate(Model(),loader,'cpu',ModelConfig(),LossConfig())
        assert result['images']==7,result
        assert result['cdf_candidates']==7*2*2*2,result
        assert abs(result['depth_mae_m']-.1)<1e-6,result
        assert result['auroc64'] is not None,result
        if rank==0: print('MR_DDP_VALIDATION_OK images=7 candidates=56, disjoint uneven shards, pooled metrics')
    finally: dist.destroy_process_group()
