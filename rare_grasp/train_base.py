"""Fresh common base training; numerical model/loss are inherited from main.

Only override validation: do NOT validate/choose models on GraspNet test_seen.
Keep all 5200 canonical mixed10 training images. Forty additional train-domain
views (same scenes, different frames) are monitoring-only, not generalization.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset
import train_cva_gntrans_mix_ddp as mixed
from utils.arguments import cfgs
from dataset.graspnet_dataset import collate_fn
from .core import make_protocol, write_json, BASE_SHA


class RareBaseTrainer(mixed.GNTransMixedTrainer):
    def __init__(self):
        if cfgs.checkpoint_path or getattr(cfgs,'resume',False):
            raise ValueError('Fresh base only; verified existing mix10 e19 can be bound separately')
        if cfgs.max_epoch != 20 or mixed.MIX.mix_train_fraction != .1:
            raise ValueError('Primary base requires 20 epochs e0-e19 and 10% per domain')
        if not cfgs.enable_point_rayrope or cfgs.use_obs_depth:
            raise ValueError('RGB Point RayRoPE base required; GVAR optional')
        super().__init__()
        if len(self.ORIG_TRAIN)!=2600 or len(self.TRANS_TRAIN)!=2600:
            raise RuntimeError('Unexpected mix10 sampling; must be 2600 images per domain')
        expected=make_protocol()
        datasets=[]
        for domain, full in (('original',self.ORIG_TRAIN.dataset),('gntrans',self.TRANS_TRAIN.dataset)):
            lookup={(int(str(s)[-4:]),int(a)):i for i,(s,a) in enumerate(zip(full.scenename,full.frameid))}
            requested=[r for r in expected['validation'] if r['domain']==domain]
            ids=[lookup[(r['scene'],r['anno'])] for r in requested]
            train_ids=self.ORIG_TRAIN.indices if domain=='original' else self.TRANS_TRAIN.indices
            got={(int(str(full.scenename[i])[-4:]),int(full.frameid[i])) for i in train_ids}
            want={(r['scene'],r['anno']) for r in expected['train'] if r['domain']==domain}
            if got!=want or set(ids)&set(train_ids): raise RuntimeError('Train/val manifest mismatch')
            datasets.append(Subset(full,ids))
        self.TEST_DATASET=ConcatDataset(datasets)
        self.test_sampler=(DistributedSampler(self.TEST_DATASET,num_replicas=self.world_size,
                           rank=self.rank,shuffle=False,drop_last=False) if self.distributed else None)
        self.TEST_DATALOADER=DataLoader(self.TEST_DATASET,batch_size=cfgs.batch_size,
                   shuffle=False,sampler=self.test_sampler,num_workers=0,collate_fn=collate_fn)
        self.mix_protocol.update(rare_base_main=BASE_SHA,rare_frame_protocol_sha256=expected['protocol_sha256'],
                  validation_origin='train-domain frame-held-out; same scenes',
                  original_seen_count=0,gntrans_seen_count=0,mixed_seen_count=0,
                  rank_validation_count=40,primary_checkpoint='epoch_index19, checkpoint epoch20',
                  original_seen_index_sha256=None,gntrans_seen_index_sha256=None,
                  test_labels_used_for_validation=False)
        if self.main:
            write_json(Path(cfgs.log_dir)/'gntrans_mix_protocol.json',self.mix_protocol)
            write_json(Path(cfgs.log_dir)/'rare_frames.json',expected)
            write_json(Path(cfgs.log_dir)/'rare_base_args.json',vars(cfgs))


def main():
    trainer=RareBaseTrainer()
    try:trainer.train(trainer.start_epoch)
    finally:trainer.close()


if __name__=='__main__':main()
