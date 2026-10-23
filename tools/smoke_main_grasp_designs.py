"""Exercise the production mixed trainer for two batches, including all sources.

Pass the normal triple-trainer CLI, with a one-scene GC6D manifest for smoke.
Works with python or torchrun; logs/checkpoints never share a production run.
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader, Subset
import train_cva_gntrans_gc6d_mix_ddp as mixed
from dataset.graspnet_dataset import collate_fn


def main():
    trainer = mixed.GraspNetGNTransGC6DMixedTrainer()
    try:
        n = mixed.cfgs.batch_size * 2
        a, b = len(trainer.ORIG_TRAIN), len(trainer.TRANS_TRAIN)
        # Every rank sees all three domains; rank offsets exercise other frames.
        rows = [(0, a, a + b)[i % 3] + (i // 3 + trainer.rank) % 4
                for i in range(n)]
        trainer.train_sampler = None
        trainer.TRAIN_DATALOADER = DataLoader(
            Subset(trainer.TRAIN_DATASET, rows), batch_size=mixed.cfgs.batch_size,
            num_workers=mixed.cfgs.num_workers, collate_fn=collate_fn)
        if os.environ.get('SMOKE_AUDIT_ONLY') == '1':
            assert not trainer.distributed
            base = mixed.pair.base
            batch = next(iter(trainer.TRAIN_DATALOADER))
            batch = base.move_batch_to_device(batch, trainer.device, use_cdf=True)
            ep = trainer.net(batch)
            loss, ep = base.get_loss_economicgrasp(ep, use_cdf=True)
            task_loss = loss - ep['A: DepthReg Loss']
            grad = torch.autograd.grad(task_loss, ep['depth_map_pred'],
                                       allow_unused=True)[0]
            grad_max = 0.0 if grad is None else float(grad.abs().max())
            assert grad_max == 0.0, grad_max
            del ep, loss, task_loss, batch, grad
            trainer.TEST_DATALOADER = DataLoader(
                Subset(trainer.TEST_DATASET, [0, 1, len(trainer.ORIG_VAL),
                                             len(trainer.ORIG_VAL) + 1]),
                batch_size=mixed.cfgs.batch_size,
                num_workers=mixed.cfgs.eval_num_workers, collate_fn=collate_fn)
            val = trainer.evaluate_one_epoch(10)
            assert math.isfinite(val)
            result = dict(task_to_depth_grad_max=grad_max, eval_epoch=10,
                          validation_loss=val, validation_workers=mixed.cfgs.eval_num_workers)
            Path(mixed.cfgs.log_dir, 'gradient_validation.json').write_text(json.dumps(result, indent=2))
            print('[GRADIENT/VALIDATION PASS] ' + json.dumps(result), flush=True)
            return
        torch.cuda.reset_peak_memory_stats()
        loss = trainer.train_one_epoch(0)
        assert math.isfinite(loss), loss
        result = dict(rank=trainer.rank, batches=2, batch_size=mixed.cfgs.batch_size,
                      loss=loss, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                      design=trainer.unwrap_model().grasp_design)
        print('[SMOKE PASS] ' + json.dumps(result), flush=True)
        Path(mixed.cfgs.log_dir, f'smoke_rank{trainer.rank}.json').write_text(
            json.dumps(result, indent=2))
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
