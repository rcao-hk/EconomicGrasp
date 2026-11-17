"""New 2x2 cells only: multi3 near-only and multi3 near+far DPT training.

Reuse previously accepted ROI near/wide last.pt as the other 2x2 cells,
after strict registration, initialization, step and source checks.

Important: crop=(step//3)%3, distance=(step%3), not step%3 for both:
all 9 physical distance×crop combinations are seen in the wide arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.depth_followup.common import check_checkpoint
from .distance_crosscrop_core import (
    ARMS,CROPS,MAIN_STEPS,MAIN_PIXELS,MAIN_SEED,schedule,
)
from .distance_train_p1 import (
    SCHEMA,_state_hash,_pred,_sample_pixels,render_sample,
)
from .distance_split_p1 import read_registered
from .factorial_contract import sha256,digest,write_json,verify_fixed_physics


def _train_samples(reg):
    # Load all nine images per training object, the only change relative to
    # original ROI-arm training (which loaded 3 distances × fixed_roi).
    records=[]
    for item in reg["train"]:
        source=json.loads(Path(item["manifest"]).read_text(encoding="utf-8"))
        verify_fixed_physics(source,strict_files=True)
        d={}
        for distance in (0,2,4):
            for crop in CROPS:
                d[(distance,crop)]=render_sample(source,distance,crop)
        records.append((item["object_sha"],d))
    return records


def _frozen_hash(model,head):
    """SHA on all frozen parameters (including DINO/FiLM/grasp) for audit."""
    excluded={id(p) for p in head.parameters()}
    h=hashlib.sha256()
    for name,p in sorted(model.named_parameters(),key=lambda x:x[0]):
        if id(p) in excluded:continue
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(a):
    if (a.arm not in ARMS or a.seed!=MAIN_SEED or a.steps!=MAIN_STEPS or
            a.pixels_per_step!=MAIN_PIXELS or a.lr!=2e-5 or
            a.weight_decay!=1e-4 or a.beta!=.03 or a.grad_clip!=1.):
        if not a.smoke:
            raise ValueError("Formal 2x2 requires exact historical 200×256, seed and AdamW protocol")
        if a.arm not in ARMS or a.steps<9 or a.pixels_per_step<1:
            raise ValueError("Invalid multi3 smoke")
    reg=read_registered(a.registration)
    preflight_path=Path(a.preflight) if a.preflight else None
    if not a.smoke:
        if preflight_path is None or not preflight_path.is_file():
            raise ValueError("Formal mixed-crop training requires signed historical ROI preflight")
        preflight=json.loads(preflight_path.read_text(encoding="utf-8"))
        signed=preflight.pop("manifest_sha256",None)
        if signed!=digest(preflight) or preflight.get("schema")!=(
                "physical-distance-crosscrop-preflight-v1") or not preflight.get("verified"):
            raise ValueError("Historical ROI training source preflight not verified")
        if (preflight["source_registration_sha256"]!=sha256(a.registration) or
            preflight["source_registration_identity_sha256"]!=reg["registration_sha256"]):
            raise ValueError("Actual registered source changed since historical ROI preflight")
    else:
        preflight=None
    if reg["status"]=="synthetic_smoke" and not a.smoke:
        raise ValueError("Procedural CAD smoke cannot stand in for formal CAD training")
    from rare_grasp.runtime import Runtime
    rt=Runtime(a.bound_config,a.graspnet_root,"",a.device)
    net=check_checkpoint(rt)
    head=net.depthnet.depth_head
    net.eval().requires_grad_(False)
    head.requires_grad_(True)
    head.eval()  # frozen BN stats, consistent with original ROI head training
    params=[p for p in rt.model.parameters() if p.requires_grad]
    if {id(p) for p in params}!={id(p) for p in head.parameters()}:
        raise RuntimeError("Only 10,890,305-parameter original DPT depth_head may update")
    init=_state_hash(head.state_dict())
    if preflight is not None and (
          preflight["initial_depth_head_sha256"]!=init or
          preflight["frozen_base_checkpoint_sha256"]!=rt.config["checkpoint_sha256"]):
        raise ValueError("Historical and mixed-crop DPT initial model / base checkpoint differ")
    frozen_before=_frozen_hash(rt.model,head)
    cached=_train_samples(reg)
    events=schedule(a.steps,a.seed,len(cached),a.arm,"multi3")
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use new multi-crop result directory; do not overwrite ROI runs")
    root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=a.weight_decay)
    pixel_rng=np.random.default_rng(a.seed+1701)
    history=[];trace=[]
    for step,item in enumerate(events):
        obj,view_map=cached[item["object_index"]]
        example=view_map[item["distance_id"],item["crop_policy"]]
        idx=_sample_pixels(example,a.pixels_per_step,pixel_rng)
        prediction=_pred(net,example,rt.device)
        ref=torch.as_tensor(example["ref"],dtype=torch.float32,
                            device=rt.device).flatten()
        ids=torch.as_tensor(idx,dtype=torch.long,device=rt.device)
        loss=F.smooth_l1_loss(prediction[ids],ref[ids],beta=a.beta,
                             reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite DPT head multi-crop loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(
            params,a.grad_clip,error_if_nonfinite=True)
        opt.step()
        trace.append(dict(step=step,object_sha=obj,
            distance_id=item["distance_id"],crop_policy=item["crop_policy"],
            distance_m=example["distance"],foreground_pixels=len(idx),
            source_rgb_sha256=example["source_rgb_sha256"]))
        history.append(dict(step=step+1,loss=float(loss.detach().cpu()),
                            grad_norm=float(norm.detach().cpu())))
        if (step+1)%max(1,a.steps//10)==0:
            print(f"[multi3 {a.arm}] {step+1}/{a.steps} "
                  f"loss={history[-1]['loss']:.5f}",flush=True)
    frozen_after=_frozen_hash(rt.model,head)
    if frozen_before!=frozen_after:
        raise RuntimeError("Frozen DINO/pose-FiLM/GVAR/grasp parameter SHA changed")
    protocol=dict(schema=SCHEMA,arm=a.arm,
       crop_policy="multi3",
       train_crop_indices=[v["crop_policy"] for v in trace],
       train_distance_indices=[v["distance_id"] for v in trace],
       train_object_sequence=[v["object_sha"] for v in trace],
       distance_schedule=[0,0,0] if a.arm=="near_only" else [0,2,4],
       train_registration_sha256=sha256(a.registration),
       historical_roi_preflight_sha256=sha256(a.preflight) if a.preflight else None,
       registration_identity_sha256=reg["registration_sha256"],
       base_checkpoint_sha256=rt.config["checkpoint_sha256"],
       frozen_base_config_digest=rt.config_digest,
       initial_depth_head_sha256=init,
       frozen_parameters_sha256_before=frozen_before,
       frozen_parameters_sha256_after=frozen_after,
       steps=a.steps,pixels_per_step=a.pixels_per_step,
       total_supervised_pixels=a.steps*a.pixels_per_step,
       learning_rate=a.lr,weight_decay=a.weight_decay,
       smooth_l1_beta=a.beta,gradient_clip=a.grad_clip,
       seed=a.seed,optimizer="AdamW",
       trainable_module="depth_net.depthnet.depth_head ONLY",
       trainable_params=sum(p.numel() for p in params),
       training_status="synthetic_smoke" if a.smoke else "preregistered_CAD_holdout",
       source_registration_status=reg["status"],
       update_step_crop_schedule="CROPS[(step//3)%3]",
       update_step_distance_schedule="near=0;wide=(0,2,4)[step%3]",
       source_train_CAD_objects=[r["object_sha"] for r in reg["train"]],
       no_grasp_training=True,
       no_evaluation_object_used=True,
       training_code_sha256=sha256(__file__),
       limitations=[
          "This is CAD-synthetic DPT decoder fitting, not real GC6D deployment",
          "The original 4 CADs had already been inspected; not untouched",
          "No full Grasp Head/FiLM/DINO training and no grasp AP",
          "Balanced frequency is not evidence of generalization to arbitrary object/material/camera",
       ])
    weight=root/"last.pt"
    torch.save(dict(schema=SCHEMA,kind="DPT_head_only",
                    state=head.state_dict(),protocol=protocol),weight)
    with (root/"sample_schedule.json").open("w",encoding="utf-8") as f:
        json.dump(trace,f,indent=2)
    with (root/"history.json").open("w",encoding="utf-8") as f:
        json.dump(history,f,indent=2)
    result=dict(schema=SCHEMA,kind="multicrop_finished",
        protocol=protocol,complete=True,
        weight_sha256=sha256(weight),
        output_head_sha256=_state_hash(head.state_dict()),
        sample_schedule_sha256=sha256(root/"sample_schedule.json"),
        history_sha256=sha256(root/"history.json"),
        loss_first=history[0]["loss"],loss_last=history[-1]["loss"])
    write_json(root/"complete.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--preflight",default=None,
                   help="Signed actual historical ROI near/wide source and model validation")
    p.add_argument("--bound-config",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--seed",type=int,default=MAIN_SEED)
    p.add_argument("--steps",type=int,default=MAIN_STEPS)
    p.add_argument("--pixels-per-step",type=int,default=MAIN_PIXELS)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--beta",type=float,default=.03)
    p.add_argument("--grad-clip",type=float,default=1.)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output",required=True)
    p.add_argument("--smoke",action="store_true")
    run(p.parse_args())


if __name__=="__main__":
    main()
