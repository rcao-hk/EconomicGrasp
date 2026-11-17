"""P1: two NEW 600-step distance-wide controls, ROI600 and Multi3-600.

Both start from the historical frozen MIX50 e15 model and exactly identical
CAD train objects, object sequence, distance sequence, AdamW parameters,
supervised foreground pixel budget and seed. Only crop design differs.

600-step Multi3 receives exactly 200 updates per crop. Original accepted
ROI200/Multi200 weights remain untouched; no previous optimizer is resumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.depth_followup.common import check_checkpoint
from .distance_budget_core import budget_schedule,exposure_summary,assert_600_pair
from .distance_budget_preflight import read_preflight
from .distance_train_p1 import SCHEMA,_state_hash,_pred,_sample_pixels,render_sample
from .distance_crosscrop_train import _frozen_hash
from .distance_split_p1 import read_registered
from .factorial_contract import sha256,write_json,verify_fixed_physics

STEPS=600
PIXELS=256
SEED=20261117
LR=2e-5
WD=1e-4
BETA=.03
CLIP=1.


def _samples(reg,design):
    from .distance_crosscrop_core import CROPS
    crops=("fixed_roi",) if design=="roi" else CROPS
    result=[]
    for item in reg["train"]:
        path=Path(item["manifest"])
        source=json.loads(path.read_text(encoding="utf-8"))
        verify_fixed_physics(source,strict_files=True)
        entries={(di,crop):render_sample(source,di,crop)
                 for di in (0,2,4) for crop in crops}
        result.append((item["object_sha"],entries))
    return result


def run(a):
    if a.design not in ("roi","multi3"):
        raise ValueError("Only preregistered 600-step ROI and Multi3")
    if (a.prefix_tol!=1e-4 or
        a.steps!=STEPS or a.pixels_per_step!=PIXELS or a.seed!=SEED or
        a.lr!=LR or a.weight_decay!=WD or a.beta!=BETA or a.grad_clip!=CLIP):
        raise ValueError("600step budget controls must retain exact original AdamW/foreground-pixel contract")
    reg=read_registered(a.registration)
    if reg["status"]=="synthetic_smoke":
        raise ValueError("Training600 needs registered actual CAD, not procedural smoke")
    preflight=read_preflight(a.preflight,a.registration)
    if [r["object_sha"] for r in reg["train"]]!=preflight["historical_CAD_train"]:
        raise ValueError("Training CAD changed from accepted 200-step study")
    from rare_grasp.runtime import Runtime
    rt=Runtime(a.bound_config,a.graspnet_root,"",a.device)
    net=check_checkpoint(rt)
    head=net.depthnet.depth_head
    net.eval().requires_grad_(False)
    head.eval().requires_grad_(True)
    params=list(head.parameters())
    all_trainable=[x for x in rt.model.parameters() if x.requires_grad]
    if {id(p) for p in params}!={id(p) for p in all_trainable}:
        raise RuntimeError("Only original DPT depth_head may be trained")
    initial=_state_hash(head.state_dict())
    if (initial!=preflight["initial_depth_head_sha256"] or
        rt.config["checkpoint_sha256"]!=preflight["base_checkpoint_sha256"]):
        raise ValueError("600-step model not initialized from exact accepted ROI/Multi original base")
    frozen_initial=_frozen_hash(rt.model,head)
    prepared=_samples(reg,a.design)
    event=budget_schedule(a.design,STEPS,SEED,len(prepared))
    # Same CAD object and physical distance per optimizer step in the ROI and
    # Multi3-600 arms. Only crop is different.
    other=budget_schedule("multi3" if a.design=="roi" else "roi",STEPS,SEED,len(prepared))
    assert_600_pair(event,other) if a.design=="roi" else assert_600_pair(other,event)
    exposure=exposure_summary(event,PIXELS)
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    optimizer=torch.optim.AdamW(params,lr=LR,weight_decay=WD)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Never overwrite accepted 200step results or other 600step arm")
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED+1701)
    history=[];source_events=[]
    prefix_acceptance=None
    old200_key="roi200" if a.design=="roi" else "multi200"
    old200=preflight["old_head_sources"][old200_key]
    # The first 200 events of a fresh 600step run MUST reproduce the
    # original same-seed training, up to preregistered FP numerical tolerance.
    old_checkpoint=torch.load(old200["path"],map_location="cpu",weights_only=True)
    old_state=old_checkpoint["state"]
    for t,e in enumerate(event):
        obj,examples=prepared[e["object_index"]]
        sample=examples[e["distance_id"],e["crop_policy"]]
        index=_sample_pixels(sample,PIXELS,rng)
        p=_pred(net,sample,rt.device)
        label=torch.as_tensor(sample["ref"],dtype=torch.float32,
                              device=rt.device).flatten()
        ids=torch.as_tensor(index,dtype=torch.long,device=rt.device)
        loss=F.smooth_l1_loss(p[ids],label[ids],beta=BETA,reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite 600step DPT distance×crop loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm=torch.nn.utils.clip_grad_norm_(params,CLIP,error_if_nonfinite=True)
        optimizer.step()
        if t+1==200:
            current=head.state_dict()
            if set(current)!=set(old_state):
                raise RuntimeError("200step DPT head parameter structure changed")
            max_abs=0.;total_diff_sq=0.;total_ref_sq=0.
            for key,val in current.items():
                old=old_state[key].detach().cpu().float()
                current_value=val.detach().cpu().float()
                if old.shape!=current_value.shape:
                    raise RuntimeError("Historical 200step head tensor shape changed")
                delta=current_value-old
                max_abs=max(max_abs,float(delta.abs().max()))
                total_diff_sq+=float((delta.double()**2).sum())
                total_ref_sq+=float((old.double()**2).sum())
            prefix_acceptance=dict(
                prefix_step=200,max_param_absdiff=max_abs,
                relative_param_l2=float((total_diff_sq/max(total_ref_sq,1e-24))**.5),
                historical_200_last_sha256=old200["sha256"],
                historical_200_state_sha256=_state_hash(old_state),
                new_200_state_sha256=_state_hash(current),
                preregistered_max_param_tol=a.prefix_tol)
            if max_abs>a.prefix_tol:
                raise RuntimeError(
                    f"First200 steps do NOT reproduce accepted 200step {a.design} "
                    f"within preregistered max parameter tolerance {a.prefix_tol}; "
                    f"actual={max_abs:.7g}")
        history.append(dict(step=t+1,loss=float(loss.detach().cpu()),
                            gradient_norm=float(gnorm.detach().cpu())))
        source_events.append(dict(step=t,object_sha=obj,distance_id=e["distance_id"],
             distance_m=float(sample["distance"]),crop_policy=e["crop_policy"],
             foreground_pixels=PIXELS,
             rgb_sha256=sample["source_rgb_sha256"]))
        if (t+1)%60==0:
            print(f"[budget600/{a.design}] {t+1}/{STEPS} "
                  f"loss={history[-1]['loss']:.5f}",flush=True)
    frozen_final=_frozen_hash(rt.model,head)
    if frozen_initial!=frozen_final:
        raise RuntimeError("Unexpected DINO/pose-FiLM/grasp weights change in budget600 training")
    protocol=dict(schema=SCHEMA,arm="near_mid_far",
        crop_policy="fixed_roi" if a.design=="roi" else "multi3",
        train_crop_indices=[s["crop_policy"] for s in source_events],
        train_distance_indices=[s["distance_id"] for s in source_events],
        train_object_sequence=[s["object_sha"] for s in source_events],
        distance_schedule=[0,2,4],
        train_registration_sha256=sha256(a.registration),
        registration_identity_sha256=reg["registration_sha256"],
        base_checkpoint_sha256=rt.config["checkpoint_sha256"],
        frozen_base_config_digest=rt.config_digest,
        initial_depth_head_sha256=initial,
        frozen_parameters_sha256_before=frozen_initial,
        frozen_parameters_sha256_after=frozen_final,
        historical_200step_preflight_sha256=sha256(a.preflight),
        first_200_checkpoint_reproduction=prefix_acceptance,
        steps=STEPS,pixels_per_step=PIXELS,
        total_supervised_pixels=STEPS*PIXELS,
        learning_rate=LR,weight_decay=WD,smooth_l1_beta=BETA,
        gradient_clip=CLIP,seed=SEED,optimizer="AdamW",
        trainable_module="depth_net.depthnet.depth_head ONLY",
        trainable_params=sum(x.numel() for x in params),
        training_status="preregistered_CAD_holdout",
        source_registration_status=reg["status"],
        training_design=a.design,exposure=exposure,
        no_grasp_training=True,model_selection="fixed_last_600",
        source_code_sha256=sha256(__file__),
        limitations=[
           "600step heads start from base, not 200step last.pt; AdamW not resumed",
           "Multi600 per-crop=200 vs ROI200 does NOT control total compute",
           "ROI600 vs Multi600 controls total compute, but ROI exposure differs",
           "Object/scene/material generalization not proven by two previously inspected test CAD",
           "No real GC6D/GraspNet grasp AP or robot evaluation"
        ])
    last=root/"last.pt"
    torch.save(dict(schema=SCHEMA,kind="DPT_head_only",
                    state=head.state_dict(),protocol=protocol),last)
    (root/"loss_history.json").write_text(
        json.dumps(history,indent=2),encoding="utf-8")
    (root/"sample_schedule.json").write_text(
        json.dumps(source_events,indent=2),encoding="utf-8")
    done=dict(schema=SCHEMA,status="complete",protocol=protocol,
        weight_sha256=sha256(last),
        last_head_parameter_sha256=_state_hash(head.state_dict()),
        loss_history_sha256=sha256(root/"loss_history.json"),
        sample_schedule_sha256=sha256(root/"sample_schedule.json"))
    write_json(root/"complete.json",done)
    return done


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--preflight",required=True)
    p.add_argument("--bound-config",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--design",choices=("roi","multi3"),required=True)
    p.add_argument("--prefix-tol",type=float,default=1e-4,
                   help="Fixed pre-registered max-absolute parameter mismatch of 600step first200 vs old last200")
    p.add_argument("--steps",type=int,default=STEPS)
    p.add_argument("--pixels-per-step",type=int,default=PIXELS)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--lr",type=float,default=LR)
    p.add_argument("--weight-decay",type=float,default=WD)
    p.add_argument("--beta",type=float,default=BETA)
    p.add_argument("--grad-clip",type=float,default=CLIP)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output",required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
