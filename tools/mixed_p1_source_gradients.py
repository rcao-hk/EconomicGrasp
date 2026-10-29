#!/usr/bin/env python3
"""P1-2: exact per-head gradient norms / cross-source cosine on one GPU.

No optimizer updates, no checkpoint writes. Restores training-mode BN buffers
between forward passes; seeds sampling and dropout deterministically. Measures
actual weighted Depth / Objectness / Graspness / View / CDF / Width losses and
Total/Task. Cosines are defined ONLY for the same loss and identical shared
parameter group across GN, paired GN-Trans, and separately sampled GC6D.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from mixed_p1_historical import (
    activate_historical, load_model, make_train_sources, sample_source_indices,
)


LOSSES={
    "depth":("B: DepthReg Loss",10.),
    "objectness":("B: Objectness Loss",1.),
    "graspness":("B: Graspness Loss",10.),
    "view":("B: View Loss",100.),
    "cdf":("B: CDF Loss",1.),
    "width":("B: Width Depth Loss",10.),
    "total":("A: Overall Loss",1.),
}
SOURCES=("gn","gntrans","gc6d")
PAIRS=(("gn","gntrans"),("gn","gc6d"),("gntrans","gc6d"))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-root",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--gntrans-root",required=True)
    p.add_argument("--gc6d-root",required=True)
    p.add_argument("--filtered-scenes",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--triplets",type=int,default=3,
                   help="Number of paired GN/GNTrans frames + independent GC6D frames")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--losses",default="depth,objectness,graspness,view,cdf,width,total")
    p.add_argument("--run-train-mode",type=int,choices=(0,1),default=1,
                   help="1: training gradients with deterministic RNG and restored buffers")
    return p.parse_args()


def csv_write(path,rows):
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)


def groups_from_model(model):
    groups={
        "geometry_depth":("base.depth_net.",),
        "objectness_graspness_proposal":("base.proposal_head.",),
        "view_head":("base.view.",),
        "rayrope_group":("base.kview_grasp_module.group.",),
        "cdf_width_decoder":("base.kview_grasp_module.decoder.",),
    }
    named=[(name,param) for name,param in model.named_parameters() if param.requires_grad]
    result={}
    for group,prefixes in groups.items():
        selected=[(name,p) for name,p in named if name.startswith(prefixes)]
        if group in ("geometry_depth","objectness_graspness_proposal",
                     "view_head","rayrope_group","cdf_width_decoder") and not selected:
            raise RuntimeError(f"Trainable expected parameter group absent: {group}")
        result[group]=selected
    used={n for items in result.values() for n,p in items}
    remainder=[(n,p) for n,p in named if n not in used]
    if remainder:
        # No uncovered task parameters silently discarded from total-gradient analysis.
        result["other_trainable"]=remainder
    if any("pretrained" in n and p.requires_grad for n,p in named):
        raise RuntimeError("Frozen DAV2 backbone unexpectedly trainable")
    return result


def dot_grads(g1,g2):
    if len(g1)!=len(g2):
        raise ValueError("Gradient vectors differ in parameter length")
    dot=0.;na=0.;nb=0.;n_used=0
    for a,b in zip(g1,g2):
        if a is not None:
            na+=float(a.double().square().sum())
        if b is not None:
            nb+=float(b.double().square().sum())
        if a is not None and b is not None:
            dot+=float((a.double()*b.double()).sum())
            n_used+=1
    return dot,math.sqrt(na),math.sqrt(nb),n_used


def projected_grads(model,loss,group_defs):
    params=[p for items in group_defs.values() for _,p in items]
    if not loss.requires_grad:
        return {group:[None for _ in items] for group,items in group_defs.items()}
    grads=torch.autograd.grad(loss,params,retain_graph=False,
                              allow_unused=True,create_graph=False)
    copied=[g.detach().float().cpu().contiguous() if g is not None else None
            for g in grads]
    k=0;out={}
    for name,items in group_defs.items():
        out[name]=copied[k:k+len(items)]
        k+=len(items)
    return out


def restore_buffers(model,old):
    with torch.no_grad():
        for name,t in model.named_buffers():
            t.copy_(old[name].to(t))


def checkpoint_buffers(model):
    return {n:v.detach().cpu().clone() for n,v in model.named_buffers()}


def loss_scalar(endpoints,key,weight):
    if key not in endpoints:
        raise KeyError(f"Historical objective missing required loss {key}")
    val=endpoints[key]
    if not torch.is_tensor(val) or val.numel()!=1 or not torch.isfinite(val):
        raise ValueError(f"Invalid scalar loss {key}: {val}")
    return val * float(weight)


def run_sample(model,raw,loss_name,group_defs,epoch_seed,train_mode,loss_cfg):
    from moge_rayrope.runtime import move_batch,seed_all
    from moge_rayrope.objective import objective
    from dataset.graspnet_dataset import collate_fn
    before=checkpoint_buffers(model)
    model.train(bool(train_mode))
    seed_all(epoch_seed)
    model.zero_grad(set_to_none=True)
    try:
        if isinstance(raw,dict):
            batch=collate_fn([raw])
        else:
            raise TypeError("Expected one dataset item dict")
        device="cuda:0"
        inputs=move_batch(batch,device,labels=True)
        ep=model(inputs)
        total,task,extra=objective(ep,model.cfg,loss_cfg)
        key,scale=LOSSES[loss_name]
        loss=total if loss_name=="total" else loss_scalar(ep,key,scale)
        valid=ep["batch_grasp_cdf_valid_mask"].detach().bool()
        bins=ep["batch_grasp_cdf_bins_angle_depth"].detach()
        positives=valid&(bins>0)
        info=dict(loss=float(loss.detach()),raw_component=float(ep[key].detach()) if key in ep else None,
            valid_matched_actions=int(valid.sum()),
            positive_matched_actions=int(positives.sum()),
            zero_positive_batch=not bool(positives.any()),
            valid_queries=int(valid.any(dim=(-1,-2)).sum()))
        return projected_grads(model,loss,group_defs),info
    finally:
        model.zero_grad(set_to_none=True)
        restore_buffers(model,before)


def check_frame_pair(sources,gn_idx,trans_idx):
    a=sources["gn"][1]
    b=sources["gntrans"][1]
    pos_a=sources["gn"][2][gn_idx]
    pos_b=sources["gntrans"][2][trans_idx]
    ka=(str(a.scenename[pos_a]),int(a.frameid[pos_a]))
    kb=(str(b.scenename[pos_b]),int(b.frameid[pos_b]))
    if ka!=kb:
        raise RuntimeError(f"GN/GNTrans must have identical paired scene/frame: {ka}!={kb}")
    return ka


def main():
    a=parser()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-2 requires one CUDA GPU")
    if a.triplets<1 or a.triplets>100:
        raise ValueError("--triplets must be 1..100")
    names=[v.strip() for v in a.losses.split(",") if v.strip()]
    if len(names)!=len(set(names)) or not all(v in LOSSES for v in names):
        raise ValueError(f"Unknown/repeated losses: {names}")
    activate_historical(a.historical_root)
    model,ck,meta=load_model(a.historical_root,a.checkpoint,"cuda:0")
    from moge_rayrope.config import LossConfig
    loss_cfg=LossConfig(**ck["protocol"]["loss"])
    datasets=make_train_sources(a.graspnet_root,a.gntrans_root,a.gc6d_root,
                a.filtered_scenes,model.cfg)
    gn_idx=sample_source_indices(datasets["gn"][0],a.triplets,a.seed)
    trans_idx=gn_idx[:]  # identical paired scene/frame ordering
    gc_idx=sample_source_indices(datasets["gc6d"][0],a.triplets,a.seed+7)
    definitions=groups_from_model(model)
    parameter_groups={name:dict(tensors=len(items),
        parameters=sum(p.numel() for n,p in items),
        examples=[n for n,p in items[:3]]) for name,items in definitions.items()}
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[];source_rows=[]
    for j in range(a.triplets):
        pair_key=check_frame_pair(datasets,gn_idx[j],trans_idx[j])
        batch_idx=dict(gn=gn_idx[j],gntrans=trans_idx[j],gc6d=gc_idx[j])
        # Repeat the *same* source items for each loss; do not retain all
        # computation graphs or gigantic gradients simultaneously.
        items={}
        for source in SOURCES:
            idx=batch_idx[source]
            item=datasets[source][0][idx]
            items[source]=item
        for loss_name in names:
            grads={}
            details={}
            for source in SOURCES:
                grad,info=run_sample(model,items[source],loss_name,definitions,
                    epoch_seed=71017+j,train_mode=a.run_train_mode,loss_cfg=loss_cfg)
                grads[source]=grad
                details[source]=info
                source_rows.append(dict(triplet=j,loss=loss_name,source=source,
                    scene=str(datasets[source][1].scenename[datasets[source][2][batch_idx[source]]])
                       if source!="gc6d" else str(datasets[source][1].scenename[batch_idx[source]]),
                    **info))
            for group in definitions:
                for left,right in PAIRS:
                    dot,na,nb,used=dot_grads(grads[left][group],grads[right][group])
                    # Absent/zero parameter gradients are undefined cosine,
                    # not zero and not evidence of conflicting gradients.
                    cosine=(dot/(na*nb)) if na>0 and nb>0 else None
                    if cosine is not None and (not math.isfinite(cosine) or abs(cosine)>1.0001):
                        raise ValueError("Non-finite or out-of-range cosine")
                    rows.append(dict(triplet=j,loss=loss_name,parameter_group=group,
                        source_a=left,source_b=right,cosine=cosine,
                        dot_product=dot,grad_norm_a=na,grad_norm_b=nb,
                        coactive_param_tensors=used,
                        zero_grad_a=(na==0),zero_grad_b=(nb==0),
                        paired_gn_scene=pair_key[0],paired_gn_frame=pair_key[1]))
            print(f"[P1-2] triplet {j+1}/{a.triplets} loss {loss_name}",flush=True)
            del grads
        del items
    csv_write(out/"per_head_source_gradients.csv",rows)
    csv_write(out/"source_batch_loss_and_coverage.csv",source_rows)
    aggregated=[]
    for loss_name in names:
        for group in definitions:
            for left,right in PAIRS:
                chunk=[r for r in rows if (r["loss"]==loss_name and
                   r["parameter_group"]==group and
                   r["source_a"]==left and r["source_b"]==right)]
                values=[r["cosine"] for r in chunk if r["cosine"] is not None]
                aggregated.append(dict(loss=loss_name,parameter_group=group,
                    source_a=left,source_b=right,
                    samples=len(chunk),cosine_defined=len(values),
                    mean_cosine=float(np.mean(values)) if values else None,
                    negative_cosine_fraction=float(np.mean(np.array(values)<0)) if values else None,
                    mean_grad_norm_a=float(np.mean([r["grad_norm_a"] for r in chunk])),
                    mean_grad_norm_b=float(np.mean([r["grad_norm_b"] for r in chunk]))))
    csv_write(out/"aggregate_cosine.csv",aggregated)
    summary=dict(status="complete",model=meta,triplets=a.triplets,seed=a.seed,
        train_mode=bool(a.run_train_mode),losses=names,
        parameter_groups=parameter_groups,
        steps_optimizer=0,files=[
            "per_head_source_gradients.csv","aggregate_cosine.csv","source_batch_loss_and_coverage.csv"],
        caveats=[
            "Gradients are evaluated at the final checkpoint, not during the historical optimizer trajectory.",
            "Batch size 1 per source; DDP global batch 9 source-mixture gradient may differ.",
            "A cosine is undefined if a head does not update the selected parameter group.",
            "Zero-positive frames remain distinct from cross-domain negative cosine.",
            "No optimizer or model update; train-mode buffers restored after each forward.",
        ])
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(status="complete",summary=str(out/"summary.json"),
                          aggregates=str(out/"aggregate_cosine.csv")),indent=2))


if __name__=="__main__":
    main()
