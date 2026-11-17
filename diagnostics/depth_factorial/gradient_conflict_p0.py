"""P0: NO-UPDATES same-world-ray cross-crop DPT gradient interference.

Three matched RGB crops for each *identical* rendered physical surface pixel.
Different scores/labels do not choose the pixels. Measure full-head and
decoder-block gradient cosine, first-order transfer, and conditioned loss.
The train CADs ONLY are used; previously inspected eval objects are forbidden.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.depth_followup.common import check_checkpoint
from diagnostics.depth_followup.multiview import to_crop
from .distance_split_p1 import read_registered
from .distance_train_p1 import _state_hash,render_sample,_pred
from .factorial_contract import source_image,sha256,write_json,verify_fixed_physics
from .gradient_conflict_core import (
    CROPS,DISTANCE_INDICES,PAIR_ORDER,deterministic_seed,shared_raw_points,
    per_group_vectors,gradient_pair,summarize_cases
)

SCHEMA="crosscrop-same-ray-dpt-gradient-v1"


def _checkpoint(path,expect_crop,reg_sha,initial_head_sha,base_sha):
    data=torch.load(path,map_location="cpu",weights_only=False)
    if data.get("schema")!="physical-distance-equal-budget-dpt-head-v1" or \
       data.get("kind")!="DPT_head_only":
        raise ValueError("Not an accepted original DPT-head-only checkpoint")
    p=data["protocol"]
    if (p.get("arm")!="near_mid_far" or
        p.get("crop_policy")!=expect_crop or
        p.get("steps")!=200 or p.get("pixels_per_step")!=256 or
        p.get("train_registration_sha256")!=reg_sha or
        p.get("initial_depth_head_sha256")!=initial_head_sha or
        p.get("base_checkpoint_sha256")!=base_sha or
        p.get("training_status")=="synthetic_smoke"):
        raise ValueError("Wrong 200-step trained source for frozen gradient audit")
    return data["state"]


def _loss_and_grad(net,head,sample,u,v,reference,device,beta):
    p=_pred(net,sample,device).reshape(1,1,448,448)
    uc,vc=to_crop(u,v,sample["box"],(448,448))
    if not np.all((uc>=1)&(uc<=446)&(vc>=1)&(vc<=446)):
        raise RuntimeError("Original physical ray outside one crop decoder frame")
    grid=np.stack([uc*2/447-1,vc*2/447-1],axis=-1)
    grid=torch.as_tensor(grid,dtype=torch.float32,device=device).reshape(1,1,-1,2)
    sampled=F.grid_sample(p,grid,mode="bilinear",align_corners=True)[0,0,0,:]
    ref=torch.as_tensor(reference,dtype=torch.float32,device=device)
    loss=F.smooth_l1_loss(sampled,ref,beta=beta,reduction="mean")
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite same-world-ray rendered depth supervision")
    params=list(head.parameters())
    named=list(head.named_parameters())
    if [id(x) for x in params]!=[id(p) for _,p in named]:
        raise RuntimeError("Parameter names/gradients not aligned")
    g=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=False)
    arr=[np.zeros(tuple(p.shape),dtype=np.float32) if v is None else
         v.detach().float().cpu().numpy().copy() for p,v in zip(params,g)]
    grouped=per_group_vectors([n for n,_ in named],arr)
    return float(loss.detach().cpu()),grouped


def _csv(path,rows):
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def run(a):
    if a.points_per_case<32 or a.seed<0 or a.beta!=.03:
        raise ValueError("P0 gradient must use pre-registered shared rays/SmoothL1")
    reg=read_registered(a.registration)
    if reg.get("status")=="synthetic_smoke" and not a.smoke:
        raise ValueError("Formal P0 cannot use procedural smoke CAD")
    if len(reg["train"])!=2:
        raise ValueError("Primary P0 gradient requires the same two signed train CADs")
    from rare_grasp.runtime import Runtime
    rt=Runtime(a.bound_config,a.graspnet_root,"",a.device)
    net=check_checkpoint(rt)
    head=net.depthnet.depth_head
    net.eval().requires_grad_(False)
    head.eval().requires_grad_(True)
    initial=_state_hash(head.state_dict())
    base_state={name:x.detach().cpu().clone() for name,x in head.state_dict().items()}
    states=[("base",base_state,None)]
    if a.roi_wide_last or a.multi_wide_last:
        if not (a.roi_wide_last and a.multi_wide_last and a.roi_wide_sha and a.multi_wide_sha):
            raise ValueError("Both signed ROI-wide and Multi-wide 200-step heads required")
        for label,path,expected_hash,crop in (
            ("roi_wide",a.roi_wide_last,a.roi_wide_sha,"fixed_roi"),
            ("multi_wide",a.multi_wide_last,a.multi_wide_sha,"multi3")
        ):
            if not Path(path).is_file() or sha256(path)!=expected_hash:
                raise ValueError(f"Missing/changed fixed head {label}")
            state=_checkpoint(path,crop,sha256(a.registration),initial,
                              rt.config["checkpoint_sha256"])
            states.append((label,state,expected_hash))
    elif not a.smoke:
        raise ValueError("Formal gradient diagnostic requires both accepted trained-wide heads")
    cached=[]
    for item in reg["train"]:
        source=json.loads(Path(item["manifest"]).read_text(encoding="utf-8"))
        verify_fixed_physics(source)
        cached.append((item,source))
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use new output; never overwrite historical CAD results")
    root.mkdir(parents=True,exist_ok=True)
    rows=[];sources=[]
    for state_name,state,state_sha in states:
        head.load_state_dict(state,strict=True)
        state_initial=_state_hash(head.state_dict())
        for obj,source in cached:
            for distance_id in DISTANCE_INDICES:
                view=source["views"][distance_id]
                _,raw_ref,raw_mask=source_image(view)
                samples={crop:render_sample(source,distance_id,crop) for crop in CROPS}
                boxes={crop:samples[crop]["box"] for crop in CROPS}
                seed=deterministic_seed(obj["object_sha"],distance_id,a.seed)
                u,v,z=shared_raw_points(raw_mask,raw_ref,boxes,
                    max_points=a.points_per_case,seed=seed)
                losses={};grads={}
                for crop,sample in samples.items():
                    loss,gradient=_loss_and_grad(
                        net,head,sample,u,v,z,rt.device,a.beta)
                    losses[crop]=loss
                    grads[crop]=gradient
                for crop_a,crop_b in PAIR_ORDER:
                    layer_names=sorted(set(grads[crop_a])&set(grads[crop_b]))
                    for layer in ("all",*layer_names):
                        if layer=="all":
                            x=np.concatenate([grads[crop_a][q] for q in layer_names])
                            y=np.concatenate([grads[crop_b][q] for q in layer_names])
                        else:
                            x=grads[crop_a][layer];y=grads[crop_b][layer]
                        outcome=gradient_pair(x,y)
                        rows.append(dict(state=state_name,
                            object_sha=obj["object_sha"],distance_id=distance_id,
                            distance_m=float(view["distance_m"]),
                            crop_a=crop_a,crop_b=crop_b,layer=layer,
                            loss_a=losses[crop_a],loss_b=losses[crop_b],
                            physical_rays=len(u),**outcome))
                sources.append(dict(state=state_name,object_sha=obj["object_sha"],
                    distance_id=distance_id,physical_rays=len(u),
                    reference_render_sha256=view["files"]["depth"]["sha256"],
                    raw_pixel_sha256=__import__("hashlib").sha256(
                        np.ascontiguousarray(np.stack([u,v],axis=1)).tobytes()).hexdigest()))
        if _state_hash(head.state_dict())!=state_initial:
            raise RuntimeError("Backprop gradient analysis inadvertently updated parameters")
        print(f"[P0 gradient] {state_name}: {len(cached)} train CAD×3 distances",flush=True)
    _csv(root/"gradient_pair_by_case.csv",rows)
    summary=dict(schema=SCHEMA,status="smoke" if a.smoke else "complete",
        registration_sha256=sha256(a.registration),
        base_checkpoint_sha256=rt.config["checkpoint_sha256"],
        initial_head_sha256=initial,
        weight_sources={k:v for k,_,v in states},
        sampled_sources=sources,
        gradient_cosines=summarize_cases(rows),
        no_optimizer_steps=True,
        training_CAD_only=True,eval_object_labels_used=False,
        matched_world_pixels=True,loss="SmoothL1 beta=0.03",
        no_grasp_AP=True,
        interpretation=[
            "Negative g_i dot g_j is a local, checkpoint-dependent first-order conflict, not a converged optimization theorem",
            "Reference depth from real physically translated CAD source, not RGB model input",
            "Same rays, same z target and exact parameter set across all three crops",
            "Two already inspected CAD training objects; do not claim population-level gradient conflict",
            "Scene/shape/motion and crop effects coexist; no module-level cause uniquely proven",
        ])
    write_json(root/"summary.json",summary)
    (root/"REPORT_CN.md").write_text(
        "# P0 同物理射线的多Crop DPT梯度一致性\n\n"
        "使用已验收训练CAD2/5；固定0.35/0.55/0.85m、同物理像素、同真值、同DPT参数。\n"
        "无optimizer.step、无CAD评价、无新AP。\n\n"
        "| 参数状态/裁剪对 | 样本组 | Mean Cosine | 负Cosine比例 |\n"
        "|---|---:|---:|---:|\n"+
        "\n".join(f"| {k} | {v['cases']} | {v['mean_cosine']} | {v['negative_fraction']} |"
                  for k,v in summary["gradient_cosines"].items())+
        "\n\n梯度冲突只反映局部一阶更新方向，不能证明长期训练失败因果。\n",
        encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--bound-config",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--roi-wide-last",default=None)
    p.add_argument("--roi-wide-sha",default=None)
    p.add_argument("--multi-wide-last",default=None)
    p.add_argument("--multi-wide-sha",default=None)
    p.add_argument("--points-per-case",type=int,default=128)
    p.add_argument("--seed",type=int,default=20261117)
    p.add_argument("--beta",type=float,default=.03)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--output",required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
