"""P1b fixed-budget near-only vs near+far DPT-head-only fine-tuning.

Each ARM begins from the same accepted frozen MIX50 GVAR e15 checkpoint.
The DINOv2 encoder, global-FiLM adapter, grasp network and all non-DPT-head
parameters stay FROZEN. Only depth_net.depthnet.depth_head receives gradients.

Matched budget: identical preregistered CAD object sequence, optimizer steps,
per-step supervised foreground pixel count, crop policy, optimizer and seeds.
The sole systematic intervention is physical camera distance in the labeled
render source (near: 0; near+far: 0/2/4). No benchmark/AP data are accessed.
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
from .distance_split_p1 import read_registered
from .factorial_contract import (CROP_POLICIES,sha256,digest,write_json,
                                  source_image)
from .factorial_run import crop_inputs,_workspace_box,_boxed

SCHEMA="physical-distance-equal-budget-dpt-head-v1"
ARMS=("near_only","near_mid_far")
DISTANCE_CHOICES={"near_only":(0,0,0),"near_mid_far":(0,2,4)}


def _state_hash(state):
    """Parameter-value identity independent of Torch checkpoint ZIP metadata."""
    h=hashlib.sha256()
    for key in sorted(state):
        t=state[key].detach().cpu().contiguous()
        h.update(key.encode());h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def render_sample(source,di,crop):
    if crop not in CROP_POLICIES:
        raise ValueError("Unknown crop policy")
    views=source["views"]
    rgb,ref,mask=source_image(views[di])
    rgb0,_,mask0=source_image(views[0])
    H,W=mask.shape
    K=np.asarray(views[di]["K"],dtype=np.float32)
    T=np.asarray(views[di]["T_c2w"],dtype=np.float32)
    if crop=="fixed_full":
        box=(0,0,W,H)
    elif crop=="fixed_roi":
        box=_boxed(mask0,margin=.2)
    else:
        box=_workspace_box(ref,mask,K,T)
    image,kn,r,fg=crop_inputs(rgb,ref,mask,K,box)
    # Same world orientation across all distance views, by render contract.
    R=T[:3,:3]
    pose=np.asarray(-R[:,2],dtype=np.float32)
    gravity=np.asarray(R.T@np.array([0.,0.,1.],dtype=np.float32),
                       dtype=np.float32)
    if not np.isclose(np.linalg.norm(pose),1.,atol=1e-4):
        raise RuntimeError("Non-unit frozen global FiLM pose")
    return dict(image=image,ref=r,fg=fg,K=kn,pose=pose,gravity=gravity,
                box=box,distance=float(views[di]["distance_m"]),
                distance_id=di,crop=crop,
                source_rgb_sha256=views[di]["files"]["rgb"]["sha256"])


def _samples(split,crop):
    """Read each object source once; the split manifest is already hashed."""
    samples=[]
    from .factorial_contract import verify_fixed_physics
    for item in split["train"]:
        manifest=json.loads(Path(item["manifest"]).read_text(encoding="utf-8"))
        verify_fixed_physics(manifest)
        views=[render_sample(manifest,di,crop) for di in (0,2,4)]
        samples.append((item["object_sha"],{0:views[0],2:views[1],4:views[2]}))
    return samples


def _sample_pixels(sample,n,rng):
    valid=np.flatnonzero(np.asarray(sample["fg"],dtype=bool)&
                         (np.asarray(sample["ref"],dtype=np.float64)>0))
    if len(valid)<n:
        raise ValueError(f"Not enough foreground at physical {sample['distance']}m "
                         f"for exact budget {n}, only {len(valid)}")
    indices=rng.choice(valid,size=n,replace=False)
    return np.asarray(indices,dtype=np.int64)


def _pred(net,sample,device):
    pose=torch.as_tensor(sample["pose"],device=device,dtype=torch.float32).reshape(1,3)
    gravity=torch.as_tensor(sample["gravity"],device=device,dtype=torch.float32).reshape(1,3)
    k=torch.as_tensor(sample["K"],device=device,dtype=torch.float32).reshape(1,3,3)
    result=net(sample["image"][None].to(device),
               camera_pose_vec=pose,camera_gravity_vec=gravity,camera_K=k)
    pred=result[0]
    if pred.shape!=(1,1,448,448) or not torch.isfinite(pred).all():
        raise RuntimeError("Invalid trainable DPT response")
    return pred.flatten()


def run(a):
    if (a.arm not in ARMS or a.crop_policy not in CROP_POLICIES or
        a.seed<0 or a.steps<1 or a.pixels_per_step<1 or
        a.lr<=0 or a.weight_decay<0 or a.beta<=0):
        raise ValueError("Invalid equal-budget DPT head training protocol")
    reg=read_registered(a.registration)
    if reg["status"]=="synthetic_smoke" and not a.smoke:
        raise ValueError("Procedural CAD can only be used for smoke")
    from rare_grasp.runtime import Runtime
    rt=Runtime(a.bound_config,a.graspnet_root,"",a.device)
    net=check_checkpoint(rt)
    head=net.depthnet.depth_head
    net.eval().requires_grad_(False)
    head.requires_grad_(True)
    # Train head with frozen batchnorm behavior and frozen DINO/pose adapter,
    # so arm comparison is not confounded by moving BN statistics.
    head.eval()
    trainable=[p for p in rt.model.parameters() if p.requires_grad]
    if {id(p) for p in trainable}!={id(p) for p in head.parameters()}:
        raise RuntimeError("Only original DPT depth_head parameters may update")
    initial=_state_hash(head.state_dict())
    samples=_samples(reg,a.crop_policy)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=a.weight_decay)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Training must use a new output root")
    root.mkdir(parents=True,exist_ok=True)
    # The object index sequence is constructed independently of pixel draws
    # and is identical for both arms with the same seed / registered split.
    obj_rng=np.random.default_rng(a.seed+701)
    obj_indices=obj_rng.integers(0,len(samples),size=a.steps)
    pixels_rng=np.random.default_rng(a.seed+1701)
    history=[];step_sources=[]
    for step,obj_ix in enumerate(obj_indices):
        obj_sha,views=samples[int(obj_ix)]
        di=DISTANCE_CHOICES[a.arm][step%3]
        sample=views[di]
        idx=_sample_pixels(sample,a.pixels_per_step,pixels_rng)
        p=_pred(net,sample,rt.device)
        reference=torch.as_tensor(sample["ref"],device=rt.device,
                                  dtype=torch.float32).flatten()
        ids=torch.as_tensor(idx,device=rt.device,dtype=torch.long)
        loss=F.smooth_l1_loss(p[ids],reference[ids],beta=a.beta,reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite equal-budget depth supervision")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(trainable,a.grad_clip,
                                             error_if_nonfinite=True)
        opt.step()
        step_sources.append(dict(step=step,object_sha=obj_sha,
                                 distance_id=di,distance_m=sample["distance"],
                                 foreground_pixels=len(idx)))
        history.append(dict(step=step+1,loss=float(loss.detach().cpu()),
                            grad_norm=float(grad.detach().cpu())))
        if (step+1)%max(1,a.steps//10)==0:
            print(f"[P1b {a.arm}] step {step+1}/{a.steps} "
                  f"loss={history[-1]['loss']:.5f}",flush=True)
    protocol=dict(schema=SCHEMA,arm=a.arm,
       distance_schedule=list(DISTANCE_CHOICES[a.arm]),
       train_object_sequence=[x["object_sha"] for x in step_sources],
       train_distance_indices=[x["distance_id"] for x in step_sources],
       train_registration_sha256=sha256(a.registration),
       registration_identity_sha256=reg["registration_sha256"],
       base_checkpoint_sha256=rt.config["checkpoint_sha256"],
       frozen_base_config_digest=rt.config_digest,
       initial_depth_head_sha256=initial,
       steps=a.steps,pixels_per_step=a.pixels_per_step,
       total_supervised_pixels=a.steps*a.pixels_per_step,
       learning_rate=a.lr,weight_decay=a.weight_decay,
       smooth_l1_beta=a.beta,gradient_clip=a.grad_clip,
       crop_policy=a.crop_policy,seed=a.seed,
       optimizer="AdamW",trainable_module="depth_net.depthnet.depth_head ONLY",
       trainable_params=sum(p.numel() for p in trainable),
       train_scene_source="predeclared physically translated CAD train objects",
       eval_objects_used_in_training=False,
       model_updates=a.steps,
       no_grasp_training=True,
       training_code_sha256=sha256(__file__),
       limitations=[
           "Rendered CAD material/lighting and depth masks are privileged training data",
           "Source CAD split may already have been inspected; not an untouched external generalization test",
           "Near-only repeats same near distance whereas near+far sees 3 distances at identical step/pixel budget",
           "Only DPT decoder is trainable; pose FiLM, DINO and grasp modules frozen",
           "No model selection on evaluation objects, best epoch or GC6D grasp AP",
           "This is synthetic controlled distance supervision, not a deployable camera-anchor fix"
       ])
    weights=root/"last.pt"
    torch.save(dict(schema=SCHEMA,kind="DPT_head_only",
                    state=head.state_dict(),protocol=protocol),weights)
    with (root/"history.json").open("w",encoding="utf-8") as f:
        json.dump(history,f,indent=2)
    with (root/"sample_schedule.json").open("w",encoding="utf-8") as f:
        json.dump(step_sources,f,indent=2)
    complete=dict(schema=SCHEMA,finished=True,
        protocol=protocol,weight_sha256=sha256(weights),
        output_head_sha256=_state_hash(head.state_dict()),
        loss_first=history[0]["loss"],loss_last=history[-1]["loss"],
        history_sha256=sha256(root/"history.json"),
        schedule_sha256=sha256(root/"sample_schedule.json"))
    write_json(root/"complete.json",complete)
    return complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--bound-config",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--crop-policy",choices=CROP_POLICIES,default="fixed_roi")
    p.add_argument("--seed",type=int,default=20261117)
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--pixels-per-step",type=int,default=256)
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
