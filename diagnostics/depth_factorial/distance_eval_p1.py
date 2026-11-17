"""P1b evaluate FIXED LAST depth head on preregistered CAD eval objects.

No test-based epoch/hyperparameter selection. Evaluate baseline/near/wide
against exactly the same physical render object-distance-crop source, save
per-cell predicted depth SHA and reuse the original same-surface B analysis.
The unit of inference is independent CAD object (not individual pixels).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from diagnostics.depth_followup.common import check_checkpoint
from diagnostics.frozen_depth_response.geometry import paired_depth_metrics
from .distance_split_p1 import read_registered
from .distance_train_p1 import SCHEMA,ARMS,_state_hash,_pred,render_sample
from .factorial_contract import (CROP_POLICIES,INFER_SCHEMA,
                                  sha256,digest,write_json,verify_fixed_physics)
from .factorial_analyze import run as analyze_same_surface

EVAL_SCHEMA="physical-distance-head-eval-v1"


def _load_trained(a,registration,net):
    if a.arm=="native":
        if a.head_checkpoint:
            raise ValueError("Native baseline must not load a fitted head")
        return None
    p=Path(a.head_checkpoint or "")
    if not p.is_file() or p.name!="last.pt":
        raise ValueError(f"Only existing fixed last.pt accepted: {p}")
    if sha256(p)!=a.head_sha256:
        raise RuntimeError("Trained head checkpoint hash differs from preregistration")
    ck=torch.load(p,map_location="cpu",weights_only=False)
    if ck.get("schema")!=SCHEMA or ck.get("kind")!="DPT_head_only":
        raise ValueError("Wrong DPT head-only checkpoint contract")
    protocol=ck["protocol"]
    if (protocol["arm"]!=a.arm or
            protocol["train_registration_sha256"]!=sha256(a.registration) or
            protocol["registration_identity_sha256"]!=registration["registration_sha256"]):
        raise ValueError("Head trained on different source CAD split/arm")
    if protocol["initial_depth_head_sha256"]!=_state_hash(net.depthnet.depth_head.state_dict()):
        raise ValueError("Train/eval started from different frozen DPT-head parameters")
    if protocol["base_checkpoint_sha256"]!=(
          "2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675"):
        raise ValueError("Head initialization not historical MIX50")
    if ck["protocol"]["crop_policy"]!=a.train_crop_policy:
        raise ValueError("Head trained with different declared crop policy")
    if protocol.get("training_status")=="synthetic_smoke" and not a.smoke:
        raise ValueError("Cannot evaluate procedural-smoke depth head formally")
    net.depthnet.depth_head.load_state_dict(ck["state"],strict=True)
    return protocol


@torch.inference_mode()
def run(a):
    from rare_grasp.runtime import Runtime
    reg=read_registered(a.registration)
    if reg["status"]=="synthetic_smoke" and not a.smoke:
        raise ValueError("Cannot claim formal P1b evaluation on procedural smoke CAD")
    rt=Runtime(a.bound_config,a.graspnet_root,"",a.device)
    net=check_checkpoint(rt)
    base_head_hash=_state_hash(net.depthnet.depth_head.state_dict())
    protocol=_load_trained(a,reg,net)
    net.eval().requires_grad_(False)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    all_rows=[];object_analysis=[]
    for i,item in enumerate(reg["eval"],1):
        srcfile=Path(item["manifest"])
        source=json.loads(srcfile.read_text(encoding="utf-8"))
        verify_fixed_physics(source)
        name=item["object_sha"][:16]
        folder=root/f"cad_{name}"
        folder.mkdir()
        entries=[];summary_rows=[]
        for di in range(len(source["views"])):
            for crop in CROP_POLICIES:
                sample=render_sample(source,di,crop)
                p=_pred(net,sample,rt.device).detach().float().cpu().numpy().reshape(448,448)
                stats=paired_depth_metrics(p,sample["ref"],sample["fg"],128)
                if not stats["valid"]:
                    raise RuntimeError("Evaluation CAD depth/seg has insufficient foreground")
                path=folder/"predictions"/crop/f"distance_{di:03d}.npy"
                path.parent.mkdir(parents=True,exist_ok=True)
                np.save(path,p,allow_pickle=False)
                view=source["views"][di]
                entries.append(dict(distance_id=di,distance_m=float(view["distance_m"]),
                    crop_policy=crop,crop_box=list(map(int,sample["box"])),
                    K=np.asarray(sample["K"]).tolist(),
                    prediction_file=str(path.resolve()),prediction_sha256=sha256(path),
                    render_source_sha256=view["files"]))
                record=dict(arm=a.arm,object_sha=item["object_sha"],
                    distance_id=di,distance_m=view["distance_m"],crop_policy=crop,
                    crop_fx=float(sample["K"][0,0]),crop_fy=float(sample["K"][1,1]),
                    crop_fraction=float((sample["box"][2]-sample["box"][0])*
                                       (sample["box"][3]-sample["box"][1])/
                                       (source["plan"]["camera"]["width"]*
                                        source["plan"]["camera"]["height"])),
                    **stats)
                summary_rows.append(record);all_rows.append(record)
        m=dict(schema=INFER_SCHEMA,complete=True,
            source_manifest_sha256=sha256(srcfile),
            source_plan_sha256=source["plan_sha256"],
            base_checkpoint_sha256=rt.config["checkpoint_sha256"],
            crop_policies=list(CROP_POLICIES),records=entries,
            evaluation_arm=a.arm,model_head_sha256=sha256(a.head_checkpoint)
                             if a.arm!="native" else None,
            eval_source_object_sha256=item["object_sha"],
            model="frozen RGB+pose global-FiLM DPT; optionally head-only trained",
            no_grasp_AP=True)
        m["manifest_sha256"]=digest(m)
        prediction_mf=folder/"manifest.json"
        write_json(prediction_mf,m)
        from types import SimpleNamespace
        analysis_dir=folder/"same_surface_analysis"
        result=analyze_same_surface(SimpleNamespace(
            source_manifest=str(srcfile),frozen_manifest=str(prediction_mf),
            output=str(analysis_dir),min_motion=.03,
            max_points=a.anchor_points,min_shared=a.min_shared,
            abs_tol=.006,rel_tol=.008,xyz_tol=.012))
        object_analysis.append(dict(object_sha=item["object_sha"],
            source_manifest_sha256=sha256(srcfile),
            prediction_manifest_sha256=sha256(prediction_mf),
            analysis_sha256=sha256(analysis_dir/"summary.json"),
            analysis_dir=str(analysis_dir.resolve()),
            per_condition=summary_rows,
            same_surface_rows=result["rows"]))
        print(f"[P1b eval {a.arm}] {i}/{len(reg['eval'])} CAD {name}",flush=True)
    csv_path=root/"per_condition.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        cols=list(all_rows[0]);w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader();w.writerows(all_rows)
    result=dict(schema=EVAL_SCHEMA,arm=a.arm,
        status="smoke" if a.smoke else "complete",
        registration_sha256=sha256(a.registration),
        registration_identity_sha256=reg["registration_sha256"],
        base_checkpoint_sha256=rt.config["checkpoint_sha256"],
        base_head_parameter_sha256=base_head_hash,
        fine_tuned_head_sha256=sha256(a.head_checkpoint)
                              if a.arm!="native" else None,
        train_protocol=protocol,train_crop_policy=a.train_crop_policy,
        eval_objects=len(object_analysis),object_analysis=object_analysis,
        per_condition_sha256=sha256(csv_path),
        evaluation_used_for_training=False,model_updated_during_eval=False,
        no_grasp_AP=True,synthetic_only=True,
        notes=[
           "All three arms are evaluated on exactly the same preregistered CAD objects and distances",
           "The five distance×three crop cells per object share one source physical scene",
           "The original four corrected CAD renders have already been inspected; not untouched confirmation",
           "Model never sees eval CAD reference depth during training"
        ])
    result["manifest_sha256"]=digest(result)
    write_json(root/"summary.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--bound-config",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--arm",choices=("native",*ARMS),required=True)
    p.add_argument("--head-checkpoint",default=None)
    p.add_argument("--head-sha256",default=None)
    p.add_argument("--train-crop-policy",choices=(*CROP_POLICIES,"multi3"),default="fixed_roi")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--anchor-points",type=int,default=256)
    p.add_argument("--min-shared",type=int,default=32)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if a.arm!="native" and not a.head_sha256:
        p.error("Formal test evaluation requires verified fixed last.pt SHA")
    run(a)


if __name__=="__main__":
    main()
