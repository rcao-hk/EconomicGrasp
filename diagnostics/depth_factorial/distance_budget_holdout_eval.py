"""Evaluate ROI/Multi200/600 DPT LAST heads on NEW signed physical CAD.

This deliberately differs from the legacy evaluation loader: the original
training CAD registration is IMMUTABLE, but evaluation CAD comes from a
second signed newly registered asset collection. No eval CAD enters training.
Uses the exact existing matched-world-surface factorial evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from diagnostics.depth_followup.common import check_checkpoint
from diagnostics.frozen_depth_response.geometry import paired_depth_metrics
from .distance_split_p1 import read_registered
from .distance_train_p1 import _state_hash,_pred,render_sample
from .distance_eval_p1 import _load_trained,EVAL_SCHEMA
from .factorial_contract import (CROP_POLICIES,INFER_SCHEMA,
                                  sha256,digest,write_json,
                                  verify_fixed_physics)
from .factorial_analyze import run as analyze_same_surface
from .distance_budget_holdout_register import read_holdout

MODELS={
    "roi200":"fixed_roi","multi200":"multi3",
    "roi600":"fixed_roi","multi600":"multi3",
    "native":None
}


@torch.inference_mode()
def run(a):
    if a.model not in MODELS:
        raise ValueError("Unknown fixed-head evaluation condition")
    training_reg=read_registered(a.registration)
    holdout=read_holdout(a.holdout,a.registration,require_formal=not a.exploratory)
    if a.model=="native" and (a.head_checkpoint or a.head_sha):
        raise ValueError("Native model cannot use a trained DPT head")
    if a.model!="native" and not a.head_sha:
        raise ValueError("Require exact signed DPT LAST checkpoint")
    from rare_grasp.runtime import Runtime
    rt=Runtime(a.bound_config,a.graspnet_root,"",a.device)
    net=check_checkpoint(rt)
    base_hash=_state_hash(net.depthnet.depth_head.state_dict())
    options=SimpleNamespace(
        arm="native" if a.model=="native" else "near_mid_far",
        registration=a.registration,
        head_checkpoint=a.head_checkpoint,
        head_sha256=a.head_sha,
        train_crop_policy=MODELS[a.model] or "fixed_roi",
        smoke=False)
    protocol=_load_trained(options,training_reg,net)
    if protocol is not None:
        n=600 if a.model.endswith("600") else 200
        if protocol["steps"]!=n or protocol["pixels_per_step"]!=256:
            raise ValueError("Unexpected training compute budget in new CAD eval")
        if protocol.get("crop_policy")!=MODELS[a.model]:
            raise ValueError("Wrong 200/600 ROI or Multi head applied")
    net.eval().requires_grad_(False)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("New-CAD evaluation requires empty output directory")
    root.mkdir(parents=True,exist_ok=True)
    rows=[];cad_results=[]
    for i,item in enumerate(holdout["source_render_objects"],1):
        source_file=Path(item["source_manifest"])
        source=json.loads(source_file.read_text(encoding="utf-8"))
        verify_fixed_physics(source,strict_files=True)
        object_sha=item["object_sha"]
        folder=root/f"cad_{object_sha[:16]}"
        folder.mkdir()
        prediction_rows=[];stats=[]
        for distance_id in range(5):
            source_view=source["views"][distance_id]
            for crop in CROP_POLICIES:
                sample=render_sample(source,distance_id,crop)
                pred=_pred(net,sample,rt.device).detach().float().cpu().numpy().reshape(448,448)
                metric=paired_depth_metrics(pred,sample["ref"],sample["fg"],128)
                if not metric["valid"]:
                    raise ValueError("Not enough valid object depth pixels in new CAD source")
                save=folder/"predictions"/crop/f"distance_{distance_id:03d}.npy"
                save.parent.mkdir(parents=True,exist_ok=True)
                np.save(save,pred,allow_pickle=False)
                prediction_rows.append(dict(distance_id=distance_id,
                    distance_m=float(source_view["distance_m"]),
                    crop_policy=crop,crop_box=list(map(int,sample["box"])),
                    K=np.asarray(sample["K"]).tolist(),
                    prediction_file=str(save.resolve()),
                    prediction_sha256=sha256(save),
                    render_source_sha256=source_view["files"]))
                record=dict(model=a.model,object_sha=object_sha,
                    distance_id=distance_id,
                    distance_m=float(source_view["distance_m"]),
                    crop_policy=crop,
                    crop_fx=float(sample["K"][0,0]),
                    crop_fy=float(sample["K"][1,1]),
                    crop_fraction=float(
                        (sample["box"][2]-sample["box"][0])*
                        (sample["box"][3]-sample["box"][1])/
                        (source["plan"]["camera"]["width"]*
                         source["plan"]["camera"]["height"])),
                    **metric)
                stats.append(record);rows.append(record)
        # Preserve exact original B evaluation manifest interface so matched
        # world-surface analysis can be reused unmodified and audited.
        pred_manifest=dict(schema=INFER_SCHEMA,complete=True,
            source_manifest_sha256=sha256(source_file),
            source_plan_sha256=source["plan_sha256"],
            base_checkpoint_sha256=rt.config["checkpoint_sha256"],
            crop_policies=list(CROP_POLICIES),
            records=prediction_rows,
            evaluation_arm="native" if a.model=="native" else "near_mid_far",
            model_head_sha256=sha256(a.head_checkpoint) if protocol else None,
            eval_source_object_sha256=object_sha,
            new_cad_source_provenance=holdout["manifest_sha256"],
            no_grasp_AP=True)
        pred_manifest["manifest_sha256"]=digest(pred_manifest)
        prediction_path=folder/"manifest.json"
        write_json(prediction_path,pred_manifest)
        analysis_dir=folder/"same_surface_analysis"
        result=analyze_same_surface(SimpleNamespace(
            source_manifest=str(source_file),frozen_manifest=str(prediction_path),
            output=str(analysis_dir),min_motion=.03,
            max_points=a.anchor_points,min_shared=a.min_shared,
            abs_tol=.006,rel_tol=.008,xyz_tol=.012))
        cad_results.append(dict(object_sha=object_sha,
            semantic_group=item["semantic_group"],
            material_source_sha256=item["material_source_sha256"],
            source_manifest_sha256=sha256(source_file),
            prediction_manifest_sha256=sha256(prediction_path),
            analysis_sha256=sha256(analysis_dir/"summary.json"),
            analysis_dir=str(analysis_dir.resolve()),
            per_condition=stats,same_surface_rows=result["rows"]))
        print(f"[new CAD eval/{a.model}] {i}/{holdout['object_count']} "
              f"sha={object_sha[:12]}",flush=True)
    csv_path=root/"per_condition.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    result=dict(schema=EVAL_SCHEMA,
        arm="native" if a.model=="native" else "near_mid_far",
        status="complete",
        model_name=a.model,
        registration_sha256=sha256(a.registration),
        registration_identity_sha256=training_reg["registration_sha256"],
        base_checkpoint_sha256=rt.config["checkpoint_sha256"],
        base_head_parameter_sha256=base_hash,
        fine_tuned_head_sha256=sha256(a.head_checkpoint) if protocol else None,
        train_protocol=protocol,
        train_crop_policy=MODELS[a.model] or "fixed_roi",
        eval_objects=len(cad_results),
        object_analysis=cad_results,
        new_cad_holdout_manifest_sha256=sha256(a.holdout),
        new_cad_holdout_status=holdout["status"],
        per_condition_sha256=sha256(csv_path),
        evaluation_used_for_training=False,
        model_updated_during_eval=False,no_grasp_AP=True,synthetic_only=True,
        notes=[
            "Evaluation CAD meshes not in old training2/evaluation2 by SHA",
            "Never-used history is user-declared provisional, not independently proven",
            "All four training modes evaluate the same 5 physical distances and 3 crop policies",
            "No new real GC6D or grasp AP inference",
        ])
    result["manifest_sha256"]=digest(result)
    write_json(root/"summary.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration",required=True)
    p.add_argument("--holdout",required=True)
    p.add_argument("--model",choices=tuple(MODELS),required=True)
    p.add_argument("--bound-config",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--head-checkpoint",default=None)
    p.add_argument("--head-sha",default=None)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--anchor-points",type=int,default=256)
    p.add_argument("--min-shared",type=int,default=32)
    p.add_argument("--exploratory",action="store_true")
    p.add_argument("--output",required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
