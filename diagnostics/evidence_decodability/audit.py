"""P1 no-retraining, action-level held-out evidence probe audit.

Produce paired physical failure/quality predictions on exactly the same native
actions in scorer validation scenes 80..89 or reused audit scenes 90..99.
Official test CAD labels are NEVER loaded by this module.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from diagnostics.candidate_conflict.core import scene_cluster_bootstrap
from .core import (SCHEMA, EvidenceProbe, failure_class, quality_stats,
                   failure_metrics, sha256, write_json)
from .p0_match import load_manifest, read_bank
from .train import _frame_subset, split_entries


def _load(path,cache_sha,device):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("schema")!=SCHEMA or ck.get("kind")!="physical-evidence-probe":
        raise ValueError("Wrong P1 evidence probe schema")
    c=ck["protocol"]
    if c.get("cache_manifest_sha256")!=cache_sha or c.get("test_labels_used") is not False:
        raise ValueError("Probe trained on wrong evidence cache or used test quality")
    if ck.get("epoch")!=c["epochs"]:
        raise ValueError("Only preregistered fixed last epoch may be audited")
    model=EvidenceProbe(c["hidden_dim"],c["evidence_mode"],
                        c["task"],width=c["head_width"]).to(device)
    model.load_state_dict(ck["model"],strict=True)
    model.eval().requires_grad_(False)
    return model,c


@torch.inference_mode()
def audit(a):
    manifest=load_manifest(a.cache_manifest)
    entries=split_entries(manifest)
    if a.partition not in ("validation","reused_audit"):
        raise ValueError("Only held-out scorer scenes may be audited")
    dev=torch.device(a.device)
    model,protocol=_load(a.weights,manifest["manifest_sha256"],dev)
    rows=[];frames=[]
    for entry in entries[a.partition]:
        x=_frame_subset(read_bank(entry),entry["row"])
        y=failure_class(x["quality_count"],x["pure_collision"],x["empty"])
        logits=model(x)
        if model.task=="quality":
            predicted=logits.sigmoid().cpu().numpy()
            probs=None;quality=quality_stats(predicted,x["quality_count"]/6.)
            confidence=np.abs(predicted-x["quality_count"]/6.)
            perframe=dict(mse=quality["mse"],pair_agreement=quality["pair_agreement"],
                          top10_quality=quality["top10_quality"],top10_zero=quality["top10_zero"])
        else:
            probs=logits.softmax(dim=-1).cpu().numpy()
            predicted=probs.argmax(axis=1).astype(int)
            quality=failure_metrics(probs,y)
            confidence=1.-probs[np.arange(len(y)),y]
            perframe=dict(balanced_accuracy=quality["balanced_accuracy"],
                          macro_f1=quality["macro_f1"])
        row=entry["row"]
        frames.append(dict(scene=int(row["scene"]),domain=row["domain"],
                           anno=int(row["anno"]),**perframe))
        for i in range(len(y)):
            record=dict(domain=row["domain"],scene=int(row["scene"]),
                        anno=int(row["anno"]),query_id=int(x["query_id"][i]),
                        depth_id=int(x["depth_id"][i]),
                        source_new=int(x["source_new"][i]),
                        nms_role=int(x["nms_role"][i]),
                        sensor_valid=int(x["sensor_valid"][i]),
                        native_score=float(x["native_score"][i]),
                        native_score_percentile=float(x["native_score_percentile"][i]),
                        quality_count=int(x["quality_count"][i]),
                        failure_class=int(y[i]),
                        prediction=float(predicted[i]),
                        absolute_error=float(confidence[i]))
            if probs is not None:
                for k in range(4):record[f"prob_class_{k}"]=float(probs[i,k])
            rows.append(record)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    def csvfile(path,values):
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]))
            w.writeheader();w.writerows(values)
    csvfile(root/"per_action.csv",rows)
    csvfile(root/"per_frame.csv",frames)
    condition_groups=defaultdict(list)
    for row in rows:
        high=int(row["native_score_percentile"]>=.75)
        k=(row["domain"],row["source_new"],row["nms_role"],high,row["failure_class"])
        condition_groups[k].append(row)
    strata=[]
    for key,values in sorted(condition_groups.items()):
        n=len(values)
        if n<a.min_cell_n:continue
        strata.append(dict(domain=key[0],source_new=key[1],nms_role=key[2],
                           native_high_score=key[3],failure_class=key[4],actions=n,
                           mean_abs_error=float(np.mean([v["absolute_error"] for v in values])),
                           mean_prediction=float(np.mean([v["prediction"] for v in values]))))
    if strata:csvfile(root/"strata.csv",strata)
    all_scenes=sorted({r["scene"] for r in frames})
    metrics={k for r in frames for k in r if k not in ("domain","scene","anno")}
    result={}
    for metric in sorted(metrics):
        by_scene={int(s):float(np.mean([r[metric] for r in frames
                                       if r["scene"]==s and r.get(metric) is not None]))
                  for s in all_scenes}
        result[metric]=scene_cluster_bootstrap(by_scene,n_boot=a.bootstrap,seed=20261115)
    write_json(root/"summary.json",dict(schema=SCHEMA,kind="P1_evidence_action_audit",
                checkpoint_sha256=sha256(a.weights),
                evidence_manifest_sha256=manifest["manifest_sha256"],
                partition=a.partition,
                train_scenes_seen_by_frozen_base=True,
                previously_inspected=a.partition=="reused_audit",
                task=model.task,mode=model.mode,frames=len(frames),
                actions=len(rows),metrics=result,min_cell_n=a.min_cell_n,
                limitations=["All actions share the same frozen trained GVAR base",
                             "No label from GraspNet test scenes 100..189 was consumed",
                             "Partial conditional strata are not independent action IID samples"]))
    print(json.dumps({k:v["mean"] for k,v in result.items()},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-manifest",required=True)
    p.add_argument("--weights",required=True)
    p.add_argument("--partition",choices=("validation","reused_audit"),default="validation")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output",required=True)
    p.add_argument("--min-cell-n",type=int,default=20)
    p.add_argument("--bootstrap",type=int,default=5000)
    audit(p.parse_args())


if __name__=="__main__":
    main()
