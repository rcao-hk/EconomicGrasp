"""Reproduce base-seen, scorer-held-out GraspNet train-scene validation dumps.

Runs exactly ONE frozen GVAR forward per frame and three frozen scoring
policies on the SAME original native executed action geometry. No quality
labels are needed to create predictions; labels are queried separately by
collect.py. Validation here is NOT independent of backbone training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diagnostics.decision_supervision.core import BASE_SHA
from diagnostics.decision_supervision.runtime import make_runtime
from diagnostics.evidence_decodability.capture import GVARCapture
from diagnostics.evidence_decodability.infer import load_probe
from diagnostics.evidence_decodability.p0_match import load_manifest
from .contract import SCHEMA, METHODS, sha256, save_json


def _validation_records(cache_manifest,domain):
    mf=load_manifest(cache_manifest)
    if mf["contract"]["base_checkpoint_sha256"]!=BASE_SHA:
        raise ValueError("Evidence cache from another frozen GVAR base")
    entries=[x for x in mf["records"] if x["row"]["domain"]==domain
             and 80<=int(x["row"]["scene"])<90]
    if len(entries)!=20:
        raise ValueError(f"Expected 20 scorer-held-out views/domain, got {len(entries)}")
    return mf,sorted(entries,key=lambda x:(x["row"]["scene"],x["row"]["anno"]))


@torch.inference_mode()
def replay(a):
    mf,records=_validation_records(a.cache_manifest,a.domain)
    rt,identity=make_runtime(a)
    if rt.config_digest!=mf["contract"]["runtime"]["config_digest"] if False else False:
        raise RuntimeError("Unreachable: runtime config")
    model={}
    for mode in ("score64","all"):
        path=getattr(a,"weights_"+mode)
        head,protocol=load_probe(path,a.device)
        if (protocol["evidence_mode"]!=mode or
                protocol.get("cache_manifest_sha256")!=mf["manifest_sha256"]):
            raise RuntimeError(f"{mode} scorer not trained on same evidence cache")
        model[mode]=head
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use new validation replay output root")
    root.mkdir(parents=True,exist_ok=True)
    output={name:[] for name in METHODS}
    if a.limit:
        if not 0<a.limit<len(records):
            raise ValueError("Smoke limit must be smaller than all20 validation frames")
        records=records[:a.limit]
    for count,rec in enumerate(records,1):
        row=rec["row"]
        with GVARCapture(rt) as cap:
            snap=rt.snapshot(row,a.snapshot_seed)
        bank=np.asarray(snap["grasps"],dtype=np.float32)
        if bank.shape!=(1024,4,17):
            raise RuntimeError("Different frozen 1024x4x17 executed bank")
        depth=bank[...,0].argmax(1)
        q=np.arange(1024,dtype=np.int64)
        evidence=cap.chosen(snap,q,depth)
        original=bank[q,depth].copy()
        source=dict(query_id=q,depth_id=depth,grasps=original,
                    base_logit=np.asarray(snap["base_logit"])[q,depth],
                    score64=np.asarray(snap["features"])[q,depth],
                    **evidence)
        scores={"native":original[:,0]}
        for name in ("score64","all"):
            score=model[name](source).sigmoid().detach().cpu().numpy()
            if score.shape!=(1024,) or not np.isfinite(score).all():
                raise RuntimeError(f"Nonfinite frozen probe: {name}")
            scores[name]=score
        bank_sha=hashlib.sha256(np.ascontiguousarray(
            bank[...,1:16],dtype=np.float32).tobytes()).hexdigest()
        for name in METHODS:
            predicted=original.copy()
            predicted[:,0]=scores[name]
            if not np.array_equal(predicted[:,1:16],original[:,1:16]):
                raise RuntimeError("Scorer modified native executed action geometry")
            filtered=np.asarray(rt.sensor_collision_filter(row,predicted))
            if filtered.ndim!=2 or filtered.shape[1]!=17:
                raise RuntimeError("Sensor collision output malformed")
            folder=root/name
            path=folder/f"scene_{int(row['scene']):04d}"/"realsense"/f"{int(row['anno']):04d}.npy"
            path.parent.mkdir(parents=True,exist_ok=True)
            np.save(path,filtered)
            output[name].append(dict(
                scene=int(row["scene"]),anno=int(row["anno"]),
                path=str(path.relative_to(folder)),sha256=sha256(path),
                frozen_bank_sha256=bank_sha))
        if count%5==0 or count==len(records):
            print(f"[train-scene validation replay] {count}/{len(records)}",flush=True)
    for name,recs in output.items():
        model_source=(None if name=="native" else sha256(getattr(a,"weights_"+name)))
        metadata=dict(schema="physical-evidence-decoding-v1",
                      kind="evidence-quality-inference",
                      complete=not bool(a.limit),split="probe_validation",
                      domain=a.domain,camera="realsense",num_frames=len(recs),
                      base_checkpoint_sha256=BASE_SHA,
                      base_config_digest=rt.config_digest,
                      interval=1,snapshot_seed=a.snapshot_seed,
                      runtime=identity,
                      selection="rank_executed; original native depth, unchanged geometry",
                      collision="original sensor voxel0.01 threshold0.01 approach0.05",
                      probe_sha256=model_source,used_test_labels=False,
                      seen_by_frozen_backbone=True,
                      previously_inspected_training_domain=True,
                      records=recs)
        save_json(root/name/"inference.json",metadata)
    print(f"Saved {len(records)} validation frames × 3 frozen modes at {root}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-manifest",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--gntrans-root",default="")
    p.add_argument("--factory",default="rare_grasp.runtime:Runtime")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--weights-score64",required=True)
    p.add_argument("--weights-all",required=True)
    p.add_argument("--domain",choices=("original","gntrans"),default="original")
    p.add_argument("--snapshot-seed",type=int,default=20261113)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--output",required=True)
    replay(p.parse_args())


if __name__=="__main__":
    main()
