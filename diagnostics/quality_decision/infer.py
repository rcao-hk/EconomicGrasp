"""P0/P1: frozen GVAR + independently learned probe, full 1024-query inference.

All grasp geometry, depth-dependent width, view and angle come unchanged from
the same native model; only scalar score and consequently depth/priority change.
The official evaluator is invoked unmodified, on a fixed 10% or full 100% grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from rare_grasp.core import select_grasps
from .core import QualityProbe, SCHEMA, sha


SPLITS={"test_seen":range(100,130),
        "test_similar":range(130,160),
        "test_novel":range(160,190)}


def load_probe(path, runtime, device):
    if not path:
        return None,None
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("schema")!=SCHEMA:
        raise ValueError("Wrong P0/P1 probe schema")
    c=ck["config"]
    if c["base_checkpoint_sha256"]!=runtime.config["checkpoint_sha256"]:
        raise ValueError("Probe belongs to a different GVAR checkpoint")
    if c["base_config_digest"]!=runtime.config_digest:
        raise ValueError("Probe runtime model arguments mismatch")
    if c["test_labels_used"] is not False:
        raise ValueError("Test-label leakage flag")
    model=QualityProbe(c["mode"],c["hidden"]).to(device)
    model.load_state_dict(ck["model"],strict=True)
    model.eval().requires_grad_(False)
    return model,ck


def sample_rows(split,interval):
    if interval not in (1,10):
        raise ValueError("Only official 100%/10% evaluation intervals 1 or 10")
    return [dict(domain="original",scene=s,anno=a,split=split,partition="test")
            for s in SPLITS[split] for a in range(0,256,interval)]


def infer(a):
    from rare_grasp.runtime import Runtime
    from .core import predictor
    rt=Runtime(a.config,a.dataset_root,a.gntrans_root,a.device)
    if rt.config["backbone"]!="point_gvar" or float(rt.config.get("mix_training_fraction",.1))!=.5:
        raise ValueError("Primary P0/P1 requires frozen MIX50 GVAR checkpoint")
    model,checkpoint=load_probe(a.weights,rt,rt.device)
    policies=list(dict.fromkeys(a.policies.split(",")))
    if not policies or any(p not in ("native","both","depth_only","rank_only") for p in policies):
        raise ValueError("Unexpected policy")
    if model is None and policies!=["native"]:
        raise ValueError("No scorer checkpoint: use native only")
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh empty inference output")
    root.mkdir(parents=True,exist_ok=True)
    rows=sample_rows(a.split,a.sample_interval)
    complete=True
    if a.limit:
        rows=rows[:a.limit];complete=False
    records=[]
    for n,row in enumerate(rows,1):
        snap=rt.snapshot(row,a.seed)
        bank=np.asarray(snap["grasps"],dtype=np.float32)
        if model is None:
            scores=bank[...,0]
        else:
            x=dict(features=snap["features"].reshape(-1,64),
                   base_logit=snap["base_logit"].reshape(-1),
                   grasps=bank.reshape(-1,17),
                   depth=np.tile(np.arange(4),1024),
                   family=np.repeat(np.arange(1024),4),
                   count=np.zeros(4096,dtype=np.int64),
                   physical_id=__import__("rare_grasp.core",fromlist=["physical_ids"]).physical_ids(bank.reshape(-1,17)))
            with torch.inference_mode():
                scores=predictor(model,x,rt.device,seed=a.seed+row["scene"]*1009+row["anno"]).sigmoid().cpu().numpy().reshape(1024,4)
        for policy in policies:
            actions=select_grasps(bank,scores,policy)
            before=len(actions)
            actions=rt.sensor_collision_filter(row,actions)
            dest=root/policy/f"scene_{row['scene']:04d}"/"realsense"/f"{row['anno']:04d}.npy"
            dest.parent.mkdir(parents=True,exist_ok=True)
            np.save(dest,actions)
            records.append(dict(scene=row["scene"],anno=row["anno"],
                                policy=policy,path=str(dest.relative_to(root)),
                                before_filter=before,after_filter=len(actions),
                                sha256=sha(dest)))
        print(f"[probe infer] {n}/{len(rows)} scene={row['scene']} anno={row['anno']}",flush=True)
    manifest=dict(schema=SCHEMA,status="complete" if complete else "smoke_only",
                  base_checkpoint_sha256=rt.config["checkpoint_sha256"],
                  base_config_digest=rt.config_digest,
                  probe_sha256=sha(a.weights) if a.weights else None,
                  probe_mode=checkpoint["config"]["mode"] if checkpoint else None,
                  training_index_sha256=checkpoint["config"]["cache_contract"]["index_sha256"] if checkpoint else None,
                  input_domain="original",camera="realsense",split=a.split,
                  sample_interval=a.sample_interval,
                  expected_frames=len(sample_rows(a.split,a.sample_interval)),
                  frames=len(rows),seed=a.seed,policies=policies,
                  geometry="frozen_native_1024_queries_depth_width_view_angle",
                  collision="original_sensor:.01voxel,.01threshold,.05approach",
                  records=records,test_quality_labels_used=False)
    (root/"inference_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"Prepared {len(rows)} inference frames: {root}")


def evaluate(a):
    root=Path(a.inference)
    manifest=json.loads((root/"inference_manifest.json").read_text())
    if manifest.get("schema")!=SCHEMA or manifest.get("status")!="complete":
        raise ValueError("Only complete non-smoke inference is eligible for official AP")
    if manifest["frames"]!=30*len(range(0,256,manifest["sample_interval"])):
        raise ValueError("Expected all 30 scenes in official test split")
    for rec in manifest["records"]:
        path=root/rec["path"]
        if not path.is_file() or sha(path)!=rec["sha256"]:
            raise ValueError(f"Missing/modified grasp dump: {path}")
        arr=np.load(path,allow_pickle=False)
        if arr.ndim!=2 or arr.shape[1]!=17 or not np.isfinite(arr).all():
            raise ValueError(f"Malformed grasp array: {path}")
    results={}
    for policy in manifest["policies"]:
        cmd=[sys.executable,"eval.py","--dataset_root",a.dataset_root,
             "--dump_dir",str(root/policy),"--camera","realsense",
             "--split",manifest["split"],"--sample_interval",str(manifest["sample_interval"]),
             "--num_workers",str(a.workers)]
        subprocess.run(cmd,check=True)
        path=root/policy/f"ap_{manifest['split']}_realsense.npy"
        ap=np.load(path,allow_pickle=False)
        expected=(30,len(range(0,256,manifest["sample_interval"])),50,6)
        if ap.shape!=expected or not np.isfinite(ap).all() or np.min(ap)<0 or np.max(ap)>1:
            raise ValueError(f"Bad official AP array {path}: {ap.shape}")
        results[policy]=dict(ap_percent=float(ap.mean()*100),sha256=sha(path))
    (root/"evaluation.json").write_text(json.dumps(
        dict(schema=SCHEMA,status="complete",official=True,
             manifest_sha256=sha(root/"inference_manifest.json"),results=results),indent=2))
    print(json.dumps(results,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    i=sub.add_parser("infer")
    i.add_argument("--config",required=True)
    i.add_argument("--weights",default=None)
    i.add_argument("--dataset-root",required=True)
    i.add_argument("--gntrans-root",default="")
    i.add_argument("--device",default="cuda:0")
    i.add_argument("--output",required=True)
    i.add_argument("--split",choices=list(SPLITS),required=True)
    i.add_argument("--sample-interval",type=int,choices=(1,10),default=10)
    i.add_argument("--policies",default="both,depth_only,rank_only")
    i.add_argument("--limit",type=int,default=0)
    i.add_argument("--seed",type=int,default=20261113)
    e=sub.add_parser("eval")
    e.add_argument("--inference",required=True)
    e.add_argument("--dataset-root",required=True)
    e.add_argument("--workers",type=int,default=16)
    a=p.parse_args()
    (infer if a.cmd=="infer" else evaluate)(a)


if __name__=="__main__":
    main()
