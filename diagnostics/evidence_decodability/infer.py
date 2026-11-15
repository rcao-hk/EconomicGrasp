"""P1 official Original GraspNet test for deployable native-depth quality probes.

Only frozen RGB-predicted-depth features are supplied at test time. Evaluator
uses original collision filter, GraspNet NMS and official [30,26,50,6] AP.
Privileged reference-depth geometry and failure-class classifiers are forbidden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from diagnostics.decision_supervision.core import BASE_SHA, decode, physical_ids
from diagnostics.decision_supervision.runtime import make_runtime
from .core import SCHEMA, EvidenceProbe, sha256, write_json
from .capture import GVARCapture

SPLITS={"test_seen":100,"test_similar":130,"test_novel":160}


def load_probe(path,device):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if (ck.get("schema")!=SCHEMA or ck.get("kind")!="physical-evidence-probe"
            or ck.get("protocol",{}).get("base_checkpoint_sha256")!=BASE_SHA
            or ck["protocol"].get("test_labels_used") is not False):
        raise ValueError("P1 checkpoint provenance/label-use mismatch")
    c=ck["protocol"]
    if c["task"]!="quality" or c["evidence_mode"]=="reference_geometry":
        raise ValueError("Only deployable RGB/predicted-geometry quality probes can run official test")
    if ck["epoch"]!=c["epochs"]:
        raise ValueError("Formal evaluation uses preregistered last epoch only")
    model=EvidenceProbe(c["hidden_dim"],c["evidence_mode"],c["task"],c["head_width"]).to(device)
    model.load_state_dict(ck["model"],strict=True)
    model.eval().requires_grad_(False)
    return model,c


def native_actions(rt,row,seed,model=None):
    with GVARCapture(rt) as capture:
        snap=rt.snapshot(row,seed)
    bank=np.asarray(snap["grasps"],dtype=np.float32)
    d=bank[...,0].argmax(axis=1)
    if model is None:
        return bank[np.arange(1024),d].copy(),snap
    q=np.arange(1024,dtype=np.int64)
    evidence=capture.chosen(snap,q,d)
    batch=dict(query_id=q,depth_id=d,
               native_score=bank[q,d,0],
               grasps=bank[q,d],
               base_logit=np.asarray(snap["base_logit"])[q,d],
               score64=np.asarray(snap["features"])[q,d],
               **evidence)
    with torch.inference_mode():
        p=model(batch).sigmoid().cpu().numpy()
    if p.shape!=(1024,) or not np.isfinite(p).all():
        raise ValueError("Invalid learned action quality scores")
    scores=bank[...,0].copy()
    scores[q,d]=p
    result=decode(bank,scores,"rank_executed")
    native=bank[q,d]
    if not np.array_equal(result[:,1:16],native[:,1:16]):
        raise RuntimeError("P1 changed grasp center/R/depth/width rather than scalar scores")
    return result,snap


def infer(a):
    rt,_=make_runtime(a)
    model,protocol=(None,None) if a.weights is None else load_probe(a.weights,rt.device)
    if protocol is not None and protocol.get("cache_manifest_sha256") is None:
        raise ValueError("Missing training evidence cache identity")
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("P1 inference output must be new and empty")
    root.mkdir(parents=True,exist_ok=True)
    scene0=SPLITS[a.split]
    frames=[(s,k) for s in range(scene0,scene0+30) for k in range(0,256,a.interval)]
    if a.limit:
        if not 0<a.limit<len(frames):
            raise ValueError("Smoke limit must be positive and smaller than full split")
        frames=frames[:a.limit]
    rec=[]
    for n,(scene,anno) in enumerate(frames,1):
        row=dict(domain="original",scene=scene,anno=anno,split=a.split,partition="test")
        result,snap=native_actions(rt,row,a.seed,model)
        filtered=np.asarray(rt.sensor_collision_filter(row,result))
        if filtered.ndim!=2 or filtered.shape[1]!=17 or not np.isfinite(filtered).all():
            raise RuntimeError("Invalid post-sensor collision result")
        out=root/f"scene_{scene:04d}"/"realsense"/f"{anno:04d}.npy"
        out.parent.mkdir(parents=True,exist_ok=True)
        np.save(out,filtered)
        rec.append(dict(scene=scene,anno=anno,path=str(out.relative_to(root)),
                        sha256=sha256(out),num_pre_filter=len(result),
                        num_post_filter=len(filtered)))
        if n%26==0 or n==len(frames):
            print(f"[P1 infer] {n}/{len(frames)} scene={scene} frame={anno}",flush=True)
    write_json(root/"inference.json",dict(
        schema=SCHEMA,kind="evidence-quality-inference",
        complete=not bool(a.limit),split=a.split,camera="realsense",
        interval=a.interval,num_frames=len(frames),
        base_checkpoint_sha256=BASE_SHA,
        probe_sha256=sha256(a.weights) if a.weights else None,
        probe_protocol=protocol,
        use_privileged_reference=False,used_test_labels=False,
        selection="rank_executed; original native depth, unchanged geometry",
        collision="original sensor voxel0.01 threshold0.01 approach0.05",
        snapshot_seed=a.seed,records=rec))


def evaluate(a):
    root=Path(a.inference)
    manifest=root/"inference.json"
    m=json.loads(manifest.read_text(encoding="utf-8"))
    if m.get("schema")!=SCHEMA or m.get("kind")!="evidence-quality-inference" or not m.get("complete"):
        raise ValueError("Only complete non-smoke inference can be officially evaluated")
    expected=30*len(range(0,256,m["interval"]))
    if len(m["records"])!=expected or m["num_frames"]!=expected:
        raise ValueError("Incomplete official GraspNet test frame count")
    keys=set()
    for r in m["records"]:
        pair=(r["scene"],r["anno"])
        if pair in keys:raise ValueError("Duplicate official test frame")
        keys.add(pair)
        path=root/r["path"]
        if sha256(path)!=r["sha256"]:
            raise ValueError(f"Changed grasp dump: {path}")
        g=np.load(path,allow_pickle=False)
        if g.ndim!=2 or g.shape[1]!=17 or not np.isfinite(g).all():
            raise ValueError("Malformed 17D grasp dump")
    cmd=[sys.executable,"eval.py","--dataset_root",a.dataset_root,
         "--dump_dir",str(root),"--camera","realsense",
         "--split",m["split"],"--sample_interval",str(m["interval"]),
         "--num_workers",str(a.workers)]
    subprocess.run(cmd,check=True)
    path=root/f"ap_{m['split']}_realsense.npy"
    ap=np.load(path,allow_pickle=False)
    shape=(30,len(range(0,256,m["interval"])),50,6)
    if ap.shape!=shape or not np.isfinite(ap).all() or ap.min()<0 or ap.max()>1:
        raise ValueError(f"Official AP array invalid: {ap.shape}")
    write_json(root/"evaluation.json",dict(
        schema=SCHEMA,status="complete",official=True,
        inference_sha256=sha256(manifest),
        ap_sha256=sha256(path),shape=shape,ap_percent=float(ap.mean()*100),
        seen_test_labels_for_training=False))
    print(json.dumps(dict(ap_percent=float(ap.mean()*100),shape=shape),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    i=sub.add_parser("infer")
    i.add_argument("--config",required=True)
    i.add_argument("--dataset-root",required=True)
    i.add_argument("--gntrans-root",default="")
    i.add_argument("--factory",default="rare_grasp.runtime:Runtime")
    i.add_argument("--device",default="cuda:0")
    i.add_argument("--weights",default=None,help="Omit for strict native baseline")
    i.add_argument("--split",choices=tuple(SPLITS),required=True)
    i.add_argument("--interval",type=int,choices=(1,10),default=10)
    i.add_argument("--seed",type=int,default=20261113)
    i.add_argument("--limit",type=int,default=0)
    i.add_argument("--output",required=True)
    e=sub.add_parser("eval")
    e.add_argument("--inference",required=True)
    e.add_argument("--dataset-root",required=True)
    e.add_argument("--workers",type=int,default=16)
    a=p.parse_args()
    (infer if a.cmd=="infer" else evaluate)(a)


if __name__=="__main__":
    main()
