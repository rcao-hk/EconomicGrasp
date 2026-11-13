"""Paired 30-scene per-split comparison of official AP for frozen-base probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import SCHEMA, sha


def load_run(root,split,policy):
    root=Path(root)
    m=json.loads((root/"inference_manifest.json").read_text())
    e=json.loads((root/"evaluation.json").read_text())
    if (m.get("schema")!=SCHEMA or e.get("schema")!=SCHEMA or
            e.get("status")!="complete" or m["split"]!=split or policy not in m["policies"]):
        raise ValueError("Run is incomplete or has incompatible split/policy")
    if e["manifest_sha256"]!=sha(root/"inference_manifest.json"):
        raise ValueError("Inference and official evaluation manifest changed")
    file=root/policy/f"ap_{split}_realsense.npy"
    if sha(file)!=e["results"][policy]["sha256"]:
        raise ValueError("Official AP array hash mismatch")
    ap=np.load(file,allow_pickle=False)
    shape=(30,len(range(0,256,m["sample_interval"])),50,6)
    if ap.shape!=shape or not np.isfinite(ap).all():
        raise ValueError(f"Bad AP shape {ap.shape}, expected {shape}")
    return m,ap


def compare(left,right,policy="both",bootstrap=5000,seed=0):
    if bootstrap<1:raise ValueError("Bootstrap iterations must be positive")
    result={}
    rng=np.random.default_rng(seed)
    for split in ("test_seen","test_similar","test_novel"):
        m0,a=load_run(left[split],split,policy)
        m1,b=load_run(right[split],split,policy)
        for key in ("base_checkpoint_sha256","base_config_digest","sample_interval",
                    "input_domain","camera","frames","seed","collision","geometry"):
            if m0[key]!=m1[key]:
                raise ValueError(f"{split}: unpaired protocol {key}")
        delta=(b-a).mean(axis=(1,2,3))*100.
        means=dict(left=float(a.mean()*100),right=float(b.mean()*100),
                   delta_pp=float(delta.mean()))
        scene_ids=list(range({"test_seen":100,"test_similar":130,"test_novel":160}[split],
                             {"test_seen":130,"test_similar":160,"test_novel":190}[split]))
        result[split]=dict(**means,per_scene_delta_pp=dict(zip(map(str,scene_ids),delta.tolist())))
        result[split]["ci95"]=np.quantile(
            delta[rng.integers(0,len(delta),size=(bootstrap,len(delta)))].mean(-1),
            [.025,.975]).tolist()
    split_delta=np.array([result[s]["delta_pp"] for s in result])
    return dict(schema=SCHEMA,method="paired_official_ap",
                splits=result,mean_delta_pp=float(split_delta.mean()),
                scope="scene paired; training seed not included",bootstrap=bootstrap,seed=seed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left",action="append",required=True,help="SPLIT=INFERENCE_ROOT")
    p.add_argument("--right",action="append",required=True,help="SPLIT=INFERENCE_ROOT")
    p.add_argument("--policy",default="both",choices=("native","both","depth_only","rank_only"))
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=0)
    a=p.parse_args()
    def parse(items):
        d={}
        for x in items:
            name,sep,path=x.partition("=")
            if not sep or name in d:raise ValueError("Use unique SPLIT=PATH")
            d[name]=path
        return d
    left,right=parse(a.left),parse(a.right)
    expected={"test_seen","test_similar","test_novel"}
    if set(left)!=expected or set(right)!=expected:
        raise ValueError("All Seen/Similar/Novel paths required")
    result=compare(left,right,a.policy,a.bootstrap,a.seed)
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v["delta_pp"] for k,v in result["splits"].items()},indent=2))


if __name__=="__main__":
    main()
