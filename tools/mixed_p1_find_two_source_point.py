#!/usr/bin/env python3
"""Find candidate historical 10% GN + 10% GN-Trans Point Mixed-v2 checkpoints.

Discovery ONLY: never auto-select, mutate, evaluate, or silently substitute a
three-source checkpoint. Use the listed protocol fingerprints with the exact
source worktree before P1-0 official GC6D inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search-root",action="append",required=True)
    p.add_argument("--max-candidates",type=int,default=80)
    p.add_argument("--output-json",default=None)
    return p.parse_args()


def classify(checkpoint):
    row=dict(path=str(checkpoint),compatible_metadata=False)
    try:
        try:
            obj=torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True)
        except (TypeError,RuntimeError,ValueError):
            obj=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if not isinstance(obj,dict):
            row["reason"]="not dict"
            return row
        p=obj.get("protocol",{})
        if not isinstance(p,dict):
            row["reason"]="missing historical protocol"
            return row
        for key in ("version","variant","train_frames","train_fraction","epochs",
                    "seed","batch_per_gpu","world_size","lr","weight_decay",
                    "partial_run","code_sha256","dav2_sha256"):
            row[key]=obj.get(key) if key=="version" else p.get(key)
        row["model_source_fingerprint"]=p.get("code_sha256")
        row["gntrans_rgb_root"]=p.get("gntrans_rgb_root")
        row["gntrans_supervision"]=p.get("mixed_depth_supervision",{}).get(
            "gntrans_supervision") if isinstance(p.get("mixed_depth_supervision"),dict) else None
        row["complete_epoch"]=int(obj.get("epoch",-100))+1==int(p.get("epochs",0))
        row["compatible_metadata"]=bool(
            p.get("variant")=="rayrope_point"
            and int(p.get("train_frames",0))==5200
            and float(p.get("train_fraction",0))==.1
            and int(p.get("epochs",0))==20
            and int(p.get("seed",-1))==0
            and int(p.get("batch_per_gpu",0))*int(p.get("world_size",0))==9
            and not p.get("partial_run",True)
            and row["complete_epoch"]
            and isinstance(p.get("mixed_depth_supervision"),dict)
            and int(p["mixed_depth_supervision"].get("contract_version",0))==2
            and bool(p.get("code_sha256"))
        )
        row["reason"]="candidate: verify source-tree SHA before loading" if row["compatible_metadata"] else "protocol mismatch"
    except Exception as e:
        row["reason"]=f"{type(e).__name__}: {e}"
    return row


def main():
    a=parse_args()
    if a.max_candidates<1:
        raise ValueError("--max-candidates must be positive")
    files=set()
    for base in a.search_root:
        root=Path(base).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for pattern in ("checkpoint_latest.pt","checkpoint_epoch_019.tar","checkpoint.tar"):
            for file in root.rglob(pattern):
                files.add(file)
                if len(files)>a.max_candidates:
                    raise RuntimeError(
                        f"Found >{a.max_candidates} potential checkpoints; narrow --search-root"
                    )
    rows=[classify(file) for file in sorted(files)]
    found=[row for row in rows if row["compatible_metadata"]]
    result=dict(discovery_only=True,scanned=len(rows),candidates=found,
                rejection_count=len(rows)-len(found),
                selection="explicit TWO_POINT_CKPT and matching TWO_HIST_ROOT required",
                caveat="Protocol metadata filtering does NOT certify the checkpoint source fingerprint; strict runtime loader does.",
                other_files=rows if not found else None)
    if a.output_json:
        path=Path(a.output_json);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,default=str)+"\n")
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__":
    main()
