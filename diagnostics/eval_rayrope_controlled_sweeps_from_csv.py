#!/usr/bin/env python3
"""Evaluate controlled width/insertion changes from an existing physical-candidate CSV.

This path does NOT require a model checkpoint. It reuses saved grasp poses,
constructs new physical actions by changing one parameter at a time and runs
the existing official-compatible CAD/DexNet exact action evaluator.

Because old CSVs do not contain the complete per-depth CDF lattice, insertion
sweeps from this script are QUALITY-ONLY; do not interpret their frozen native
model score as a valid new-depth prediction.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .analyze_rayrope_candidates import load_candidates
    from .rayrope_action_lattice import physical_action_id
except ImportError:
    from analyze_rayrope_candidates import load_candidates
    from rayrope_action_lattice import physical_action_id


MUS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


def _rotation(raw):
    normed = {re.sub(r"[^a-z0-9]", "", k.lower()):v for k,v in raw.items()}
    values=[]
    for i in range(3):
        for j in range(3):
            opts=(f"r{i}{j}",f"rot{i}{j}",f"rotation{i}{j}")
            val=next((normed[x] for x in opts if x in normed),None)
            if val is None:
                values=[]
                break
            values.append(float(val))
        if not values:
            break
    if len(values)!=9:
        for key in ("rotation_matrix","grasp_rotation","rotation_9","rotation"):
            v=raw.get(key,"")
            if v:
                parsed=json.loads(v)
                values=list(np.asarray(parsed,dtype=float).reshape(-1))
                break
    if len(values)!=9:
        raise ValueError("Re-evaluation requires 3x3 rotation, e.g. r00..r22 or rotation_matrix JSON.")
    R=np.asarray(values,dtype=float).reshape(3,3)
    if not np.isfinite(R).all() or np.linalg.norm(R.T@R-np.eye(3))>1e-2 or np.linalg.det(R)<0.95:
        raise ValueError(f"Saved rotation not a proper SO(3) matrix: {R}")
    return R.astype(np.float32)


def _row(x, height_m=0.02):
    center=np.asarray(x["center"],dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all():
        raise ValueError(f"Missing camera-frame center for {x['sample']}/{x['candidate']}")
    width=x["width_m"]
    depth=x["insertion"]
    if not (math.isfinite(width) and math.isfinite(depth) and width>=0 and depth>0):
        raise ValueError(f"Bad width/insertion for {x['sample']}/{x['candidate']}")
    rot=_rotation(x["raw"]).reshape(-1)
    return np.asarray([x["score"],width,height_m,depth,*rot,*center,-1],dtype=np.float32)


def _anno(x):
    raw=x["raw"]
    for key in ("anno_id","annotation_id","frame_idx","frame_id","image_idx"):
        if key in raw and str(raw[key]).strip():
            return int(raw[key])
    match=re.search(r"(?:frame|anno)[_-]?(\d+)",x["sample"],flags=re.I)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer annotation id for {x['sample']}; supply anno_id.")


def _stratified_candidates(items,n,rng):
    order=sorted(items,key=lambda r:(r["score"],r["candidate"]))
    n=min(len(order),n)
    if n==0:
        return []
    parts=np.array_split(np.arange(len(order)),n)
    return [order[int(rng.choice(p))] for p in parts]


def make_controlled(rows, *, widths_m, depths_m, num_anchors, seed,
                    original_condition="native", max_width_m=.1):
    if num_anchors<=0:
        raise ValueError("num_anchors must be positive")
    if any(w<=0 or w>max_width_m+1e-8 for w in widths_m):
        raise ValueError("Requested width exceeds configured physical gripper max width.")
    if any(d<=0 for d in depths_m):
        raise ValueError("Insertion depths must be positive")
    grouped=defaultdict(list)
    for row in rows:
        if row["condition"]==original_condition:
            grouped[row["sample"]].append(row)
    if not grouped:
        raise ValueError(f"No rows with condition {original_condition!r}")
    rng=np.random.default_rng(seed)
    selected=[]
    for sample,items in sorted(grouped.items()):
        chosen=_stratified_candidates(items,num_anchors,rng)
        for item in chosen:
            native=_row(item)
            parent=str(item["candidate"])
            for w in [float(native[1]), *widths_m]:
                new=native.copy();new[1]=float(w)
                selected.append((sample,item,"width_sweep",parent,new,"predicted" if abs(w-native[1])<=1e-7 else "fixed"))
            for d in depths_m:
                new=native.copy();new[3]=float(d)
                selected.append((sample,item,"insertion_sweep_quality_only",parent,new,"fixed"))
    # Keep unique physical actions per sample and condition, but permit the
    # same action to appear in different conditions for paired comparisons.
    out=[];seen=set()
    for sample,orig,condition,parent,row,width_mode in selected:
        key=(sample,condition,physical_action_id(row))
        if key in seen:
            continue
        seen.add(key)
        out.append((sample,orig,condition,parent,row,width_mode))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_csv",required=True)
    p.add_argument("--output_csv",required=True)
    p.add_argument("--output_manifest",default=None)
    p.add_argument("--columns_json",default=None)
    p.add_argument("--dataset_root",required=True)
    p.add_argument("--camera",default="realsense")
    p.add_argument("--native_condition",default="native")
    p.add_argument("--num_anchors_per_frame",type=int,default=16)
    p.add_argument("--width_grid",default="0.04,0.06,0.08,0.10")
    p.add_argument("--insertion_grid",default="0.01,0.02,0.03,0.04")
    p.add_argument("--grasp_max_width_m",type=float,default=.1)
    p.add_argument("--width_unit",choices=("auto","m","mm"),default="auto")
    p.add_argument("--seed",type=int,default=20261106)
    p.add_argument("--fc_mode",choices=("official","reuse_contacts"),default="reuse_contacts")
    p.add_argument("--verify_n",type=int,default=10)
    p.add_argument("--eval_chunk",type=int,default=256)
    args=p.parse_args()

    widths=[float(x) for x in args.width_grid.split(",") if x.strip()]
    depths=[float(x) for x in args.insertion_grid.split(",") if x.strip()]
    rows,schema=load_candidates(args.input_csv,args.columns_json,args.width_unit)
    actions=make_controlled(rows,widths_m=widths,depths_m=depths,
                            num_anchors=args.num_anchors_per_frame,seed=args.seed,
                            original_condition=args.native_condition,max_width_m=args.grasp_max_width_m)
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
    evaluator={}
    by_frame=defaultdict(list)
    for action in actions:
        sample,original,_,_,_,_=action
        scene=original["scene"]
        if scene is None:
            raise ValueError("Exact evaluator requires scene_id")
        by_frame[(sample,scene,_anno(original),original["split"])].append(action)
    output=[]
    for (sample,scene,anno,split),items in sorted(by_frame.items()):
        if split not in evaluator:
            evaluator[split]=ExactGraspNetActionEvaluator(
                args.dataset_root,args.camera,split="test_"+split if split in ("seen","similar","novel") else split,
                collision_chunk=args.eval_chunk,fc_mode=args.fc_mode,verify_n=args.verify_n,strict=True)
        unique={}
        for _,_,_,_,grasp,_ in items:
            unique.setdefault(physical_action_id(grasp),grasp)
        ids=sorted(unique)
        values=np.stack([unique[x] for x in ids])
        labels=evaluator[split].evaluate(scene,anno,values)
        label_by_id={k:i for i,k in enumerate(ids)}
        for _,source,condition,parent,row,width_mode in items:
            pid=physical_action_id(row)
            i=label_by_id[pid];mu=float(labels.friction[i])
            bits={f"success_mu_{m:.1f}":int(mu>0 and mu<=m+1e-7) for m in MUS}
            record={
                "sample_id":sample, "scene_id":scene,"anno_id":anno,"split":split,
                "candidate_id":pid,"parent_id":parent,"condition":condition,
                "intervention":condition,"score":float(row[0]),
                "score_semantics":"native_frozen_not_depth_specific" if "quality_only" in condition else "native_width_independent",
                "width_m":float(row[1]),"width_mode":width_mode,
                "insertion_depth_m":float(row[3]),"height_m":float(row[2]),
                "center_x":float(row[13]),"center_y":float(row[14]),"center_z":float(row[15]),
                **{f"r{i0}{j0}":float(row[4+i0*3+j0]) for i0 in range(3) for j0 in range(3)},
                "friction":mu,"quality":float(np.mean(list(bits.values()))),
                "collision_or_empty":int(labels.collision_or_empty[i]),
                "pure_collision":int(labels.pure_collision[i]),
                "empty":int(labels.empty[i]), **bits,
            }
            output.append(record)
        print(f"[P2 CSV] {sample}: {len(items)} rows, {len(ids)} unique physical actions",flush=True)
    dest=Path(args.output_csv)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]))
        writer.writeheader();writer.writerows(output)
    manifest={
        "input_csv":args.input_csv,"output_csv":str(dest),
        "input_schema":schema,"source_condition":args.native_condition,
        "source_sampling":"score-rank stratified within each frame",
        "num_anchors_per_frame":args.num_anchors_per_frame,
        "width_grid_m":widths,"insertion_grid_m":depths,
        "grasp_max_width_m":args.grasp_max_width_m,
        "evaluator":"GraspNet CAD/DexNet exact action; no model-free collision prefilter",
        "insertion_score_semantics":"native frozen score, NOT insertion-depth-specific scoring",
        "training_or_network_forward":False,
        "num_rows":len(output),"num_frames":len(by_frame),
    }
    man=Path(args.output_manifest) if args.output_manifest else dest.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Saved {dest} and {man}")


if __name__=="__main__":
    main()
