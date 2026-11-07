#!/usr/bin/env python3
"""P3: replay per-depth predicted-width action labels from frozen P2 CSV.

This avoids rerunning a historical checkpoint if the P2 export already has
  width_predicted_decoded_m
for EVERY insertion-depth bin. The depth-bin CDF scores are copied unchanged,
but the *physical* width is replaced and every new grasp is evaluated with
the same exact CAD/empty/DexNet pipeline, not assigned its old quality label.

The selected original native grasp must match between width protocols.
If the per-depth width channel is absent, fail; do not invent its values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np

from diagnostics.analyze_rayrope_candidates import load_candidates
from diagnostics.eval_rayrope_controlled_sweeps_from_csv import _row, _anno
from diagnostics.rayrope_action_lattice import physical_action_id

MUS=(0.2,0.4,0.6,0.8,1.0,1.2)
FIXED="insertion_sweep_fixed_width"
PRED="insertion_sweep_predicted_width"


def sha256_file(path: str) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _depth_id(row):
    raw=row["raw"].get("depth_id","")
    if str(raw).strip():
        return int(raw)
    depth=row["insertion"]
    d=int(round(depth/.01))-1
    if d<0 or abs(depth-.01*(d+1))>1e-5:
        raise ValueError(f"Unsupported insertion depth {depth!r}")
    return d


def construct_replay(rows,width_column="width_predicted_decoded_m",max_width_m=.1,tol=1e-5):
    """Pure transform. Returns (source_row, new_physical_17D_row) pairs."""
    if max_width_m<=0 or tol<=0:
        raise ValueError("Invalid width cap/tolerance")
    fixed=defaultdict(dict)
    for row in rows:
        if row["condition"]!=FIXED:
            continue
        if not row["parent"]:
            raise ValueError(f"Missing parent_id in {FIXED}")
        if not math.isfinite(row["quality"]):
            raise ValueError("P3 requires previously evaluated fixed-width quality")
        if width_column not in row["raw"]:
            raise KeyError(f"Missing {width_column!r} in fixed-width CSV; rerun historical forward to export per-depth width")
        d=_depth_id(row)
        key=(row["sample"],row["parent"])
        if d in fixed[key]:
            raise ValueError(f"Duplicated insertion bin for {key}/d{d}")
        fixed[key][d]=row
    if not fixed:
        raise ValueError(f"No {FIXED} rows in input")
    outputs=[]
    for key,by_depth in sorted(fixed.items()):
        if len(by_depth)<2:
            raise ValueError(f"Need >=2 insertion bins for {key}")
        records=[by_depth[k] for k in sorted(by_depth)]
        widths=[float(r["width_m"]) for r in records]
        if np.ptp(widths)>tol:
            raise ValueError(f"Fixed-width intervention did not keep width constant: {key}")
        base=_row(records[0],height_m=float(records[0]["raw"].get("height_m") or .02))
        for item in records[1:]:
            candidate=_row(item,height_m=float(item["raw"].get("height_m") or .02))
            if not np.allclose(base[2:3],candidate[2:3],atol=tol,rtol=0):
                raise ValueError(f"Fixed insertion family changes gripper height: {key}")
            if not np.allclose(base[4:16],candidate[4:16],atol=tol,rtol=0):
                raise ValueError(f"Fixed insertion family changes center or rotation: {key}")
        chosen_depth=sorted(by_depth)[int(np.argmax([r["score"] for r in records]))]
        selected_flags=[d for d,r in by_depth.items() if str(r["raw"].get("query_winner","")).strip() in ("1","True","true")]
        if selected_flags and (len(selected_flags)!=1 or selected_flags[0]!=chosen_depth):
            raise ValueError(f"Depth-argmax and native query winner disagree: {key}")
        for d,row in sorted(by_depth.items()):
            raw_width=row["raw"].get(width_column)
            try:
                width=float(raw_width)
            except (TypeError,ValueError):
                raise ValueError(f"Missing or non-numeric {width_column} in {key}/d{d}")
            if not math.isfinite(width) or width<0 or width>max_width_m+tol:
                raise ValueError(f"Bad predicted width {width} m for {key}/d{d}")
            fixed_grasp=_row(row,height_m=float(row["raw"].get("height_m") or .02))
            if not np.isclose(fixed_grasp[1],widths[0],atol=tol,rtol=0):
                raise RuntimeError(f"Physical grasp width differs from CSV's fixed width: {key}/d{d}")
            pred_grasp=fixed_grasp.copy()
            pred_grasp[1]=float(np.clip(width,0,max_width_m))
            if d==chosen_depth and not np.allclose(pred_grasp[1:16],fixed_grasp[1:16],atol=tol,rtol=0):
                raise ValueError(f"Predicted width != fixed native width at selected depth {key}/d{d}")
            if not np.isfinite(pred_grasp).all():
                raise ValueError(f"Nonfinite physical grasp {key}/d{d}")
            outputs.append((row,pred_grasp))
    return outputs


def _write_csv(path,rows,source_columns):
    existing=list(source_columns)
    extras=sorted(set().union(*(set(x) for x in rows))-set(existing))
    fields=existing+extras
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate_csv",required=True)
    p.add_argument("--output_csv",required=True)
    p.add_argument("--output_manifest",default=None)
    p.add_argument("--lattice_manifest_json",default=None)
    p.add_argument("--expected_checkpoint_sha256",default=None)
    p.add_argument("--dataset_root",required=True)
    p.add_argument("--camera",default="realsense")
    p.add_argument("--width_source_column",default="width_predicted_decoded_m")
    p.add_argument("--grasp_max_width_m",type=float,default=.1)
    p.add_argument("--tol_m",type=float,default=1e-5)
    p.add_argument("--fc_mode",choices=("official","reuse_contacts"),default="reuse_contacts")
    p.add_argument("--verify_n",type=int,default=10)
    p.add_argument("--eval_chunk",type=int,default=256)
    p.add_argument("--include_source_rows",type=int,choices=(0,1),default=1)
    args=p.parse_args()

    src=Path(args.candidate_csv)
    rows,schema=load_candidates(str(src),None,"m")
    if any(x["condition"]==PRED for x in rows):
        raise ValueError(f"Input already contains {PRED}; do not duplicate replayed actions")
    manifest_path=Path(args.lattice_manifest_json) if args.lattice_manifest_json else src.with_name("lattice_manifest.json")
    original_manifest=None
    if manifest_path.is_file():
        original_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if original_manifest.get("insertion_sweep_width") not in (None,"fixed"):
            raise ValueError("Source manifest does not declare a fixed-width insertion experiment")
        cap=original_manifest.get("grasp_max_width_m")
        if cap is not None and abs(float(cap)-args.grasp_max_width_m)>args.tol_m:
            raise ValueError("Declared gripper width cap does not match source manifest")
        checkpoint_sha=original_manifest.get("checkpoint_sha256")
        if args.expected_checkpoint_sha256 and checkpoint_sha!=args.expected_checkpoint_sha256:
            raise ValueError("Checkpoint hash mismatch")
    elif args.lattice_manifest_json or args.expected_checkpoint_sha256:
        raise FileNotFoundError(f"Missing provenance manifest: {manifest_path}")

    pending=construct_replay(
        rows,width_column=args.width_source_column,
        max_width_m=args.grasp_max_width_m,tol=args.tol_m,
    )
    # Shared physical candidates are evaluated once per frame/split. No GT or
    # CAD is used to choose width or model scores, only to label new actions.
    by_frame=defaultdict(list)
    for source,grasp in pending:
        if source["scene"] is None:
            raise ValueError("Need scene_id for exact CAD re-evaluation")
        by_frame[(source["sample"],source["split"],source["scene"],_anno(source))].append((source,grasp))

    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
    evaluators={}
    new_rows=[]
    per_frame=[]
    for (sample,split,scene,anno),items in sorted(by_frame.items()):
        eval_split="test_"+split if split in ("seen","similar","novel") else split
        if eval_split not in evaluators:
            evaluators[eval_split]=ExactGraspNetActionEvaluator(
                args.dataset_root,args.camera,split=eval_split,
                collision_chunk=args.eval_chunk,fc_mode=args.fc_mode,
                verify_n=args.verify_n,strict=True,
            )
        unique={}
        for _,grasp in items:
            unique.setdefault(physical_action_id(grasp),grasp)
        ids=sorted(unique)
        grasps=np.stack([unique[cid] for cid in ids])
        result=evaluators[eval_split].evaluate(scene,anno,grasps)
        lookup={cid:i for i,cid in enumerate(ids)}
        for source,grasp in items:
            cid=physical_action_id(grasp)
            i=lookup[cid]
            friction=float(result.friction[i])
            bits={f"success_mu_{mu:.1f}":int(friction>0 and friction<=mu+1e-7) for mu in MUS}
            updated=dict(source["raw"])
            updated.update({
                "condition":PRED,"intervention":PRED,
                "candidate_id":cid,"source_fixed_candidate_id":source["candidate"],
                "width_m":float(grasp[1]),
                "width_mode":"predicted",
                "score":float(source["score"]),
                "quality":float(np.mean(list(bits.values()))),
                "friction":friction,
                "collision_or_empty":int(result.collision_or_empty[i]),
                "pure_collision":int(result.pure_collision[i]),
                "empty":int(result.empty[i]),
                "assigned_obj_id":int(result.assigned_obj[i]),
                "p3_replayed_width":"1",
                "score_semantics":"CDF_depth_specific_score_held_constant_across_width_protocols",
                **bits,
            })
            new_rows.append(updated)
        per_frame.append({"sample_id":sample,"scene_id":scene,"anno_id":anno,
                          "num_predicted_width_records":len(items),
                          "num_distinct_physical_actions":len(ids)})
        print(f"[P3 replay] {sample}: {len(items)} predicted-width records, {len(ids)} unique physical actions",flush=True)

    output=( [x["raw"] for x in rows] if args.include_source_rows else [] )+new_rows
    if not output:
        raise ValueError("No output records")
    _write_csv(args.output_csv,output,schema["headers"])
    provenance={
        "version":"p3-predicted-width-replay-v1",
        "source_candidate_csv":str(src),
        "source_csv_sha256":sha256_file(str(src)),
        "source_manifest":str(manifest_path) if original_manifest else None,
        "source_checkpoint_sha256":(original_manifest or {}).get("checkpoint_sha256"),
        "dataset_root":str(Path(args.dataset_root).resolve()),"camera":args.camera,
        "source_condition":FIXED,"new_condition":PRED,
        "width_source_column":args.width_source_column,
        "grasp_max_width_m":args.grasp_max_width_m,
        "score_semantics":"same native CDF depth-bin score, only physical width changes",
        "quality_semantics":"all new 17D physical actions independently CAD/DexNet re-evaluated",
        "collision_prefilter":False,
        "num_source_rows":len(rows),
        "num_replayed_rows":len(new_rows),
        "num_frames":len(per_frame),
        "fc_mode":args.fc_mode,"verify_n":args.verify_n,
        "frames":per_frame,
        "output_csv":str(args.output_csv),
        "limits":[
            "This reuses per-depth width predictions already exported by the historical frozen forward; does not synthesize new scores.",
            "If width_predicted_decoded_m is absent, rerun the frozen historical model with full CDF endpoints.",
            "CAD labels are offline diagnostics, not inference inputs.",
        ],
    }
    dest=Path(args.output_manifest) if args.output_manifest else Path(args.output_csv).with_suffix(".manifest.json")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output_csv":args.output_csv,"manifest":str(dest),
                      "new_records":len(new_rows),"frames":len(per_frame)},indent=2))


if __name__=="__main__":
    main()
