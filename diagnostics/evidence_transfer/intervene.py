"""Frozen, no-training GVAR evidence stress interventions on known exact actions.

Purpose: identify whether a trained 'all' score responds to reader/role/relative
geometry groups. These are out-of-distribution diagnostic interventions, NOT
causal evidence that one group represents observed collision or grasp force.
No test labels are consumed by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from diagnostics.evidence_decodability.infer import load_probe
from diagnostics.evidence_decodability.p0_match import load_manifest, read_bank
from diagnostics.evidence_decodability.train import _frame_subset, split_entries
from .contract import save_json, sha256

ROLE_NAMES=("left_contact","right_contact","closing","finger_body","palm","approach")
INTERVENTIONS=(
    "unaltered","mask_pred_geometry","mask_pre_reader","mask_post_reader",
    "mask_reader_delta","mask_all_roles","shuffle_pred_geometry",
    "shuffle_all_roles",
    *(f"mask_role_{role}" for role in ROLE_NAMES)
)


def intervene(x,mode,seed):
    if mode not in INTERVENTIONS:
        raise ValueError(f"Unknown stress intervention {mode}")
    output=dict(x)
    if mode=="unaltered":
        return output
    fields={
        "mask_pred_geometry":("pred_geometry",),
        "mask_pre_reader":("pre_reader",),
        "mask_post_reader":("post_reader",),
        "mask_reader_delta":("reader_delta",),
        "mask_all_roles":("role_tokens","role_mass","role_valid"),
    }
    if mode in fields:
        for key in fields[mode]:
            output[key]=np.zeros_like(x[key])
    elif mode.startswith("mask_role_"):
        role=mode.removeprefix("mask_role_")
        role_id=ROLE_NAMES.index(role)
        for key in ("role_tokens","role_mass","role_valid"):
            output[key]=np.array(x[key],copy=True)
            output[key][:,role_id,...]=0
    elif mode.startswith("shuffle_"):
        keys=("pred_geometry",) if mode=="shuffle_pred_geometry" else (
            "role_tokens","role_mass","role_valid")
        rng=np.random.default_rng(seed)
        perm=rng.permutation(len(x["query_id"]))
        for key in keys:
            output[key]=np.asarray(x[key])[perm].copy()
    else:
        raise ValueError(f"Missing intervention implementation: {mode}")
    # Exact physical action and quality labels must remain bit-identical.
    for key in ("physical_id","grasps","query_id","depth_id",
                "quality_count","quality_bits"):
        if not np.array_equal(output[key],x[key]):
            raise RuntimeError(f"Intervention changed forbidden physical field {key}")
    return output


@torch.inference_mode()
def run(args):
    mf=load_manifest(args.cache_manifest)
    groups=split_entries(mf)
    if args.partition not in ("validation","reused_audit"):
        raise ValueError("Only scorer-held-out training-domain action labels allowed")
    model,protocol=load_probe(args.weights,args.device)
    if model.mode!="all":
        raise ValueError("Stress experiment requires frozen P1 'all' quality head")
    if protocol.get("cache_manifest_sha256")!=mf["manifest_sha256"]:
        raise ValueError("All head not trained from this exact hashed evidence cache")
    entries=groups[args.partition]
    if args.limit:
        if not 0<args.limit<len(entries):
            raise ValueError("Smoke limit must be smaller than full partition")
        entries=entries[:args.limit]
    root=Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use new stress experiment directory")
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for rec in entries:
        row=rec["row"]
        x=_frame_subset(read_bank(rec),row)
        q=np.asarray(x["quality_count"],dtype=np.int64)/6.
        failure=np.full(len(q),3,dtype=np.int64)
        failure[q>0]=0
        failure[(q==0)&np.asarray(x["empty"],dtype=bool)]=2
        failure[(q==0)&np.asarray(x["pure_collision"],dtype=bool)]=1
        native_high=(np.asarray(x["native_score_percentile"])>=.75)
        base=model(x).sigmoid().cpu().numpy()
        for mode in INTERVENTIONS:
            salt=sum(ord(c) for c in mode)
            inp=intervene(x,mode,int(row["scene"])*1009+int(row["anno"])+salt)
            p=model(inp).sigmoid().cpu().numpy()
            if p.shape!=base.shape or not np.isfinite(p).all():
                raise RuntimeError("Nonfinite intervention quality scores")
            for cause in range(4):
                mask=failure==cause
                if not mask.any():continue
                top=np.argsort(-p)[:10]
                rows.append(dict(domain=row["domain"],scene=int(row["scene"]),
                    anno=int(row["anno"]),intervention=mode,fail_class=cause,
                    actions=int(mask.sum()),
                    mean_score_shift=float((p[mask]-base[mask]).mean()),
                    mse_shift=float(((p[mask]-q[mask])**2-
                                     (base[mask]-q[mask])**2).mean()),
                    high_native_actions=int((mask&native_high).sum()),
                    top10_q=float(q[top].mean()),
                    top10_zero_rate=float((q[top]==0).mean())))
    fields=list(rows[0])
    with (root/"per_frame_cause.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=fields)
        wr.writeheader();wr.writerows(rows)
    result={}
    for mode in INTERVENTIONS:
        subset=[r for r in rows if r["intervention"]==mode]
        by_cause={}
        for cause in range(4):
            records=[r for r in subset if r["fail_class"]==cause]
            if not records:continue
            by_cause[str(cause)]=dict(
                frames_with_class=len(records),
                mean_score_shift=float(np.average([r["mean_score_shift"] for r in records],
                    weights=[r["actions"] for r in records])),
                delta_mse=float(np.average([r["mse_shift"] for r in records],
                    weights=[r["actions"] for r in records])))
        # top10 duplicates in class rows; use one record per frame.
        unique={(r["domain"],r["scene"],r["anno"]):r for r in subset}
        result[mode]=dict(
            by_failure=by_cause,
            top10_quality=float(np.mean([r["top10_q"] for r in unique.values()])),
            top10_q0_rate=float(np.mean([r["top10_zero_rate"] for r in unique.values()])),
            frames=len(unique))
    out=dict(schema="gvar-evidence-transfer-v1",kind="P0_frozen_evidence_intervention",
        cache_manifest_sha256=mf["manifest_sha256"],head_sha256=sha256(args.weights),
        partition=args.partition,frames=len(entries),complete=not bool(args.limit),
        interventions=result,role_names=ROLE_NAMES,
        notes=["All interventions preserve executed grasp geometry and quality labels",
               "Zeroing/shuffling features is out-of-distribution; not causal proof of semantic role",
               "No new CAD labels, no retraining, no official test AP",
               "Reused audit scenes have previously been inspected"])
    save_json(root/"summary.json",out)
    lines=["# 冻结 all 评分器的物理证据干预审计","",
           "仅训练域留出场景；动作不变、权重不变。",
           "零化/置乱为OOD敏感性测试，不可直接解释为真实物理因果贡献。","",
           "| 干预 | Top10平均Q | Top10零质量率 | 帧数 |",
           "|---|---:|---:|---:|"]
    for key,item in result.items():
        lines.append(f"| {key} | {item['top10_quality']:.5f} | "
                     f"{item['top10_q0_rate']:.5f} | {item['frames']} |")
    (root/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-manifest",required=True)
    p.add_argument("--weights",required=True)
    p.add_argument("--partition",choices=("validation","reused_audit"),default="validation")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--output",required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
