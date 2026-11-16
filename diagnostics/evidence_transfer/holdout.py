"""P1: provenance-gated evaluation of frozen native/score64/all on new scenes.

Independent validation cannot be manufactured by renaming GraspNet Seen/Novel:
the historic base trained scenes 0..99; test scenes 100..189 have repeatedly
been inspected during method development. A separate signed provenance ledger
and a previously uninspected dataset are required for an 'untouched' claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostics.decision_supervision.core import BASE_SHA
from .contract import (
    SCHEMA, METHODS, digest, failure_classes, load_manifest, load_frame,
    save_json, sha256,
)

PROV_SCHEMA="gvar-base-development-ledger-v1"


def scene_key(dataset,scene):
    return f"{dataset}::{scene}"


def read_ledger(path):
    path=Path(path)
    x=json.loads(path.read_text(encoding="utf-8"))
    signature=x.pop("ledger_sha256",None)
    if signature!=digest(x) or x.get("schema")!=PROV_SCHEMA:
        raise ValueError("Base/development provenance ledger is not signed")
    x["ledger_sha256"]=signature
    if x.get("base_checkpoint_sha256")!=BASE_SHA:
        raise ValueError("Holdout ledger belongs to different frozen GVAR base")
    for required in ("base_training_scene_keys","development_scene_keys",
                     "base_training_manifest_path","base_training_manifest_sha256",
                     "development_log_path","development_log_sha256",
                     "object_overlap_review"):
        if required not in x:
            raise ValueError(f"Missing mandatory holdout provenance: {required}")
    if (not x["base_training_scene_keys"] or not x["development_scene_keys"] or
            not all(len(str(x[k]))==64 for k in
                ("base_training_manifest_sha256","development_log_sha256"))):
        raise ValueError("Empty/invalid training and development ledgers")
    for path_key,sha_key in (
        ("base_training_manifest_path","base_training_manifest_sha256"),
        ("development_log_path","development_log_sha256")):
        actual=Path(x[path_key])
        if not actual.is_file() or sha256(actual)!=x[sha_key]:
            raise ValueError(f"Training/development evidence file missing or changed: {actual}")
    for key,filekey in (
        ("base_training_scene_keys","base_training_manifest_path"),
        ("development_scene_keys","development_log_path")):
        val=x[key]
        if len(set(val))!=len(val):
            raise ValueError(f"Duplicate provenance scene keys: {key}")
        claimed=json.loads(Path(x[filekey]).read_text(encoding="utf-8"))
        if not isinstance(claimed,list) or set(claimed)!=set(val):
            raise ValueError(f"Signed ledger scene keys do not match actual source list: {filekey}")
    # Verified current-project lower bound. No one can declare the already
    # explored 90 GraspNet test scenes to be an untouched holdout.
    known={scene_key("GraspNet",s) for s in range(100,190)}
    if not known.issubset(set(x["development_scene_keys"])):
        raise ValueError("Current project already inspected GraspNet scenes 100..189")
    trained={scene_key("GraspNet",s) for s in range(100)}
    if not trained.issubset(set(x["base_training_scene_keys"])):
        raise ValueError("Frozen base training registry must include GraspNet 0..99")
    return x


def gate(manifest, ledger, *, require_untouched):
    frame_keys={scene_key(r["dataset"],r["scene"]) for r in manifest["frames"]}
    base=set(ledger["base_training_scene_keys"])
    dev=set(ledger["development_scene_keys"])
    overlapped_training=sorted(frame_keys&base)
    overlapped_development=sorted(frame_keys&dev)
    has_external=all(r["split"]=="external_holdout" for r in manifest["frames"])
    unverified_object_lineage=not bool(ledger["object_overlap_review"].get("documented",False))
    report=dict(
        scenes=len(frame_keys),frames=len(manifest["frames"]),
        intersects_base_training=overlapped_training,
        intersects_prior_development=overlapped_development,
        external_holdout_split=has_external,
        object_overlap_lineage_documented=not unverified_object_lineage,
        base_unseen=not bool(overlapped_training),
        untouched_scene_claim=(not overlapped_training and
                               not overlapped_development and has_external),
        object_disjoint_proven=False,
        object_disjoint_claimed=bool(
            ledger["object_overlap_review"].get("disjoint_verified",False)),
        provenance_ledger_sha256=ledger["ledger_sha256"],
        limitation="A signed ledger is a reproducibility and contradiction gate, not independent proof that its human-maintained contents are complete.",
    )
    # A human-maintained provenance manifest cannot prove physical
    # object-level CAD independence without a separately executed overlap audit.
    if require_untouched:
        if not report["untouched_scene_claim"]:
            raise ValueError("Previously trained/inspected scenes cannot be independent validation")
        provenance=manifest.get("provenance",{})
        if provenance.get("prior_development_use") is not False:
            raise ValueError("Untouched claim requires explicit prior_development_use=false")
        if provenance.get("dataset_source_sha256") is None:
            raise ValueError("External holdout requires dataset provenance SHA256")
        if not provenance.get("external_evaluator_name"):
            raise ValueError("External holdout must name its actual evaluation/quality protocol")
        if len(str(provenance["dataset_source_sha256"]))!=64:
            raise ValueError("Bad external dataset source SHA256")
    return report


def _mean(q,mask):
    if not mask.any():return None
    return float(q[mask].mean())


def metrics_for_frame(x):
    q=np.asarray(x["quality_bits"]).sum(axis=1)/6.
    cause=failure_classes(x)
    result={}
    for method in METHODS:
        p=np.asarray(x[f"score_{method}"])
        top=np.asarray(x[f"top10_{method}"],dtype=bool)
        result.update({
            f"{method}_mse":float(np.mean((p-q)**2)),
            f"{method}_top10_quality":_mean(q,top),
            f"{method}_top10_q0":_mean((q==0).astype(float),top),
            f"{method}_top10_strong":_mean((q>=5/6-1e-8).astype(float),top),
            f"{method}_top10_collision":_mean((cause==1).astype(float),top),
            f"{method}_top10_empty":_mean((cause==2).astype(float),top),
            f"{method}_q0_score":_mean(p,cause!=0),
        })
    for target in ("mse","top10_quality","top10_q0",
                   "top10_strong","top10_collision","top10_empty","q0_score"):
        if result[f"all_{target}"] is not None and result[f"score64_{target}"] is not None:
            result[f"delta_all_score64_{target}"]=(
                result[f"all_{target}"]-result[f"score64_{target}"])
        if result[f"all_{target}"] is not None and result[f"native_{target}"] is not None:
            result[f"delta_all_native_{target}"]=(
                result[f"all_{target}"]-result[f"native_{target}"])
    result["labeled_actions"]=len(q)
    return result


def _paired_scene_bootstrap(records,field,draws,seed):
    by_scene=defaultdict(list)
    for row in records:
        if row.get(field) is not None:
            by_scene[row["scene"]].append(float(row[field]))
    if not by_scene:
        return dict(mean=None,ci95=None,scenes=0)
    values=np.array([np.mean(v) for v in by_scene.values()],dtype=float)
    if len(values)<2:
        return dict(mean=float(values.mean()),ci95=None,scenes=len(values))
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(values),size=(draws,len(values)))
    return dict(mean=float(values.mean()),
                ci95=np.quantile(values[idx].mean(axis=1),[.025,.975]).tolist(),
                scenes=len(values))


def run(a):
    if a.bootstrap<1:raise ValueError("bootstrap must be >=1")
    mf=load_manifest(a.manifest)
    registry=read_ledger(a.ledger)
    scope=gate(mf,registry,require_untouched=a.require_untouched)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use an independent output root")
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for rec in mf["frames"]:
        x=load_frame(rec)
        row=dict(dataset=rec["dataset"],split=rec["split"],
                 scene=scene_key(rec["dataset"],rec["scene"]),
                 anno=int(rec["anno"]),**metrics_for_frame(x))
        rows.append(row)
    with (root/"per_frame.csv").open("w",newline="",encoding="utf-8") as f:
        keys=sorted(set().union(*(r.keys() for r in rows)))
        wr=csv.DictWriter(f,fieldnames=keys)
        wr.writeheader();wr.writerows(rows)
    report={}
    for dataset,split in sorted({(r["dataset"],r["split"]) for r in rows}):
        part=[r for r in rows if r["dataset"]==dataset and r["split"]==split]
        keys=[k for k in part[0] if k.startswith("delta_")]
        report[f"{dataset}/{split}"]={
            "frames":len(part),
            "scenes":len({r["scene"] for r in part}),
            "paired":{k:_paired_scene_bootstrap(part,k,a.bootstrap,a.seed+i)
                      for i,k in enumerate(sorted(keys))},
            "mean_source_metrics":{k:float(np.mean([r[k] for r in part if r.get(k) is not None]))
                          if any(r.get(k) is not None for r in part) else None
                          for k in part[0] if k.endswith(("_top10_quality","_top10_q0","_top10_strong"))},
        }
    result=dict(schema=SCHEMA,kind="frozen_evidence_independence_audit",
                base_checkpoint_sha256=BASE_SHA,
                input_manifest_sha256=mf["manifest_sha256"],
                ledger_sha256=registry["ledger_sha256"],
                independence_gate=scope,
                interpretation=("PROVISIONALLY_UNTOUCHED_SCENE_HOLDOUT_LEDGER_GATED"
                                if scope["untouched_scene_claim"] and a.require_untouched
                                else "EXPLORATORY_NOT_INDEPENDENT"),
                by_dataset=report,
                caution=["No retraining or model selection using evaluation labels",
                         "Object-level independence is separate from scene-level independence",
                         "Same-action survivor-union quality regression is not official GraspNet AP",
                         "Human-provided provenance declarations require external audit",
                         "Sample size and scene-dependent bootstrap limits apply"])
    save_json(root/"summary.json",result)
    lines=["# P1 冻结 GVAR 独立验证协议检查", "",
           f"状态：{result['interpretation']}；总场景{scope['scenes']}、帧{scope['frames']}。",
           "不按已观察Novel的结果重新调阈值、选择模式或权重。","",
           "| 数据 | ΔTop10Q(all-score64) | ΔTop10强质量比例 | 场景数 |",
           "|---|---:|---:|---:|"]
    for key,v in report.items():
        p=v["paired"]
        def fmt(k):
            val=p.get(k,{}).get("mean")
            return "NA" if val is None else f"{val:+.5f}"
        lines.append(f"| {key} | {fmt('delta_all_score64_top10_quality')} | "
                     f"{fmt('delta_all_score64_top10_strong')} | {v['scenes']} |")
    (root/"RESULTS_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--ledger",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--require-untouched",action="store_true")
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261116)
    run(p.parse_args())


if __name__=="__main__":
    main()
