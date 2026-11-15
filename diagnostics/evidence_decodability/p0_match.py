"""P0: conditional, paired Old-vs-Expanded scorer diagnostics on identical actions.

No new training and NO official test labels. Uses only signed train-scene
native-action evidence caches. The reused audit scenes 90..99 are exploratory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from diagnostics.decision_supervision.core import BASE_SHA, QualityProbe
from diagnostics.candidate_conflict.core import scene_cluster_bootstrap
from .core import SCHEMA, check_bank, digest, failure_class, quality_stats, sha256, write_json


def load_manifest(path, require_complete=True):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    signature=obj.pop("manifest_sha256",None)
    if signature!=digest(obj) or obj.get("schema")!=SCHEMA or obj.get("kind")!="evidence-cache-manifest":
        raise ValueError("Evidence manifest signature/schema failed")
    obj["manifest_sha256"]=signature
    if require_complete and (not obj.get("complete") or len(obj["records"])!=400):
        raise ValueError("P0 requires all 400 training-domain frames, not a smoke cache")
    return obj


def read_bank(entry):
    path=Path(entry["path"])
    if not path.exists() or sha256(path)!=entry["sha256"]:
        raise ValueError(f"Evidence cache changed: {path}")
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    check_bank(data)
    return data


def load_old_head(spec,device):
    path=Path(spec["file"])
    if sha256(path)!=spec["sha256"]:
        raise ValueError(f"Old/Expanded checkpoint SHA mismatch: {path}")
    ck=torch.load(path,map_location="cpu",weights_only=False)
    config=ck.get("config",{})
    if config.get("base_checkpoint_sha256")!=BASE_SHA:
        raise ValueError("Scorer checkpoint not bound to accepted MIX50 GVAR e15 base")
    if config.get("test_labels_used",False) is not False:
        raise ValueError("Scorer used official test labels")
    mode=config.get("mode","score_feature_geometry")
    if mode not in ("score_feature_geometry",):
        raise ValueError("Expected same 5,633-parameter feature+geometry readout")
    hidden=int(config.get("hidden",64))
    model=QualityProbe(mode=mode,hidden=hidden).to(device)
    state=ck.get("model")
    if not isinstance(state,dict):
        raise ValueError("Scorer checkpoint missing strict model state")
    model.load_state_dict(state,strict=True)
    if sum(p.numel() for p in model.parameters())!=5633:
        raise ValueError("Unexpected scorer capacity; refusing unfair comparison")
    model.eval().requires_grad_(False)
    return model,dict(checkpoint_sha256=spec["sha256"],
                      source_config=config,epoch=ck.get("epoch"))


def _infer(model,bank,device):
    feat=torch.as_tensor(bank["score64"],device=device,dtype=torch.float32)
    logit=torch.as_tensor(bank["base_logit"],device=device,dtype=torch.float32)
    grasps=torch.as_tensor(bank["grasps"],device=device,dtype=torch.float32)
    depth=torch.as_tensor(bank["depth_id"],device=device,dtype=torch.long)
    with torch.inference_mode():
        pred=model(feat,logit,grasps,depth).sigmoid().cpu().numpy()
    if pred.shape!=(len(feat),) or not np.isfinite(pred).all():
        raise ValueError("Invalid matched scorer prediction")
    return pred


def _strata(x):
    count=x["quality_count"].astype(int)
    failure=failure_class(count,x["pure_collision"],x["empty"])
    percentile=np.asarray(x["native_score_percentile"],dtype=float)
    native_group=np.minimum((percentile*4).astype(int),3)
    role=x["nms_role"].astype(int)
    source=x["source_new"].astype(int)
    sensor=x["sensor_valid"].astype(int)
    return dict(native_quartile=native_group,failure=failure,
                role=role,source=source,sensor_valid=sensor,
                quality_group=np.where(count==0,0,np.where(count<4,1,2)))


def _inversion_counts(pold,pnew,x):
    """Within-frame q0 versus positive conflicts and high-native-score pairs.

    Counts reflect only labeled source bank; geometry graph edges are not
    necessarily native historical suppressor assignments.
    """
    q=x["quality_count"].astype(int)
    score_pct=np.asarray(x["native_score_percentile"])
    role=x["nms_role"].astype(int)
    high=(score_pct>=.75) & (x["sensor_valid"].astype(bool))
    bad=np.flatnonzero((q==0)&high)
    good=np.flatnonzero((q>0)&high)
    if not len(bad) or not len(good):
        return dict(pairs=0,wrong_old=0,wrong_expanded=0,
                    newly_wrong=0,rescued=0)
    i=np.repeat(bad,len(good))
    j=np.tile(good,len(bad))
    # Restrict the diagnostic to candidates belonging to the top50/neighbor
    # region in the accepted native NMS scene.
    keep=(role[i]>0)&(role[j]>0)
    i,j=i[keep],j[keep]
    if not len(i):
        return dict(pairs=0,wrong_old=0,wrong_expanded=0,
                    newly_wrong=0,rescued=0)
    wrong_old=pold[i]>=pold[j]
    wrong_expanded=pnew[i]>=pnew[j]
    return dict(pairs=len(i),wrong_old=int(wrong_old.sum()),
                wrong_expanded=int(wrong_expanded.sum()),
                newly_wrong=int((~wrong_old&wrong_expanded).sum()),
                rescued=int((wrong_old&~wrong_expanded).sum()))


def _run_group(records,models,device,stage):
    per_frame=[];per_bucket=[]
    for rec in records:
        row=rec["row"]
        if not (stage[0]<=int(row["scene"])<stage[1]):
            continue
        bank=read_bank(rec)
        pold=_infer(models["old"],bank,device)
        pexp=_infer(models["expanded"],bank,device)
        q=np.asarray(bank["quality_count"],dtype=float)/6.
        old=quality_stats(pold,q)
        expanded=quality_stats(pexp,q)
        inverse=_inversion_counts(pold,pexp,bank)
        out=dict(domain=row["domain"],scene=int(row["scene"]),anno=int(row["anno"]),
                 **{f"old_{k}":v for k,v in old.items()},
                 **{f"expanded_{k}":v for k,v in expanded.items()},
                 **{f"delta_{k}":expanded[k]-old[k]
                    for k in ("mse","pair_agreement","top10_quality","top10_zero")
                    if expanded[k] is not None and old[k] is not None},
                 **inverse)
        per_frame.append(out)
        groups=_strata(bank)
        # Conditional paired comparison. No reweighting by new/old train arm
        # composition: every cell contains exactly the same physical actions.
        keys=list(groups)
        composite=np.stack([groups[k] for k in keys],axis=1)
        uniq,indices=np.unique(composite,axis=0,return_inverse=True)
        for index,values in enumerate(uniq):
            ix=np.flatnonzero(indices==index)
            if len(ix)==0:continue
            per_bucket.append(dict(domain=row["domain"],scene=int(row["scene"]),
                anno=int(row["anno"]),count=len(ix),
                **{k:int(v) for k,v in zip(keys,values)},
                old_score=float(pold[ix].mean()),
                expanded_score=float(pexp[ix].mean()),
                mean_quality=float(q[ix].mean()),
                old_mse=float(np.mean((pold[ix]-q[ix])**2)),
                expanded_mse=float(np.mean((pexp[ix]-q[ix])**2)),
                delta_mse=float(np.mean((pexp[ix]-q[ix])**2-(pold[ix]-q[ix])**2))))
    return per_frame,per_bucket


def _csv(path,rows):
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--cache-manifest",required=True)
    a.add_argument("--head-map",required=True,
                   help="JSON seed -> {old:{file,sha256}, expanded:{file,sha256}}")
    a.add_argument("--output",required=True)
    a.add_argument("--device",default="cpu")
    a.add_argument("--min-bucket-actions",type=int,default=20)
    args=a.parse_args()
    mf=load_manifest(args.cache_manifest)
    heads=json.loads(Path(args.head_map).read_text(encoding="utf-8"))
    if not heads or any(set(v)!={"old","expanded"} for v in heads.values()):
        raise ValueError("Expected old/expanded heads for every predefined seed")
    root=Path(args.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    all_frames=[];all_buckets=[];head_info={}
    device=torch.device(args.device)
    for seed,arms in sorted(heads.items()):
        models={};head_info[seed]={}
        for name,spec in arms.items():
            models[name],head_info[seed][name]=load_old_head(spec,device)
        for stage,interval in (("fit",(0,80)),("validation",(80,90)),
                               ("reused_audit",(90,100))):
            frames,buckets=_run_group(mf["records"],models,device,interval)
            for rec in frames:rec.update(seed=seed,stage=stage)
            for rec in buckets:rec.update(seed=seed,stage=stage)
            all_frames.extend(frames);all_buckets.extend(buckets)
            print(f"[P0] {seed} {stage}: {len(frames)} paired scenes/domain frames",flush=True)
    _csv(root/"per_frame.csv",all_frames)
    _csv(root/"per_stratum_frame.csv",all_buckets)
    # Aggregate by scene and fixed head seed, avoiding candidate/frame pseudo-IID.
    metrics=("delta_mse","delta_pair_agreement","delta_top10_quality",
             "delta_top10_zero")
    summary={}
    for stage in ("fit","validation","reused_audit"):
        subset=[r for r in all_frames if r["stage"]==stage]
        by_metric={}
        for key in metrics:
            scene_ids=sorted({r["scene"] for r in subset})
            per_scene={int(scene):float(np.mean([r[key] for r in subset
                                        if r["scene"]==scene and key in r]))
                       for scene in scene_ids}
            by_metric[key]=scene_cluster_bootstrap(per_scene,5000,seed=20261115)
        inversion={k:int(sum(r[k] for r in subset)) for k in
                   ("pairs","wrong_old","wrong_expanded","newly_wrong","rescued")}
        summary[stage]=dict(metrics=by_metric,inversion=inversion,frames=len(subset))
    # Joined strata are potentially sparse; suppress unstable cells from
    # interpretation but retain their raw rows for independent inspection.
    strata_rows=[]
    groups=defaultdict(list)
    for r in all_buckets:
        key=tuple(r[k] for k in
                  ("stage","native_quartile","quality_group","failure","role","source","sensor_valid"))
        groups[key].append(r)
    for k,rs in sorted(groups.items()):
        n=sum(r["count"] for r in rs)
        if n<args.min_bucket_actions:continue
        strata_rows.append(dict(
            stage=k[0],native_quartile=k[1],quality_group=k[2],failure=k[3],
            nms_role=k[4],source_new=k[5],sensor_valid=k[6],
            actions=n,mean_quality=float(np.average([r["mean_quality"] for r in rs],
                                                     weights=[r["count"] for r in rs])),
            delta_mse=float(np.average([r["delta_mse"] for r in rs],
                                       weights=[r["count"] for r in rs])),
            score_shift=float(np.average(
                [r["expanded_score"]-r["old_score"] for r in rs],
                weights=[r["count"] for r in rs]))))
    _csv(root/"strata_summary.csv",strata_rows)
    write_json(root/"summary.json",dict(schema=SCHEMA,kind="P0_matched_distribution",
                scope="train-scene conditional analysis; not official test AP",
                cached_actions_fixed=True,reference_pool_hash=mf["manifest_sha256"],
                head_map_sha256=sha256(args.head_map),
                heads=head_info,min_bucket_actions=args.min_bucket_actions,
                stages=summary,
                limitations=["The original frozen base saw all train scenes",
                             "Scenes 90..99 are previously inspected, not new confirmation",
                             "High-rank inversions are potential decision errors, not actual compiled NMS suppressor events",
                             "No new training and no test CAD labels were used"]))
    print(json.dumps({stage:{k:v["mean"] for k,v in d["metrics"].items()}
                      for stage,d in summary.items()},indent=2))


if __name__=="__main__":
    main()
