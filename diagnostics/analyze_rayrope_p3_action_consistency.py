#!/usr/bin/env python3
"""P3: paired insertion-depth / width-protocol action-consistency diagnostics.

Inputs are per-physical-action CAD/Dex-Net labels from P2 lattice runner:
  insertion_sweep_fixed_width, insertion_sweep_predicted_width, width_sweep,
  query_winner. For a matched insertion family the CDF score at each d is
identical across protocols; the ONLY variable is executed width w(d).

No candidate pool mixing, no ground truth in model inference, no AP claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from .analyze_rayrope_candidates import load_candidates, rank_corr
except ImportError:
    from analyze_rayrope_candidates import load_candidates, rank_corr

MUS=(0.2,0.4,0.6,0.8,1.0,1.2)
FIXED="insertion_sweep_fixed_width"
PRED="insertion_sweep_predicted_width"
WIDTH="width_sweep"
NATIVE="query_winner"
LABELS=("collision","empty_grasp","force_closure_failure","success","unknown")


def _number(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def mean(values):
    a=np.asarray([v for v in values if v is not None and math.isfinite(float(v))],dtype=float)
    return float(a.mean()) if len(a) else None


def _metadata(row):
    raw=row["raw"]
    return {
        "sample":row["sample"],"split":row["split"],"scene":row["scene"],
        "parent":row["parent"],"depth":row["insertion"],
        "query_id":raw.get("query_id",""),"angle_id":raw.get("angle_id",""),
    }


def _frame_scene(row):
    if row["scene"] is None or row["split"]=="unknown":
        raise ValueError("P3 requires stable scene_id and split in every row")
    return (row["split"],row["scene"])


def scene_cluster_bootstrap(records,metric,n_boot=5000,seed=20261107):
    """Equal split / scene weighting; cluster related family records together.

    Returns right-minus-left when metric is a paired delta. No independent
    grasp/frame pseudo-replication or training seed variance estimate.
    """
    if n_boot<0:
        raise ValueError("n_boot must be >=0")
    scenes=defaultdict(list)
    sample_ids=set()
    for x in records:
        val=_number(x.get(metric))
        if val is None:
            continue
        scenes[_frame_scene(x)].append(val)
        sample_ids.add(x["sample"])
    if not scenes:
        return {"mean":None,"ci95":None,"num_scenes":0,"num_frames":0,
                "num_records":0}
    split_values=defaultdict(list)
    for (split,scene),values in sorted(scenes.items()):
        split_values[split].append(float(np.mean(values)))
    estimate=float(np.mean([np.mean(v) for v in split_values.values()]))
    rng=np.random.default_rng(seed)
    draws=np.empty(n_boot,dtype=float)
    for i in range(n_boot):
        draws[i]=float(np.mean([
            np.mean(rng.choice(v,size=len(v),replace=True))
            for v in split_values.values()
        ]))
    return {
        "mean":estimate,
        "ci95":[float(x) for x in np.quantile(draws,[0.025,0.975])] if n_boot else None,
        "num_scenes":len(scenes),"num_frames":len(sample_ids),
        "num_records":sum(len(v) for v in scenes.values()),
        "bootstrap_unit":"scene_cluster_split_stratified",
    }


def _rot(row):
    raw=row["raw"]
    names=[f"r{i}{j}" for i in range(3) for j in range(3)]
    if not all(key in raw for key in names):
        raise ValueError("P3 requires explicit r00..r22 rotation fields for geometry matching")
    vals=np.asarray([_number(raw.get(k)) for k in names],dtype=object)
    if any(v is None for v in vals):
        raise ValueError("Nonfinite or missing rotation value")
    return np.asarray(vals,dtype=float)


def _depth_id(row):
    x=row["raw"].get("depth_id","")
    if str(x).strip():
        return int(x)
    value=float(row["insertion"])
    d=int(round(value/.01))-1
    if d<0 or not np.isclose(value,(d+1)*.01,atol=1e-5,rtol=0):
        raise ValueError(f"Insertion depth is not an official 1cm bin: {value}")
    return d


def _group(rows, condition):
    groups=defaultdict(dict)
    for row in rows:
        if row["condition"]!=condition:
            continue
        if not row["parent"]:
            raise ValueError(f"{condition} requires a stable parent_id")
        if not math.isfinite(row["quality"]):
            raise ValueError(f"Missing exact CAD quality in {condition}, sample={row['sample']}")
        key=(row["sample"],row["parent"])
        d=_depth_id(row)
        if d in groups[key]:
            raise ValueError(f"Duplicate insertion depth in {condition}/{key}/{d}")
        groups[key][d]=row
    return groups


def _validate_pair(a,b,key,depth_id,tol=1e-5):
    if not np.allclose(a["center"],b["center"],atol=tol,rtol=0):
        raise ValueError(f"Center mismatch {key}/d{depth_id}")
    if not np.allclose(_rot(a),_rot(b),atol=tol,rtol=0):
        raise ValueError(f"Rotation mismatch {key}/d{depth_id}")
    if abs(a["insertion"]-b["insertion"])>tol:
        raise ValueError(f"Insertion mismatch {key}/d{depth_id}")
    if abs(a["score"]-b["score"])>tol:
        raise ValueError(f"CDF score changed across width modes {key}/d{depth_id}")
    height_a=_number(a["raw"].get("height_m"))
    height_b=_number(b["raw"].get("height_m"))
    if height_a is not None and height_b is not None and abs(height_a-height_b)>tol:
        raise ValueError(f"Gripper height changed across width protocols {key}/d{depth_id}")
    expected_width=_number(b["raw"].get("width_predicted_decoded_m"))
    if expected_width is not None and abs(expected_width-b["width_m"])>tol:
        raise ValueError(f"Predicted-width row violates exported depth-bin width {key}/d{depth_id}")
    if a["raw"].get("angle_id","")!=b["raw"].get("angle_id",""):
        raise ValueError(f"Angle ID mismatch {key}/d{depth_id}")
    if a["raw"].get("query_id","")!=b["raw"].get("query_id",""):
        raise ValueError(f"Query ID mismatch {key}/d{depth_id}")
    if not a["candidate"] or not b["candidate"]:
        raise ValueError("Missing physical candidate ID")


def _selection(family):
    depth_ids=sorted(family)
    vals=[family[d]["score"] for d in depth_ids]
    chosen=depth_ids[int(np.argmax(vals))]
    quality=[family[d]["quality"] for d in depth_ids]
    oracle=depth_ids[int(np.argmax(quality))]
    losers=[d for d in depth_ids if d!=chosen]
    if not losers:
        raise ValueError("Need >=2 insertion bins to define a hard negative")
    hard=max(losers,key=lambda d:(family[d]["score"],-d))
    return chosen,oracle,hard


def _depth_pairwise_agreement(family,tol=1e-8):
    """Within-angle score/quality ordering over distinct-quality depth pairs.

    Quality ties are excluded. A score tie receives half credit; we report
    coverage so all-zero families never count as correct rankings.
    """
    ids=sorted(family)
    concordant=0;discordant=0;score_tied=0;quality_distinct=0
    for i,da in enumerate(ids):
        for db in ids[i+1:]:
            qa,qb=family[da]["quality"],family[db]["quality"]
            ds=family[da]["score"]-family[db]["score"]
            dq=qa-qb
            if abs(dq)<=tol:
                continue
            quality_distinct+=1
            if abs(ds)<=tol:
                score_tied+=1
            elif ds*dq>0:
                concordant+=1
            else:
                discordant+=1
    accuracy=((concordant+.5*score_tied)/quality_distinct
              if quality_distinct else None)
    return {
        "comparable_pairs":quality_distinct,
        "concordant_pairs":concordant,
        "discordant_pairs":discordant,
        "score_tied_pairs":score_tied,
        "pairwise_accuracy":accuracy,
    }


def _family_stats(family,condition):
    depths=sorted(family)
    if len(depths)<2:
        raise ValueError("Insertion family must contain at least two distinct depths")
    reference=family[depths[0]]
    center0=np.asarray(reference["center"],dtype=float)
    rot0=_rot(reference)
    for depth_id in depths[1:]:
        item=family[depth_id]
        if (not np.allclose(np.asarray(item["center"],dtype=float),center0,atol=1e-5,rtol=0)
            or not np.allclose(_rot(item),rot0,atol=1e-5,rtol=0)):
            raise ValueError("Insertion sweep family changes center or rotation across depth")
        h_ref=_number(reference["raw"].get("height_m"))
        h_item=_number(item["raw"].get("height_m"))
        if h_ref is not None and h_item is not None and abs(h_ref-h_item)>1e-5:
            raise ValueError("Insertion sweep changes gripper height across depth")
    selected,oracle,hard=_selection(family)
    winner_flags=[d for d in depths
                  if str(family[d]["raw"].get("query_winner","")).strip().lower() in ("1","true")]
    if winner_flags and (len(winner_flags)!=1 or winner_flags[0]!=selected):
        raise ValueError("Full CDF score argmax inconsistent with exported native query winner")
    x,y,z=family[selected],family[oracle],family[hard]
    agreement=_depth_pairwise_agreement(family)
    spearman=rank_corr([family[d]["score"] for d in depths],
                       [family[d]["quality"] for d in depths])
    if any(row["condition"]!=condition for row in family.values()):
        raise RuntimeError("Mixed width protocols inside family")
    return {
        **_metadata(x),"condition":condition,"num_depth_bins":len(depths),
        "selected_depth_id":selected,"oracle_depth_id":oracle,
        "hard_negative_depth_id":hard,
        "selected_depth_m":x["insertion"],"oracle_depth_m":y["insertion"],
        "selected_score":x["score"],"hard_negative_score":z["score"],
        "score_margin_to_hard":x["score"]-z["score"],
        "selected_quality":x["quality"],"oracle_quality":y["quality"],
        "hard_negative_quality":z["quality"],
        "regret":y["quality"]-x["quality"],
        "hard_negative_quality_minus_selected":z["quality"]-x["quality"],
        "hard_negative_better":float(z["quality"]>x["quality"]+1e-8),
        "oracle_nonselected":float(oracle!=selected),
        "selected_failure":x["failure"],"oracle_failure":y["failure"],
        "hard_negative_failure":z["failure"],
        "selected_width_m":x["width_m"],
        "oracle_width_m":y["width_m"],
        "quality_varies":float(max(row["quality"] for row in family.values())-
                               min(row["quality"] for row in family.values())>1e-8),
        "depth_score_quality_spearman":spearman,
        "depth_pairwise_accuracy":agreement["pairwise_accuracy"],
        "depth_comparable_pairs":agreement["comparable_pairs"],
        "depth_discordant_pairs":agreement["discordant_pairs"],
    }


def _depth_summary(groups,condition):
    bins=defaultdict(list)
    for group in groups.values():
        for d,row in group.items():
            bins[d].append(row)
    output={}
    for d,rows in sorted(bins.items()):
        output[f"depth_{d+1}cm"]={
            "num_actions":len(rows),
            "quality":mean([r["quality"] for r in rows]),
            "model_score":mean([r["score"] for r in rows]),
            "failure_counts":{k:int(sum(r["failure"]==k for r in rows)) for k in LABELS},
            "failure_rates":{k:float(np.mean([r["failure"]==k for r in rows])) for k in LABELS},
            "mu_success_rates":{
                f"{mu:.1f}":mean([
                    float(r["mu_flags"][mu]) for r in rows
                    if r["mu_flags"][mu] is not None
                ]) for mu in MUS
            },
            "mean_width_m":mean([r["width_m"] for r in rows]),
        }
    return {"condition":condition,"by_depth":output}


def _failure_attribution(family_stats):
    positive=[x for x in family_stats if x["regret"]>1e-8]
    transitions=Counter((x["selected_failure"],x["oracle_failure"]) for x in positive)
    return {
        "n_positive_regret":len(positive),
        "positive_regret_fraction":len(positive)/len(family_stats) if family_stats else None,
        "regret_mass_total":float(sum(x["regret"] for x in positive)),
        "regret_mass_by_selected_failure":{
            k:float(sum(x["regret"] for x in positive if x["selected_failure"]==k))
            for k in LABELS
        },
        "positive_regret_selected_failure_counts":{
            k:int(sum(x["selected_failure"]==k for x in positive)) for k in LABELS
        },
        "selected_to_oracle_failures":{
            f"{a} -> {b}":int(n) for (a,b),n in sorted(transitions.items())
        },
        "warning":"Mutually exclusive evaluator reason classes; oracle is best within 4 depths, not a continuous-action oracle.",
    }


def _width_protocol_compare(rows,n_boot,seed,strict=True):
    fixed=_group(rows,FIXED)
    pred=_group(rows,PRED)
    if not fixed or not pred:
        return {"available":False,"reason":"Need BOTH insertion_sweep_fixed_width and insertion_sweep_predicted_width.",
                "fixed_families":len(fixed),"predicted_families":len(pred)}
    if set(fixed)!=set(pred):
        raise ValueError(f"Fixed/predicted width family mismatch: missing_pred={len(set(fixed)-set(pred))}, missing_fixed={len(set(pred)-set(fixed))}")
    per_family=[]
    variant={}
    for key in sorted(fixed):
        a,b=fixed[key],pred[key]
        if len(a)<2 or set(a)!=set(b):
            raise ValueError(f"Different depth bins in width comparison {key}")
        base_widths=[a[d]["width_m"] for d in sorted(a)]
        if max(base_widths)-min(base_widths)>1e-5:
            raise ValueError(f"Fixed width changed across insertion {key}")
        for depth_id in sorted(a):
            _validate_pair(a[depth_id],b[depth_id],key,depth_id)
        af=_family_stats(a,FIXED)
        bp=_family_stats(b,PRED)
        if af["selected_depth_id"]!=bp["selected_depth_id"]:
            raise RuntimeError(f"Same CDF scores chose different depths: {key}")
        d_star=af["selected_depth_id"]
        if (abs(a[d_star]["width_m"]-b[d_star]["width_m"])>1e-5 or
            a[d_star]["candidate"]!=b[d_star]["candidate"] or
            abs(a[d_star]["quality"]-b[d_star]["quality"])>1e-6):
            raise ValueError(f"Native selected physical action / quality differs across width modes {key}")
        m={
            "sample":af["sample"],"scene":af["scene"],"split":af["split"],
            "parent":af["parent"],"selected_depth_id":d_star,
            "regret_fixed":af["regret"],"regret_predicted":bp["regret"],
            "regret_diff_pred_minus_fixed":bp["regret"]-af["regret"],
            "oracle_quality_fixed":af["oracle_quality"],
            "oracle_quality_predicted":bp["oracle_quality"],
            "oracle_quality_diff_pred_minus_fixed":bp["oracle_quality"]-af["oracle_quality"],
            "selected_quality_fixed":af["selected_quality"],
            "selected_quality_predicted":bp["selected_quality"],
            "oracle_depth_changed":float(af["oracle_depth_id"]!=bp["oracle_depth_id"]),
            "depth_pairwise_accuracy_fixed":af["depth_pairwise_accuracy"],
            "depth_pairwise_accuracy_predicted":bp["depth_pairwise_accuracy"],
            "depth_pairwise_accuracy_diff_pred_minus_fixed":(
                bp["depth_pairwise_accuracy"]-af["depth_pairwise_accuracy"]
                if af["depth_pairwise_accuracy"] is not None and bp["depth_pairwise_accuracy"] is not None else None
            ),
            "num_depths_with_width_change":int(sum(abs(a[d]["width_m"]-b[d]["width_m"])>1e-5 for d in a)),
            "max_abs_width_change_m":float(max(abs(a[d]["width_m"]-b[d]["width_m"]) for d in a)),
        }
        if abs(m["regret_diff_pred_minus_fixed"]-m["oracle_quality_diff_pred_minus_fixed"])>1e-5:
            raise RuntimeError("Regret difference does not equal oracle difference despite same selected action")
        per_family.append(m)
        variant[key]=(af,bp)
    return {
        "available":True,"families":len(per_family),
        "mean_fixed_regret":mean([r["regret_fixed"] for r in per_family]),
        "mean_predicted_regret":mean([r["regret_predicted"] for r in per_family]),
        "pred_minus_fixed_regret":scene_cluster_bootstrap(per_family,"regret_diff_pred_minus_fixed",n_boot,seed),
        "pred_minus_fixed_oracle":scene_cluster_bootstrap(per_family,"oracle_quality_diff_pred_minus_fixed",n_boot,seed),
        "pred_minus_fixed_depth_pairwise_accuracy":scene_cluster_bootstrap(per_family,"depth_pairwise_accuracy_diff_pred_minus_fixed",n_boot,seed),
        "fraction_oracle_depth_changed":mean([r["oracle_depth_changed"] for r in per_family]),
        "fraction_width_diff_at_any_depth":mean([float(r["num_depths_with_width_change"]>0) for r in per_family]),
        "per_family":per_family,
        "fixed":_variant_summary([x[0] for x in variant.values()],fixed,n_boot,seed),
        "predicted":_variant_summary([x[1] for x in variant.values()],pred,n_boot,seed),
    }


def _variant_summary(stats,groups,n_boot,seed):
    return {
        "families":len(stats),
        "selected_quality":scene_cluster_bootstrap(stats,"selected_quality",n_boot,seed),
        "oracle_quality":scene_cluster_bootstrap(stats,"oracle_quality",n_boot,seed),
        "regret":scene_cluster_bootstrap(stats,"regret",n_boot,seed),
        "quality_varies_fraction":mean([x["quality_varies"] for x in stats]),
        "hard_negative_better_fraction":mean([x["hard_negative_better"] for x in stats]),
        "depth_pairwise_accuracy":scene_cluster_bootstrap(stats,"depth_pairwise_accuracy",n_boot,seed),
        "depth_score_quality_spearman":scene_cluster_bootstrap(stats,"depth_score_quality_spearman",n_boot,seed),
        "mean_depth_comparable_pairs":mean([x["depth_comparable_pairs"] for x in stats]),
        "families_with_rankable_depth_pairs":sum(x["depth_comparable_pairs"]>0 for x in stats),
        "hard_negative_quality_minus_selected":scene_cluster_bootstrap(stats,"hard_negative_quality_minus_selected",n_boot,seed),
        "mean_selected_score_margin":mean([x["score_margin_to_hard"] for x in stats]),
        "failure_attribution":_failure_attribution(stats),
        "depth_distribution":_depth_summary(groups,stats[0]["condition"]),
        "per_family":stats,
    }


def _rank12(rows,n_boot,seed):
    grouped=defaultdict(list)
    for row in rows:
        if row["condition"]==NATIVE:
            grouped[row["sample"]].append(row)
    pairs=[]
    for sample,group in sorted(grouped.items()):
        if len(group)<2:
            continue
        ordered=sorted(group,key=lambda x:(-x["score"],x["candidate"]))
        a,b=ordered[:2]
        if a["scene"]!=b["scene"] or a["split"]!=b["split"]:
            raise ValueError("Rank-1 and rank-2 not drawn from the same frame")
        record={
            "sample":sample,"scene":a["scene"],"split":a["split"],
            "rank1_quality":a["quality"],"rank2_quality":b["quality"],
            "rank2_minus_rank1":b["quality"]-a["quality"],
            "rank2_better":float(b["quality"]>a["quality"]+1e-8),
            "rank1_better":float(a["quality"]>b["quality"]+1e-8),
            "quality_tie":float(abs(a["quality"]-b["quality"])<=1e-8),
            "score_margin":a["score"]-b["score"],
            "rank1_failure":a["failure"],"rank2_failure":b["failure"],
            "rank1_id":a["candidate"],"rank2_id":b["candidate"],
        }
        for mu in MUS:
            av,bv=a["mu_flags"][mu],b["mu_flags"][mu]
            record[f"mu_{mu:.1f}_rank2_minus_rank1"]=(float(bv)-float(av)
                                                     if av is not None and bv is not None else None)
        pairs.append(record)
    if not pairs:
        return {"available":False}
    return {
        "available":True,"frames":len(pairs),
        "rank2_minus_rank1":scene_cluster_bootstrap(pairs,"rank2_minus_rank1",n_boot,seed),
        "rank2_better_fraction":mean([r["rank2_better"] for r in pairs]),
        "rank1_better_fraction":mean([r["rank1_better"] for r in pairs]),
        "tie_fraction":mean([r["quality_tie"] for r in pairs]),
        "mean_score_margin":mean([r["score_margin"] for r in pairs]),
        "rank1_failure_counts":{k:int(sum(x["rank1_failure"]==k for x in pairs)) for k in LABELS},
        "rank2_failure_counts":{k:int(sum(x["rank2_failure"]==k for x in pairs)) for k in LABELS},
        "mu_rank2_minus_rank1":{
            f"{mu:.1f}":scene_cluster_bootstrap(pairs,f"mu_{mu:.1f}_rank2_minus_rank1",n_boot,seed)
            for mu in MUS
        },
        "per_frame":pairs,
        "note":"Rank 1/2 come from same query-winner subset; results depend on its fixed query sampling.",
    }


def _width_baselines(rows,n_boot,seed,width_targets=(0.08,0.10)):
    groups=defaultdict(list)
    for r in rows:
        if r["condition"]==WIDTH:
            if not r["parent"]:
                raise ValueError("Missing parent_id for width_sweep")
            groups[(r["sample"],r["parent"])].append(r)
    if not groups:
        return {"available":False,"reason":"No width_sweep rows"}
    summary=[]
    missing=Counter()
    for (sample,parent),group in sorted(groups.items()):
        if len(group)<2: continue
        selected=[r for r in group if r["raw"].get("width_mode")=="predicted"]
        if len(selected)!=1:
            missing["predicted_width_reference"]+=1
            continue
        pred=selected[0]
        if any(np.max(np.abs(np.asarray(r["center"])-np.asarray(pred["center"])))>1e-5 for r in group):
            raise ValueError("Width sweep changed center")
        if any(abs(r["insertion"]-pred["insertion"])>1e-5 for r in group):
            raise ValueError("Width sweep changed insertion")
        if any(np.max(np.abs(_rot(r)-_rot(pred)))>1e-5 for r in group):
            raise ValueError("Width sweep changed rotation")
        if any(abs(r["score"]-pred["score"])>1e-5 for r in group):
            raise ValueError("Width sweep scores changed despite width-invariant reader")
        oracle=max(r["quality"] for r in group)
        rec={
            "sample":sample,"scene":pred["scene"],"split":pred["split"],"parent":parent,
            "pred_width_m":pred["width_m"],"pred_quality":pred["quality"],
            "oracle_quality":oracle,"pred_regret":oracle-pred["quality"],
        }
        valid=True
        for target in width_targets:
            candidates=[r for r in group if abs(r["width_m"]-target)<=1e-5]
            key=f"width_{target:.2f}"
            if len(candidates)!=1:
                missing[key]+=1
                valid=False
                break
            rec[f"{key}_quality"]=candidates[0]["quality"]
            rec[f"{key}_regret"]=oracle-candidates[0]["quality"]
            rec[f"pred_minus_{key}_quality"]=pred["quality"]-candidates[0]["quality"]
        if valid:
            summary.append(rec)
    if not summary:
        return {"available":False,"reason":"No complete width families with predicted and all constant baselines.",
                "missing":dict(missing)}
    return {
        "available":True,"families":len(summary),"missing":dict(missing),
        "pred_regret":scene_cluster_bootstrap(summary,"pred_regret",n_boot,seed),
        "oracle_quality":scene_cluster_bootstrap(summary,"oracle_quality",n_boot,seed),
        "constant_baselines":{
            f"{w:.2f}":{
                "regret":scene_cluster_bootstrap(summary,f"width_{w:.2f}_regret",n_boot,seed),
                "pred_minus_constant_quality":scene_cluster_bootstrap(summary,f"pred_minus_width_{w:.2f}_quality",n_boot,seed),
            } for w in width_targets
        },
        "per_family":summary,
    }


def analyze(rows,n_boot=5000,seed=20261107):
    if not rows:
        raise ValueError("No candidate records")
    for r in rows:
        if not math.isfinite(r["quality"]):
            raise ValueError("P3 requires exact-action quality, not --diag_skip_eval")
        if r["condition"] in (FIXED,PRED):
            semantics=str(r["raw"].get("score_semantics","")).lower()
            if "not_depth_specific" in semantics or "native_frozen" in semantics:
                raise ValueError("Insertion action has a frozen native score, not a matching depth-specific CDF score")
    paired=_width_protocol_compare(rows,n_boot,seed)
    fixed=_group(rows,FIXED)
    predicted=_group(rows,PRED)
    if paired["available"]:
        extra_fixed=None
        extra_pred=None
    else:
        # Existing P2 fixed-width exports are still sufficient for
        # failure/depth/rank12/width-baseline diagnostics.
        extra_fixed=_variant_summary(
            [_family_stats(f,FIXED) for f in fixed.values()],fixed,n_boot,seed
        ) if fixed else None
        extra_pred=_variant_summary(
            [_family_stats(f,PRED) for f in predicted.values()],predicted,n_boot,seed
        ) if predicted else None
    by_split={}
    for split in sorted({r["split"] for r in rows}):
        subset=[r for r in rows if r["split"]==split]
        pair_subset=_width_protocol_compare(subset,n_boot,seed)
        rank_subset=_rank12(subset,n_boot,seed)
        width_subset=_width_baselines(subset,n_boot,seed)
        fixed_subset=_group(subset,FIXED)
        pred_subset=_group(subset,PRED)
        by_split[split]={
            "frames":len({r["sample"] for r in subset}),
            "scenes":len({r["scene"] for r in subset if r["scene"] is not None}),
            "paired_insertion_available":pair_subset["available"],
            "paired_insertion_families":pair_subset.get("families",0),
            "fixed_mean_regret":pair_subset.get("mean_fixed_regret",
                mean([_family_stats(g,FIXED)["regret"] for g in fixed_subset.values()]) if fixed_subset else None),
            "predicted_mean_regret":pair_subset.get("mean_predicted_regret",
                mean([_family_stats(g,PRED)["regret"] for g in pred_subset.values()]) if pred_subset else None),
            "pred_minus_fixed_regret":pair_subset.get("pred_minus_fixed_regret"),
            "rank2_minus_rank1":rank_subset.get("rank2_minus_rank1"),
            "width_constant_baselines":width_subset.get("constant_baselines"),
        }
    return {
        "num_rows":len(rows),
        "num_frames":len({r["sample"] for r in rows}),
        "num_scenes":len({(r["split"],r["scene"]) for r in rows if r["scene"] is not None}),
        "by_split":by_split,
        "fixed_vs_predicted_width":paired,
        "fixed_only_when_unpaired":extra_fixed,
        "predicted_only_when_unpaired":extra_pred,
        "rank1_vs_rank2":_rank12(rows,n_boot,seed),
        "width_predicted_vs_constants":_width_baselines(rows,n_boot,seed),
        "limitations":[
            "Exact CAD/DexNet candidate quality is not official AP or physical robot success.",
            "The chosen depth's width must be equal across fixed/predicted modes; if this contract fails, paired regret is invalid.",
            "Compare fixed and predicted width on the identical query/angle/depth bins and checkpoint; no re-ranking or different proposal pool.",
            "Four-bin insertion oracle is not the continuous-action oracle.",
            "Depth hard negative is the highest-scoring alternative depth at the same center and angle, not a random nonwinner over all angles.",
            "Rank 1/2 statistics reflect the sampled query-winner set, not global candidate ordering.",
            "Scene bootstrap does not capture training-seed or dataset-level generalization uncertainty.",
        ],
    }


def _clean(x):
    if isinstance(x,dict): return {str(k):_clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_clean(v) for v in x]
    if isinstance(x,(np.floating,float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.ndarray):return _clean(x.tolist())
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate_csv",required=True,action="append",
                   help="Repeat this flag to aggregate disjoint Seen/Similar/Novel files.")
    p.add_argument("--output_json",required=True)
    p.add_argument("--n_boot",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261107)
    p.add_argument("--require_both_width_modes",action="store_true",
                   help="Fail rather than report incomplete paired insertion diagnostics.")
    p.add_argument("--expected_frames",type=int,default=0)
    p.add_argument("--expected_scenes",type=int,default=0)
    p.add_argument("--expected_families",type=int,default=0)
    args=p.parse_args()
    rows=[]
    schemas={}
    seen=set()
    for file in args.candidate_csv:
        subset,schema=load_candidates(file,None,"m")
        for item in subset:
            key=(item["sample"],item["condition"],item["candidate"])
            if key in seen:
                raise ValueError(f"Duplicate physical action across input CSV files: {key}")
            seen.add(key)
        rows.extend(subset)
        schemas[file]=schema
    result=analyze(rows,n_boot=args.n_boot,seed=args.seed)
    if args.require_both_width_modes and not result["fixed_vs_predicted_width"]["available"]:
        raise RuntimeError("Required paired fixed/predicted insertion depth protocols are missing")
    for name,expected,observed in (
        ("frames",args.expected_frames,result["num_frames"]),
        ("scenes",args.expected_scenes,result["num_scenes"]),
        ("paired families",args.expected_families,result["fixed_vs_predicted_width"].get("families",0)),
    ):
        if expected and observed!=expected:
            raise RuntimeError(f"Expected {expected} {name}, found {observed}")
    result["input_csv"]=args.candidate_csv
    result["schema_by_input"]=schemas
    dest=Path(args.output_json)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+".tmp")
    tmp.write_text(json.dumps(_clean(result),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(dest)
    paired=result["fixed_vs_predicted_width"]
    print(json.dumps({
        "output":str(dest),
        "frames":result["num_frames"],
        "scenes":result["num_scenes"],
        "matched_insertion_available":paired["available"],
        "matched_families":paired.get("families",0),
        "rank1_vs_rank2":result["rank1_vs_rank2"].get("rank2_minus_rank1",{}).get("mean"),
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
