"""P6-0: offline opportunity, rescue/harm and decision-flip analysis.

Consumes signed P5-1 matched-reader outputs only. No model fitting, changes to
the physical action bank, test inference or access to CAD geometry.
Confidence intervals are scene-clustered descriptions on reused development data.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .data import load_signed, sha256, signed, atomic_json
from .p0_relative_decision import scene_balanced, scene_bootstrap
from .p4_common import new_directory, write_records

SCHEMA = "shared-visible-geometry-p6-opportunity-v1"
MODES = (
    "predicted_p2_fixed",
    "corrected_p2_fixed",
    "clean_p2_fixed",
    "predicted_retrained",
    "corrected_adapted",
)
PAIRS = (
    ("corrected_p2_fixed", "predicted_p2_fixed"),
    ("corrected_adapted", "predicted_retrained"),
    ("corrected_adapted", "corrected_p2_fixed"),
    ("clean_p2_fixed", "predicted_p2_fixed"),
)


def identity(row):
    return (int(row["scene"]), row["domain"], int(row["anno"]), int(row["family"]))


def _read_artifact(root, entry):
    path = Path(entry["path"])
    if path.is_absolute() or path.name != str(path) or ".." in path.parts:
        raise ValueError("P5 family artifact must be a basename in its own directory")
    full = root / path
    if sha256(full) != entry["sha256"]:
        raise ValueError("P5 family artifact SHA256 mismatch: " + str(path))
    with gzip.open(full, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != entry["rows"] or not rows:
        raise ValueError("Missing or incomplete P5 family artifact")
    return rows


def _validate_family(row):
    q = float(row["native_q"])
    if not (0 <= int(row["native_count"]) <= 6 and abs(q-int(row["native_count"])/6.) < 1e-8):
        raise ValueError("Native quality/count mismatch")
    alternatives = row["alternatives"]
    if len(alternatives) != 6:
        raise ValueError("Expected six fixed offsets")
    vals = [q] + [float(x["q"]) for x in alternatives]
    if any(not (0. <= v <= 1.) for v in vals):
        raise ValueError("Invalid exact-action quality")
    if any(abs(x*6-round(x*6))>1e-7 for x in vals):
        raise ValueError("Quality not on count/6 grid")
    if abs(max(vals)-float(row["oracle_q"]))>1e-8 or bool(row["opportunity"])!=(max(vals)>q):
        raise ValueError("Oracle/action opportunity disagreement")
    k=int(row["choice"])
    if not 0<=k<7 or abs(vals[k]-float(row["selected_q"]))>1e-8:
        raise ValueError("Selected action/quality mismatch")
    if abs(float(row["gain"]) - (vals[k]-q))>1e-8:
        raise ValueError("Selected gain mismatch")
    for x in alternatives:
        if abs(float(x["delta_q"])-(float(x["q"])-q))>1e-8:
            raise ValueError("Alternative quality delta mismatch")
    if (bool(row["rescue"]) != (vals[k] > q) or
        bool(row["harm"]) != (vals[k] < q) or
        bool(row["move"]) != (k != 0)):
        raise ValueError("Rescue/harm/move classification mismatch")
    return np.asarray(vals)


def load_matched(summary_path, smoke=False):
    summary_path=Path(summary_path)
    s=load_signed(summary_path)
    if s.get("schema")!="shared-visible-geometry-p5-reader-v1" or s.get("kind")!="matched-reader-comparison":
        raise ValueError("Expected signed P5-1 matched-reader summary")
    if s.get("partition")!="validation" or not s.get("official_ap_not_run", False):
        raise ValueError("P6-0 is development-validation only")
    if not smoke and (not s.get("weights_matched_budget") or
                     not s.get("training_quality_labels_only") or
                     s.get("cross_source_scores_are_OOD") is not True):
        raise ValueError("P5 formal provenance/invariance checks failed")
    modes={}
    for name in MODES:
        if name not in s["artifacts"] or name not in s["scores"]:
            raise ValueError("Missing matched P5 source: "+name)
        rows=_read_artifact(summary_path.parent,s["artifacts"][name])
        bykey={}
        for row in rows:
            key=identity(row)
            if key in bykey:raise ValueError("Duplicate physical family")
            if not 0<=key[0]<100 or row["domain"] not in ("original","gntrans"):
                raise ValueError("P5-1 output contains a test or unknown domain")
            _validate_family(row)
            bykey[key]=row
        if not smoke and len(bykey)!=5120:
            raise ValueError("Formal P5-1 expects exactly 5120 families")
        if abs(scene_balanced(rows,"gain")-s["scores"][name]["gain"])>2e-6:
            raise ValueError("P5 published gain cannot be reproduced: "+name)
        modes[name]=bykey
    keys=sorted(modes[MODES[0]])
    for name in MODES[1:]:
        if set(modes[name])!=set(keys):raise ValueError("Family identity differs between readers")
    for key in keys:
        first=modes[MODES[0]][key]
        for name in MODES[1:]:
            r=modes[name][key]
            for field in ("native_count","native_q","oracle_q","opportunity","physical_ids"):
                if r[field]!=first[field]:
                    raise ValueError("Physical action/quality bank differs at "+str(key))
            if any(abs(a["q"]-b["q"])>1e-8 for a,b in
                   zip(first["alternatives"],r["alternatives"])):
                raise ValueError("Alternative truth changed at "+str(key))
    return s,modes,keys


def subset_metrics(rows):
    """Scene-balanced rates; None means denominator has no eligible families."""
    def mean(name,condition=None):
        chosen=rows if condition is None else [r for r in rows if condition(r)]
        return scene_balanced(chosen,name) if chosen else None
    opportunity=lambda r:r["opportunity"]
    no_opportunity=lambda r:not r["opportunity"]
    notfull=lambda r:r["native_count"]<6
    return dict(
        n_families=len(rows),
        opportunity_fraction=mean("opportunity"),
        oracle_gain=mean("oracle_gain"),
        native_selected_gain=mean("gain"),
        oracle_headroom_recovered_ratio=(
            mean("gain")/mean("oracle_gain") if mean("oracle_gain") and mean("oracle_gain")>0 else None),
        selected_q=mean("selected_q"),
        move_rate=mean("move"),
        rescue_rate=mean("rescue"),
        harm_rate=mean("harm"),
        rescue_given_opportunity=mean("rescue",opportunity),
        harm_given_opportunity=mean("harm",opportunity),
        harm_without_opportunity=mean("harm",no_opportunity),
        unnecessary_move_rate=mean("move",no_opportunity),
        move_given_opportunity=mean("move",opportunity),
        nativepair_correct=mean("nativepair_correct"),
        allpair_correct=mean("allpair_correct"),
        native_nonfull_gain=mean("gain",notfull),
        native_full_count6_fraction=mean("native_full_count6"),
        selected_gain_pp=100*mean("gain") if mean("gain") is not None else None,
    )


def _enrich(r):
    t=dict(r)
    t["oracle_gain"]=float(r["oracle_q"]-r["native_q"])
    t["native_full_count6"]=int(r["native_count"]==6)
    return t


def _paired_ci(left,right,metric,n_boot,seed,predicate=None):
    a=left if predicate is None else [r for r in left if predicate(r)]
    b=right if predicate is None else [r for r in right if predicate(r)]
    if not a:return None
    if [identity(x) for x in a]!=[identity(x) for x in b]:
        raise ValueError("Paired CI family order/mask differs")
    return scene_bootstrap(a,b,metric=metric,n_boot=n_boot,seed=seed)


def analyze(summary_path, output, bootstrap=5000, seed=0, smoke=False):
    n_boot=bootstrap
    if n_boot<1:raise ValueError("Bootstrap budget must be positive")
    s,modes,keys=load_matched(summary_path,smoke=smoke)
    dest=new_directory(output)
    rows={m:[_enrich(modes[m][k]) for k in keys] for m in MODES}
    summaries={m:subset_metrics(rr) for m,rr in rows.items()}
    comparisons={}
    for left,right in PAIRS:
        x,y=rows[left],rows[right]
        comparisons[left+"_minus_"+right]={
            "selected_gain_ci":_paired_ci(x,y,"gain",n_boot,seed),
            "rescue_given_opportunity_ci":_paired_ci(x,y,"rescue",n_boot,seed,lambda r:r["opportunity"]),
            "harm_without_opportunity_ci":_paired_ci(x,y,"harm",n_boot,seed,lambda r:not r["opportunity"]),
            "nativepair_correct_ci":_paired_ci(
                x,y,"nativepair_correct",n_boot,seed,
                lambda r:r["nativepair_correct"] is not None),
        }
    # Per-family actions are aligned by physical IDs; scores never become labels.
    flips=[]
    for key in keys:
        p=modes["predicted_p2_fixed"][key]
        c=modes["corrected_p2_fixed"][key]
        q=modes["clean_p2_fixed"][key]
        rr=dict(scene=key[0],domain=key[1],anno=key[2],family=key[3],
                native_q=p["native_q"],oracle_q=p["oracle_q"],opportunity=p["opportunity"],
                oracle_gain=p["oracle_q"]-p["native_q"],
                predicted_choice=p["choice"],corrected_choice=c["choice"],clean_choice=q["choice"],
                predicted_gain=p["gain"],corrected_gain=c["gain"],clean_gain=q["gain"],
                changed_selection=bool(p["choice"]!=c["choice"]),
                corrected_more_useful=bool(c["gain"]>p["gain"]),
                corrected_more_harmful=bool(c["gain"]<p["gain"]),
                predicted_harm=p["harm"],corrected_harm=c["harm"],
                predicted_rescue=p["rescue"],corrected_rescue=c["rescue"],
                corrected_delta_gain=c["gain"]-p["gain"])
        flips.append(rr)
    decision_flips=dict(
        changed_selection=scene_balanced(flips,"changed_selection"),
        corrected_more_useful=scene_balanced(flips,"corrected_more_useful"),
        corrected_more_harmful=scene_balanced(flips,"corrected_more_harmful"),
        corrected_delta_gain_ci=scene_bootstrap(
            flips,metric="corrected_delta_gain",n_boot=n_boot,seed=seed),
        transitions={
            "harm_to_rescue":sum(x["predicted_harm"] and x["corrected_rescue"] for x in flips),
            "rescue_to_harm":sum(x["predicted_rescue"] and x["corrected_harm"] for x in flips),
            "harm_avoided":sum(x["predicted_harm"] and not x["corrected_harm"] for x in flips),
            "new_harm":sum(not x["predicted_harm"] and x["corrected_harm"] for x in flips),
            "rescue_found":sum(not x["predicted_rescue"] and x["corrected_rescue"] for x in flips),
            "rescue_lost":sum(x["predicted_rescue"] and not x["corrected_rescue"] for x in flips),
        })
    report=dict(schema=SCHEMA,partition="development_validation",
        source_summary_sha256=sha256(summary_path),
        source_validation_fingerprint=s["validation_fingerprint"],
        source_checkpoint_hashes=s["sources"],
        n_families=len(keys),n_scenes=len({x[0] for x in keys}),
        modes=summaries,comparisons=comparisons,decision_flips=decision_flips,
        conditional_metrics_use_same_physical_quality_bank=True,
        official_ap_not_run=True,bootstrap_unit="paired scene, domains equal within scene",
        limits=("All readers were selected on reused scene80-99; CIs are exploratory. "
                "Clean is privileged. This fixed action bank does not measure official AP."))
    report["artifacts"]={"flips":write_records(dest/"decision_flips.jsonl.gz",flips)}
    atomic_json(dest/"summary.json",signed(report))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p5-summary",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    result=analyze(a.p5_summary,a.output,a.bootstrap,a.seed,a.smoke)
    print(json.dumps({"n_families":result["n_families"],
                      "opportunity_fraction":result["modes"]["predicted_p2_fixed"]["opportunity_fraction"],
                      "decision_flips":result["decision_flips"]["changed_selection"]},indent=2))


if __name__=="__main__":
    main()
