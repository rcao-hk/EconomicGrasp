#!/usr/bin/env python3
"""Summarize Rep-P0 source-wise held-out results into one comparison table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_root",required=True)
    p.add_argument("--output_dir",required=True)
    p.add_argument("--sources",default="pred,sensor,rendered,cad_full")
    p.add_argument("--splits",default="test_similar,test_novel")
    return p.parse_args()


def main():
    args=parse_args()
    root=Path(args.test_root)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    sources=[x.strip() for x in args.sources.split(",") if x.strip()]
    splits=[x.strip() for x in args.splits.split(",") if x.strip()]
    rows=[]
    for source in sources:
        for split in splits:
            p=root/source/split/"summary.json"
            if not p.is_file():
                continue
            d=json.loads(p.read_text())
            c=d["candidate_metrics"]; m=d["policy"]
            rows.append({
                "source":source,
                "split":split,
                "candidate_spearman":c["pred_exact_utility_spearman"],
                "candidate_success08_auroc":c["success08_auroc"],
                "candidate_success08_auprc":c["success08_auprc"],
                "native_utility":m["native_utility"],
                "selected_utility":m["selected_utility"],
                "utility_gain":m["utility_gain"],
                "oracle_utility":m["oracle_utility"],
                "headroom_recovery":m["utility_headroom_recovery"],
                "native_success08":m["native_success08"],
                "selected_success08":m["selected_success08"],
                "success08_gain":m["success08_gain"],
                "rescue08":m["rescue08"],
                "harm08":m["harm08"],
                "change_rate":m["change_rate"],
                "selection_margin":d["selection_margin"],
            })
    if not rows:
        raise SystemExit("No Rep-P0 summary.json files found.")
    with (out/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out/"comparison.json").open("w") as f:
        json.dump(rows,f,indent=2,sort_keys=True)
    print("[REP-P0] comparison")
    for r in rows:
        print(
            f"  {r['source']:10s} {r['split']:12s} "
            f"rho={r['candidate_spearman']:+.3f} "
            f"AUROC08={r['candidate_success08_auroc']:.3f} "
            f"dU={r['utility_gain']:+.5f} "
            f"dS08={100*r['success08_gain']:+.2f}pp "
            f"harm={100*r['harm08']:.2f}% "
            f"headroom={100*r['headroom_recovery']:.2f}%"
        )


if __name__=="__main__":
    main()
