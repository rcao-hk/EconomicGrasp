#!/usr/bin/env python3
"""Summarize Rep-C2-v2 decision diagnostic across splits/cases."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rep_a_common import atomic_file, save_json


METRICS=(
    "verified_gain",
    "a1_fixed0_gain",
    "oracle_accept_gain",
    "verifier_increment",
    "oracle_gap",
    "oracle_gap_recovery",
    "beneficial_retention",
    "harmful_rejection",
    "accept_precision",
    "benefit_harm_auc",
    "pearson_with_delta",
    "accept_rate",
    "proposal_rate",
)


def read_csv(path):
    with open(path,newline="") as f:
        return list(csv.DictReader(f))


def fval(row,key):
    value=row.get(key,"")
    return None if value in ("",None,"None") else float(value)


def macro(rows,key):
    vals=[fval(r,key) for r in rows]
    vals=[x for x in vals if x is not None]
    return sum(vals)/len(vals) if vals else None


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    import io
    buf=io.StringIO(newline="")
    w=csv.DictWriter(buf,fieldnames=keys,restval="")
    w.writeheader();w.writerows(rows)
    with atomic_file(path) as f:f.write(buf.getvalue().encode())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--splits",default="test_seen,test_similar,test_novel")
    args=p.parse_args()
    root=Path(args.root)
    splits=[x for x in args.splits.split(",") if x]
    rows=[]
    for split in splits:
        path=root/"eval"/split/"comparison.csv"
        if not path.is_file():raise FileNotFoundError(path)
        rows.extend(read_csv(path))

    keys=sorted({(r["variant"],r["rule"],r["policy"],r["split"]) for r in rows})
    by_split=[]
    for variant,rule,policy,split in keys:
        rr=[r for r in rows if (r["variant"],r["rule"],r["policy"],r["split"])==(variant,rule,policy,split)]
        by_split.append({
            "variant":variant,"rule":rule,"policy":policy,"split":split,
            "num_cases":len(rr),
            **{f"macro_{k}":macro(rr,k) for k in METRICS},
        })

    heldout=[r for r in rows if r["split"] in ("test_similar","test_novel")]
    heldout_keys=sorted({(r["variant"],r["rule"],r["policy"]) for r in heldout})
    heldout_macro=[]
    for variant,rule,policy in heldout_keys:
        rr=[r for r in heldout if (r["variant"],r["rule"],r["policy"])==(variant,rule,policy)]
        heldout_macro.append({
            "variant":variant,"rule":rule,"policy":policy,
            "num_split_case_cells":len(rr),
            **{f"macro_{k}":macro(rr,k) for k in METRICS},
        })

    write_csv(root/"macro_by_split.csv",by_split)
    write_csv(root/"heldout_macro.csv",heldout_macro)
    save_json(root/"decision_summary.json",{
        "macro_by_split":by_split,
        "heldout_macro":heldout_macro,
        "interpretation_note":"Global is the deployable-style policy. case_specific_privileged assumes synthetic case identity and is diagnostic only.",
    })
    print(f"[C2-v2 DECISION SUMMARY] {root/'heldout_macro.csv'}")


if __name__=="__main__":
    main()
