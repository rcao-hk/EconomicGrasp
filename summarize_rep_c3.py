#!/usr/bin/env python3
"""Summarize Rep-C3 exact-action results together with GT stress diagnostics."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from rep_a_common import atomic_file, save_json


def weighted(rows,key):
    use=[r for r in rows if r.get(key,None) not in (None,"")]
    return sum(float(r[key])*int(r["num_queries"]) for r in use)/sum(int(r["num_queries"]) for r in use) if use else None


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    import io
    buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=keys,restval="")
    w.writeheader(); w.writerows(rows)
    with atomic_file(path) as f: f.write(buf.getvalue().encode())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--splits",default="test_seen,test_similar,test_novel")
    args=p.parse_args(); root=Path(args.root)
    splits=set(args.splits.split(","))
    rows=[]
    for inf in sorted((root/"inference").glob("test_*/scene_*/ann_*.npz")):
        split=inf.parts[-3]
        if split not in splits: continue
        ev=root/"evaluation"/inf.relative_to(root/"inference")
        if not ev.is_file(): raise FileNotFoundError(ev)
        with np.load(inf,allow_pickle=False) as z:
            meta={k:float(z[k]) for k in ("gt_mae_mm","gt_rmse_mm","gt_bias_mm","depth_rms_mm","depth_bias_mm") if k in z}
        with np.load(ev,allow_pickle=False) as z:
            rr=json.loads(str(z["rows_json"]))
        for r in rr:
            r.update(meta); rows.append(r)
    if not rows: raise RuntimeError("No Rep-C3 evaluated rows")
    write_csv(root/"c3_per_frame.csv",rows)
    keys=("method","policy","split","mode","case")
    groups=sorted({tuple(r[k] for k in keys) for r in rows})
    cols=(
        "selected_utility","native_utility","utility_gain","success08","native_success08",
        "success08_gain","rescue08","harm08","move_rate","pure_collision","empty",
        "raw_top1_success08","raw_top10_success08","raw_top50_success08",
        "depth_rms_mm","depth_bias_mm","gt_mae_mm","gt_rmse_mm","gt_bias_mm",
    )
    table=[]
    for g in groups:
        rr=[r for r in rows if tuple(r[k] for k in keys)==g]
        table.append({**dict(zip(keys,g)),**{c:weighted(rr,c) for c in cols},
                      "num_frames":len(rr),"num_queries":sum(int(r["num_queries"]) for r in rr)})
    for x in table:
        base=next(r for r in table if (r["method"],r["policy"],r["split"],r["case"])==
                  (x["method"],x["policy"],x["split"],"nominal"))
        x["utility_drop_from_nominal"]=base["selected_utility"]-x["selected_utility"]
        x["success08_drop_from_nominal"]=base["success08"]-x["success08"]
    write_csv(root/"c3_comparison.csv",table); save_json(root/"c3_comparison.json",table)
    print(f"[REP-C3 SUMMARY] {root/'c3_comparison.csv'}")


if __name__=="__main__":
    main()
