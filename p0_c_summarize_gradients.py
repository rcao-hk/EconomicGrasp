#!/usr/bin/env python3
"""Summarize module-wise PKD gradient alignment JSONL."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from pkd_p0.common import atomic_json_dump


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="gradient_audit.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--early_audits", type=int, default=5)
    return p.parse_args()


def load(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No audit records in {path}")
    return rows


def aggregate(records: Sequence[Mapping[str, Any]], phase: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        for module, values in record.get("gradient_relations", {}).items():
            groups.setdefault(module, []).append(values)
    output=[]
    for module, values in sorted(groups.items()):
        cos=np.asarray([float(x["cosine"]) for x in values if math.isfinite(float(x["cosine"]))])
        ratio=np.asarray([float(x["kd_to_sup"]) for x in values if math.isfinite(float(x["kd_to_sup"]))])
        sup=np.asarray([float(x["sup_norm"]) for x in values])
        kd=np.asarray([float(x["kd_norm"]) for x in values])
        output.append({
            "phase": phase,
            "module": module,
            "num_audits": len(values),
            "cosine_mean": float(cos.mean()) if len(cos) else float("nan"),
            "cosine_median": float(np.median(cos)) if len(cos) else float("nan"),
            "negative_cosine_ratio": float((cos < 0).mean()) if len(cos) else float("nan"),
            "strong_conflict_ratio_cos_lt_minus_0p2": float((cos < -0.2).mean()) if len(cos) else float("nan"),
            "kd_to_sup_mean": float(ratio.mean()) if len(ratio) else float("nan"),
            "kd_to_sup_median": float(np.median(ratio)) if len(ratio) else float("nan"),
            "sup_norm_mean": float(sup.mean()),
            "kd_norm_mean": float(kd.mean()),
        })
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields=sorted({k for row in rows for k in row})
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    args=parse_args(); records=load(Path(args.input))
    n=min(max(1,int(args.early_audits)),len(records))
    rows=[]
    rows.extend(aggregate(records,"all"))
    rows.extend(aggregate(records[:n],"early"))
    if len(records)>n:
        rows.extend(aggregate(records[-n:],"late"))
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"gradient_summary.csv",rows)
    payload={"num_audits":len(records),"early_audits":n,"summary":rows}
    atomic_json_dump(payload,out/"gradient_summary.json")
    print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=="__main__": main()
