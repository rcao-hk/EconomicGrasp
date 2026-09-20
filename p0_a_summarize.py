#!/usr/bin/env python3
"""Summarize four P0-A evaluator logs into the causal matrix."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from pkd_p0.common import atomic_json_dump


CELL_NAMES = ("S_P", "S_G", "T_P", "T_G")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--output_dir", default="")
    return p.parse_args()


def extract_ap(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"AP\s+(?:Seen|Similar|Novel)\s*=\s*([0-9.eE+-]+)",
        r"AP\s*=\s*([0-9.eE+-]+)",
        r"(?:mean\s+)?AP[^0-9]*([0-9]+\.[0-9]+)\s*$",
    )
    matches: List[float] = []
    for pattern in patterns:
        matches.extend(float(value) for value in re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            break
    if not matches:
        raise RuntimeError(f"No final AP found in {path}")
    return matches[-1]


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    aps: Dict[str, float] = {}
    logs: Dict[str, str] = {}
    for cell in CELL_NAMES:
        candidates = [
            root / cell / args.split / "eval.log",
            root / cell / args.split / "evaluation.log",
            root / "_eval_logs" / f"{cell}_{args.split}.log",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No evaluator log for {cell}/{args.split}; checked {candidates}")
        aps[cell] = extract_ap(path)
        logs[cell] = str(path)

    deltas = {
        "student_geometry_gain": aps["S_G"] - aps["S_P"],
        "teacher_geometry_gain": aps["T_G"] - aps["T_P"],
        "weight_gain_under_predicted_geometry": aps["T_P"] - aps["S_P"],
        "weight_gain_under_gt_geometry": aps["T_G"] - aps["S_G"],
        "teacher_total_gap": aps["T_G"] - aps["S_P"],
    }
    rows = [{"split": args.split, "cell": cell, "ap": aps[cell]} for cell in CELL_NAMES]
    with open(output / f"p0_a_matrix_{args.split}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "cell", "ap"))
        writer.writeheader(); writer.writerows(rows)
    payload = {"split": args.split, "ap": aps, "deltas": deltas, "logs": logs}
    atomic_json_dump(payload, output / f"p0_a_matrix_{args.split}.json")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
