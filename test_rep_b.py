#!/usr/bin/env python3
"""Rep-B fixed-action depth-robustness stress test.

Physical actions and exact labels never change across depth perturbations.
For B0, depth is not a model input and predictions should be numerically
invariant. B1/B2 use depth only through the soft prior.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from rep_a_common import (
    DEFAULT_CASES,
    aggregate,
    array_sha,
    check_runtime_sources,
    digest,
    exclusive_run,
    file_sha,
    list_frames,
    load_torch,
    metrics,
    parse_case,
    perturb_depth,
    read_frame,
    save_json,
    seed_for,
    atomic_file,
)
from rep_b_common import rep_b_tensors
from rep_b_model import RepBModel


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=("test_similar", "test_novel"), required=True)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress-every", type=int, default=50)
    return p


def _masked_mean(x, valid):
    return float(x[valid].float().mean())


def _component_drift(current, nominal, valid):
    out = {}
    for key in ("image", "action", "offset", "prior", "pre_relation"):
        x = current.get(key)
        y = nominal.get(key)
        if x is None or y is None:
            continue
        # Components are [K,Q,D].
        cos = 1 - F.cosine_similarity(x.float(), y.float(), dim=-1)
        rel = (x-y).float().norm(dim=-1) / y.float().norm(dim=-1).clamp_min(1e-6)
        out[f"{key}_cosine_drift"] = _masked_mean(cos, valid)
        out[f"{key}_relative_l2_drift"] = _masked_mean(rel, valid)
        out[f"{key}_norm"] = _masked_mean(x.float().norm(dim=-1), valid)
    return out


@torch.no_grad()
def main():
    args = parser().parse_args()
    sys.argv = [sys.argv[0]]
    cases = list(dict.fromkeys(
        ["nominal"] + [x.strip() for x in args.cases.split(",") if x.strip()]
    ))
    for case in cases:
        parse_case(case)

    check_runtime_sources(args.cache_root)
    device = torch.device(args.device)
    paths = list_frames(args.cache_root, args.split, args.max_frames)

    ck = load_torch(args.checkpoint)
    if ck.get("experiment") != "Rep-B":
        raise RuntimeError(f"Not a Rep-B checkpoint: {args.checkpoint}")
    model = RepBModel(**ck["model_spec"]).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    variant = ck["variant"]
    margin = float(ck["margin"])
    epoch = int(ck["epoch"])
    contract = ck["contract"]
    model_spec = ck["model_spec"]

    signature = digest({
        "checkpoint_sha256": file_sha(args.checkpoint),
        "split": args.split,
        "cases": cases,
        "seed": args.seed,
        "model_spec": model_spec,
        "frames": [
            (p.parent.name, p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in paths
        ],
    })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with exclusive_run(out / ".test.lock"):
        if args.resume and (out / "summary.json").exists():
            saved = json.loads((out / "summary.json").read_text())
            if saved["signature"] != signature:
                raise RuntimeError("Rep-B test signature changed; use another output-dir")
            print(f"[REP-B TEST] already complete: {out}", flush=True)
            return

        rows = []
        for i, path in enumerate(paths):
            d = read_frame(path, contract)
            t = rep_b_tensors(d, device)
            before_sha = array_sha(
                d["actions"], d["valid"], d["friction"], d["query_ids"]
            )
            valid = t["valid"]

            logits0, rep0, comp0 = model(
                t, return_repr=True, return_components=True
            )
            p0 = logits0.sigmoid().cpu().numpy()
            _, sel0 = metrics(p0, d, margin)

            for case in cases:
                seed = seed_for(
                    args.seed, "rep_b_test", args.split,
                    int(d["scene_id"]), int(d["anno_id"]), parse_case(case)[0]
                )
                depth, err = perturb_depth(t["depth"], case, seed)
                if case == "nominal":
                    logits, rep, comp = logits0, rep0, comp0
                else:
                    logits, rep, comp = model(
                        t, depth=depth, return_repr=True, return_components=True
                    )

                prob = logits.sigmoid().cpu().numpy()
                met, selected = metrics(prob, d, margin)
                rep_cos = 1 - F.cosine_similarity(
                    rep.float(), rep0.float(), dim=-1
                )
                rep_rel = (
                    (rep-rep0).float().norm(dim=-1)
                    / rep0.float().norm(dim=-1).clamp_min(1e-6)
                )

                ids = np.argsort(
                    -d["native_score"], kind="stable"
                )[:max(1, len(d["native_score"]) // 2)]
                small = {
                    key: d[key][:, ids]
                    for key in ("valid", "friction", "utility")
                }
                small["zero_index"] = d["zero_index"]
                top, _ = metrics(prob[:, ids], small, margin)

                row = {
                    "variant": variant,
                    "split": args.split,
                    "scene_id": int(d["scene_id"]),
                    "anno_id": int(d["anno_id"]),
                    "action_sha": before_sha,
                    "case": case,
                    "margin": margin,
                    **met,
                    **err,
                    "probability_drift": float(np.abs(prob-p0)[d["valid"]].mean()),
                    "representation_cosine_distance": _masked_mean(rep_cos, valid),
                    "representation_relative_l2": _masked_mean(rep_rel, valid),
                    "selection_turnover": float((selected != sel0).mean()),
                    "top_half_utility_gain": top["utility_gain"],
                    "top_half_success08_gain": top["success08_gain"],
                    **_component_drift(comp, comp0, valid),
                }
                rows.append(row)

            if before_sha != array_sha(
                d["actions"], d["valid"], d["friction"], d["query_ids"]
            ):
                raise RuntimeError("Rep-B mutated fixed actions/labels")
            del d, t, logits0, rep0, comp0, p0, logits, rep, comp, prob, depth

            if (i+1) % max(args.progress_every, 1) == 0:
                print(
                    f"[REP-B TEST {variant}] {i+1}/{len(paths)} "
                    f"frames x {len(cases)} cases",
                    flush=True,
                )

        with atomic_file(out / "per_frame.csv") as f:
            import io
            text = io.StringIO()
            fieldnames = sorted({k for row in rows for k in row})
            w = csv.DictWriter(text, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
            f.write(text.getvalue().encode())

        case_results = {}
        base_extra = (
            "probability_drift",
            "representation_cosine_distance",
            "representation_relative_l2",
            "selection_turnover",
            "depth_rms_mm",
            "depth_bias_mm",
            "clamped_fraction",
            "top_half_utility_gain",
            "top_half_success08_gain",
        )
        component_keys = sorted({
            k for row in rows for k in row
            if k.endswith("_cosine_drift")
            or k.endswith("_relative_l2_drift")
            or k.endswith("_norm")
        })
        for case in cases:
            rr = [x for x in rows if x["case"] == case]
            case_results[case] = aggregate(rr)
            nq = sum(x["num_queries"] for x in rr)
            for key in base_extra + tuple(component_keys):
                vals = [x for x in rr if key in x]
                if vals:
                    case_results[case][key] = (
                        sum(x[key] * x["num_queries"] for x in vals)
                        / sum(x["num_queries"] for x in vals)
                    )

        clean = case_results["nominal"]
        for item in case_results.values():
            item["utility_drop_from_nominal"] = (
                clean["selected_utility"] - item["selected_utility"]
            )
            item["success08_drop_from_nominal"] = (
                clean["success08"] - item["success08"]
            )

        # B0 should be exactly depth independent except floating-point reruns.
        invariant_check = None
        if variant == "B0":
            worst_prob = max(v["probability_drift"] for v in case_results.values())
            worst_rep = max(
                v["representation_relative_l2"] for v in case_results.values()
            )
            invariant_check = {
                "max_probability_drift": worst_prob,
                "max_representation_relative_l2": worst_rep,
                "passed_1e-7": bool(worst_prob < 1e-7 and worst_rep < 1e-7),
            }

        result = {
            "signature": signature,
            "experiment": "Rep-B",
            "variant": variant,
            "split": args.split,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": file_sha(args.checkpoint),
            "epoch": epoch,
            "validation_selected_margin": margin,
            "cache_contract": contract,
            "model_spec": model_spec,
            "seed": args.seed,
            "fixed_actions_unchanged": True,
            "depth_role": (
                "none" if not model.use_prior
                else "soft prior only; RGB evidence is depth-independent"
            ),
            "b0_invariance_check": invariant_check,
            "not_official_AP": True,
            "cases": case_results,
        }
        save_json(out / "summary.json", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
