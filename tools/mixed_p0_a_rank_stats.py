#!/usr/bin/env python3
"""P0-A: scene-paired AP@k bootstrap from *official* GraspNet accuracy arrays.

Input: two checkpoint runs with identical scene/annotation ordering, each having
six arrays [30 scenes, 26 views, 50 ranks, 6 friction thresholds]. No candidate
oracle can be reconstructed from prefix AP. Optional --candidate_csv contains
separately evaluated ranked grasps for a *diagnostic* fixed-candidate oracle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLITS = ("test_seen", "test_similar", "test_novel")
DOMAINS = ("graspnet", "gntrans")
MU = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
START_SCENE = {"test_seen": 100, "test_similar": 130, "test_novel": 160}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, action="append", metavar="NAME=DIR",
                   help="Exactly two runs, e.g. --run point=/.../rayrope_point --run gvar=/.../rayrope_point_gvar")
    p.add_argument("--accuracy-template",
                   default="{run}/{domain}/test_collision_off/official/{split}/accuracy.npy",
                   help="Format fields: run (directory), domain, split.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ranks", default="1,5,10,50")
    p.add_argument("--verify-dump-identities", action="store_true",
                   help="Assert all 780 scene/frame predictions per split under each run/domain.")
    p.add_argument("--dump-root-template", default="{run}/{domain}/test_collision_off",
                   help="Directory to scan when --verify-dump-identities is enabled.")
    p.add_argument("--candidate-csv", default=None,
                   help="Optional physical friction CSV from mixed_p0_a_candidate_eval.py. "
                        "Required columns run,domain,split,scene,frame,rank,friction.")
    return p.parse_args()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def load_official(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Official accuracy array unavailable: {path}")
    a = np.load(path, allow_pickle=False)
    if a.shape != (30, 26, 50, 6):
        raise ValueError(f"Expected [30,26,50,6], got {a.shape} at {path}")
    if not np.isfinite(a).all() or np.min(a) < 0 or np.max(a) > 1:
        raise ValueError(f"Invalid accuracy values: {path}")
    return a.astype(np.float64, copy=False)


def ap_at_k(a, k):
    """Official AP@k-style mean of precision@1,...,precision@k and all mu."""
    return np.mean(a[..., :k, :], axis=(-1, -2))


def precision_at_k(a, k):
    """Precision at *exactly* rank k; this is not AP@k."""
    return np.mean(a[..., k - 1, :], axis=-1)


def mean_metric(a, k=50):
    return float(np.mean(ap_at_k(a, k)) * 100)


def percentile_summary(draws, obs):
    draws = np.asarray(draws, np.float64)
    ci = np.percentile(draws, [2.5, 97.5])
    return dict(delta_pp=float(obs), ci95_pp=[float(ci[0]), float(ci[1])],
                bootstrap_prob_positive=float(np.mean(draws > 0)),
                bootstrap_prob_negative=float(np.mean(draws < 0)),
                bootstrap_replicates=int(len(draws)))


def verify_dump_identifiers(run, domain, split, template):
    """Check full [30 scenes × 26 stride-10 frames] identity and uniqueness."""
    root=Path(template.format(run=run,domain=domain,split=split))
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dump root: {root}")
    keys=[]
    for path in root.rglob("scene_*/realsense/*.npy"):
        scene=int(path.parent.parent.name.split("_")[-1])
        frame=int(path.stem)
        if START_SCENE[split]<=scene<START_SCENE[split]+30:
            keys.append((scene,frame))
    expected={(scene,frame)
              for scene in range(START_SCENE[split],START_SCENE[split]+30)
              for frame in range(0,256,10)}
    if len(keys)!=len(set(keys)):
        raise ValueError(f"Duplicate prediction dump identifiers in {root}/{split}")
    missing=sorted(expected-set(keys))
    extra=sorted(set(keys)-expected)
    if missing or extra or len(keys)!=780:
        raise ValueError(
            f"{run}/{domain}/{split}: expected 780 exactly; have {len(keys)}, "
            f"missing={missing[:10]}, extraneous={extra[:10]}"
        )
    fingerprint=hashlib.sha256(
        "".join(f"{sid:04d}_{fid:04d}\n" for sid,fid in sorted(keys)).encode()
    ).hexdigest()
    return dict(count=len(keys),identities_sha256=fingerprint,root=str(root))


def load_candidate_oracle(path, run_names, ranks):
    """Diagnostic oracle on a *fixed evaluated candidate set*, not official AP.

    Scores are the evaluator's minimum successful friction threshold;
    -1 denotes failed/colliding grasp. Sorting by this label is privileged.
    Predictions missing among the top K are counted as failures.
    """
    rows = list(csv.DictReader(Path(path).open("r", newline="", encoding="utf-8")))
    required = {"run", "domain", "split", "scene", "frame", "rank", "friction"}
    if not rows or not required.issubset(set(rows[0])):
        raise ValueError(f"Candidate CSV must contain {sorted(required)} and ≥1 row")
    groups = defaultdict(list)
    for r in rows:
        if r["run"] not in run_names or r["domain"] not in DOMAINS or r["split"] not in SPLITS:
            raise ValueError(f"Unexpected candidate key: {r}")
        mu = float(r["friction"])
        if not np.isfinite(mu):
            raise ValueError(f"Invalid friction: {r}")
        groups[(r["run"], r["domain"], r["split"], r["scene"], r["frame"])].append(
            (int(r["rank"]), mu)
        )
    out = []
    for key, values in sorted(groups.items()):
        order = [a for a, _ in sorted(values)]
        if len(set(order)) != len(order) or min(order) != 1:
            raise ValueError(f"Missing or duplicate first candidate rank for {key}")
        original = np.asarray([v for _, v in sorted(values)], np.float64)
        oracle = np.sort(np.where(original > 0, original, np.inf))
        for k in ranks:
            for name, fr in (("original", original), ("privileged_oracle", oracle)):
                # Do not inflate precision on frames with <K predicted candidates.
                success = np.zeros((k, len(MU)), dtype=np.float64)
                n = min(k, len(fr))
                if n:
                    success[:n, :] = ((fr[:n, None] > 0) & (fr[:n, None] <= np.asarray(MU))).astype(float)
                precisions = np.cumsum(success, axis=0) / np.arange(1, k + 1)[:, None]
                out.append(dict(run=key[0], domain=key[1], split=key[2], scene=key[3],
                                frame=key[4], k=k, method=name, diagnostic_ap_percent=
                                float(100.0 * precisions.mean())))
    return out


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be ≥100")
    ranks = tuple(int(x) for x in args.ranks.split(","))
    if any(k not in range(1, 51) for k in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("--ranks must contain unique integers within [1,50]")
    runs = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError("--run needs NAME=DIR")
        name, root = spec.split("=", 1)
        if not name or name in runs:
            raise ValueError(f"Invalid or repeated run name: {name}")
        runs[name] = str(Path(root).expanduser().resolve())
    if len(runs) != 2:
        raise ValueError("Paired bootstrap requires exactly two runs")
    names = list(runs)
    arr = {}
    sources = {}
    for name, root in runs.items():
        for domain in DOMAINS:
            for split in SPLITS:
                file = Path(args.accuracy_template.format(run=root, domain=domain, split=split))
                arr[(name, domain, split)] = load_official(file)
                sources[f"{name}/{domain}/{split}"] = dict(path=str(file), sha256=file_hash(file))

    dump_manifest={}
    if args.verify_dump_identities:
        for name,root in runs.items():
            for domain in DOMAINS:
                for split in SPLITS:
                    dump_manifest[f"{name}/{domain}/{split}"]=verify_dump_identifiers(
                        root,domain,split,args.dump_root_template
                    )
        # Same domain/split has identical scene+frame identities for Point/GVAR.
        for domain in DOMAINS:
            for split in SPLITS:
                if dump_manifest[f"{names[0]}/{domain}/{split}"]["identities_sha256"] != (
                    dump_manifest[f"{names[1]}/{domain}/{split}"]["identities_sha256"]
                ):
                    raise ValueError(f"Paired dump identity mismatch in {domain}/{split}")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, scene_rows = [], []
    for domain in DOMAINS:
        for split in SPLITS:
            a, b = (arr[(name, domain, split)] for name in names)
            for local_idx in range(30):
                scene_rows.append(dict(domain=domain, split=split,
                    scene=f"scene_{START_SCENE[split] + local_idx:04d}",
                    **{f"ap50_{name}":mean_metric(v[local_idx]) for name, v in [(names[0], a), (names[1], b)]},
                    delta_pp=mean_metric(b[local_idx])-mean_metric(a[local_idx])))
            for k in ranks:
                rows.append(dict(domain=domain, split=split, rank_k=k,
                                 metric="AP@k_mean_prefix", run_a=names[0], run_b=names[1],
                                 run_a_percent=mean_metric(a, k), run_b_percent=mean_metric(b, k),
                                 delta_pp=mean_metric(b, k)-mean_metric(a, k),
                                 precision_at_k_a_percent=float(100 * precision_at_k(a, k).mean()),
                                 precision_at_k_b_percent=float(100 * precision_at_k(b, k).mean())))

    rng = np.random.default_rng(args.seed)
    # Precompute per-scene AP@k once; bootstrapping the full
    # [30,26,50,6] arrays thousands of times would be prohibitively slow.
    scene_delta = {
        (domain, split, k): (
            ap_at_k(arr[(names[1], domain, split)], k).mean(axis=1)
            - ap_at_k(arr[(names[0], domain, split)], k).mean(axis=1)
        ) * 100.0
        for domain in DOMAINS for split in SPLITS for k in ranks
    }
    # Paired/stratified: identical sampled scene IDs for GN/GNTrans and both
    # models within a split; splits are independently resampled.
    index = {
        split: rng.integers(0, 30, size=(args.bootstrap, 30))
        for split in SPLITS
    }
    draws = {}
    for domain in (*DOMAINS, "macro"):
        for k in ranks:
            components = [
                scene_delta[(d, split, k)][index[split]].mean(axis=1)
                for split in SPLITS
                for d in (DOMAINS if domain == "macro" else (domain,))
            ]
            draws[(domain, k)] = np.mean(np.stack(components, axis=0), axis=0)

    bootstrap = []
    for domain in (*DOMAINS, "macro"):
        for k in ranks:
            relevant = [r for r in rows if r["rank_k"] == k and (domain == "macro" or r["domain"] == domain)]
            obs = float(np.mean([r["delta_pp"] for r in relevant]))
            bootstrap.append(dict(domain=domain, rank_k=k, **percentile_summary(draws[(domain, k)], obs)))

    write_csv(outdir / "official_ap_at_k.csv", rows)
    write_csv(outdir / "scene_pair_deltas.csv", scene_rows)
    write_csv(outdir / "scene_paired_bootstrap.csv", bootstrap)
    oracle_rows = []
    if args.candidate_csv:
        oracle_rows = load_candidate_oracle(args.candidate_csv, set(runs), ranks)
        write_csv(outdir / "candidate_oracle_diagnostic.csv", oracle_rows)
    summary = dict(source_sha256=sources, verified_dump_identities=dump_manifest,
        runs=runs, ranks=ranks, bootstrap_seed=args.seed,
        bootstrap_replicates=args.bootstrap, unit="percentage points",
        official_metric="mean prefix precision ranks 1..K across all six mu",
        scene_resampling="stratified per split; paired across runs and RGB domains",
        candidate_oracle_available=bool(oracle_rows),
        candidate_oracle_note="privileged fixed-list diagnostic, never official AP",
        results=bootstrap)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status="complete", summary=str(outdir / "summary.json"),
                          candidate_oracle=bool(oracle_rows), six_split_AP_delta_pp=
                          next(x["delta_pp"] for x in bootstrap if x["domain"] == "macro" and x["rank_k"] == 50)), indent=2))


if __name__ == "__main__":
    main()
