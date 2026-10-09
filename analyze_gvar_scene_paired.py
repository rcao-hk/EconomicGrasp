#!/usr/bin/env python3
"""Strict, scene-paired evaluation of independent GVAR checkpoints.

Reads *existing* GVAR Original GraspNet AP tensors and their training/inference
manifests; never trains models, re-runs inference, or calls the evaluator.

AP tensor contract: [30 scenes, 26 frames, 50 prefix ranks, 6 frictions].
GraspNet 10% frames: 0,10,...,250, with canonical scene ids per split.
Units in outputs are AP *percentage points* (inputs use probabilities [0,1]).

Bootstrap samples scenes (not frames) *with replacement* within each split,
using one common bootstrap-index draw for all compared variants and metrics.
The three-split Mean uses a stratified bootstrap and equal split weights.
This CI quantifies scene sampling only, NOT checkpoint/seed variability.

The resulting prefix metrics are cumulative prefix precision, never grasp recall
or an isolated score-ranking effect (models generated different candidate sets).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

SPLITS = ("test_seen", "test_similar", "test_novel")
SCENE_FIRST = {"test_seen": 100, "test_similar": 130, "test_novel": 160}
SPLIT_TITLE = {"test_seen": "Seen", "test_similar": "Similar", "test_novel": "Novel"}
FRICTIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
PREFIXES = (1, 5, 10, 20, 50)
FRAMES = tuple(range(0, 256, 10))
SHAPE = (30, 26, 50, 6)
TOOL_VERSION = "1.0"
METRIC_NAMES = ("AP",) + tuple(f"AP_mu{mu:.1f}" for mu in FRICTIONS) + tuple(f"prefix_precision@{k}" for k in PREFIXES)
OUTPUT_FILES = frozenset(("REPORT.md", "paired_summary.csv", "paired_metrics.csv", "scene_level.csv", "frame_level.csv", "cross_variant_contrasts.csv", "cross_variant_scenes.csv", "input_audit.csv", "gvar_scene_paired_audit.json"))
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class AuditError(ValueError):
    """Raised when paired comparisons violate an input/protocol contract."""


def _require(cond: bool, message: str) -> None:
    if not bool(cond):
        raise AuditError(message)


def _json(path: Path) -> dict:
    _require(path.is_file(), f"Missing required JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"Unreadable JSON at {path}: {exc}") from exc
    _require(isinstance(value, dict), f"Expected JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(split: str) -> str:
    lines = [f"scene_{s:04d}/{f:04d}" for s in range(SCENE_FIRST[split], SCENE_FIRST[split] + 30) for f in FRAMES]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def canonical_scene_ids(split: str) -> list[int]:
    return list(range(SCENE_FIRST[split], SCENE_FIRST[split] + SHAPE[0]))


def _field(data: Mapping, name: str, where: str) -> Any:
    _require(name in data, f"Required field {name!r} absent in {where}")
    return data[name]


def _same_value(a: Any, b: Any, what: str) -> None:
    _require(a == b, f"Incompatible {what}: {a!r} != {b!r}")


def _train_contract(root: Path, variant: str) -> tuple[dict, dict, list[dict]]:
    folder = root / "train" / variant
    gpath = folder / "gvar_protocol.json"
    mpath = folder / "gntrans_mix_protocol.json"
    g, m = _json(gpath), _json(mpath)
    _same_value(_field(g["gvar_config"], "variant", str(gpath)), variant, f"train gvar_config.variant for {variant}")
    for k, expected in (("train_original", 2600), ("train_gntrans", 2600),
                        ("validation_original_seen", 780), ("validation_gntrans", 0),
                        ("frame_stride", 10), ("smoke_max_batches", 0), ("per_rank_batch", 3),
                        ("world_size", 3), ("global_batch", 9)):
        _same_value(_field(g, k, str(gpath)), expected, f"{variant}.gvar_protocol.{k}")
    _same_value(_field(g, "detach_policy", str(gpath)), {"E": True, "Q": True, "C": True}, f"{variant}.detach_policy")
    _same_value(_field(g, "network_geometry", str(gpath)), "predicted metric depth", f"{variant}.network_geometry")
    _same_value(_field(g, "validation_gntrans", str(gpath)), 0, f"{variant}.validation_gntrans")
    aconf = _field(g, "architecture_config", str(gpath))
    _require(isinstance(aconf, dict), f"Bad architecture_config in {gpath}")
    for key, expected in (("pose_depth_mode", "global_film"), ("kview_mode", "A1"), ("use_cdf", True)):
        _same_value(_field(aconf, key, str(gpath)), expected, f"{variant}.architecture_config.{key}")
    for k, expected in (("train_fraction_each_domain", 0.1), ("eval_fraction_each_domain", 0.1),
                        ("original_train_count", 2600), ("gntrans_train_count", 2600),
                        ("mixed_train_count", 5200), ("paired_scene_frame_sampling", True),
                        ("gntrans_fused_background", True),
                        ("gntrans_observed_depth_network_input", False)):
        _same_value(_field(m, k, str(mpath)), expected, f"{variant}.mixed_protocol.{k}")
    audits = [{"variant": variant, "split": "train", "file_type": label, "path": str(p), "sha256": sha256_file(p)}
              for label, p in (("gvar_protocol", gpath), ("gntrans_mix_protocol", mpath))]
    return g, m, audits


@dataclass
class SplitData:
    variant: str
    split: str
    tensor: np.ndarray
    protocol: dict
    summary: dict
    audit: list[dict]


def _load_split(root: Path, variant: str, split: str, epoch: int) -> SplitData:
    d = root / "eval" / f"{variant}_e{epoch}" / split
    ppath, spath = d / "gvar_inference_protocol.json", d / "gvar_inference_summary.json"
    proto, summ = _json(ppath), _json(spath)
    _same_value(_field(proto, "gvar_config", str(ppath))["variant"], variant, f"{d}.variant")
    for key, value in (("split", split), ("camera", "realsense"), ("completed_epoch", epoch),
                       ("selected_samples", 780), ("frame_stride", 10), ("batch_size", 3),
                       ("gvar_contract_version", 1), ("collision_source", "original_sensor"), ("network_geometry", "predicted metric depth"),
                       ("depth_assisted_dataset_preprocessing", True)):
        _same_value(_field(proto, key, str(ppath)), value, f"{d}.{key}")
    for key, value in (("collision_thresh", 0.01), ("collision_voxel_size", 0.01)):
        _require(abs(float(_field(proto, key, str(ppath))) - value) <= 1e-9, f"{d}.{key} must be {value}")
    _same_value(_field(proto, "frame_fingerprint", str(ppath)), _fingerprint(split), f"{d}.frame_fingerprint (canonical scene/frame order)")
    _require(bool(summ.get("complete")), f"Incomplete inference at {spath}")
    _same_value(_field(summ, "valid_dump_count", str(spath)), 780, f"{d}.valid_dump_count")
    _same_value(_field(summ, "expected_count", str(spath)), 780, f"{d}.expected_count")
    ck = _field(proto, "checkpoint_sha256", str(ppath))
    _require(isinstance(ck, str) and bool(HASH_RE.fullmatch(ck)), f"Bad checkpoint_sha256 at {ppath}")
    path = d / f"ap_{split}_realsense.npy"
    _require(path.is_file(), f"Missing official AP tensor: {path}")
    try:
        ap = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AuditError(f"Cannot load AP tensor {path}: {exc}") from exc
    _same_value(ap.shape, SHAPE, f"{path}.shape")
    _require(ap.dtype.kind == "f", f"AP tensor must be floating-point: {path} has {ap.dtype}")
    _require(bool(np.isfinite(ap).all()), f"AP has NaN/Inf values: {path}")
    _require(float(ap.min()) >= -1e-6 and float(ap.max()) <= 1.0 + 1e-6,
             f"AP outside probability [0,1] range: {path}, min/max={ap.min()}/{ap.max()}")
    audits = [{"variant": variant, "split": split, "file_type": name, "path": str(p), "sha256": sha256_file(p)}
              for name, p in (("inference_protocol", ppath), ("inference_summary", spath), ("official_ap", path))]
    return SplitData(variant, split, ap, proto, summ, audits)


def load_all(root: Path, variants: list[str], epoch: int) -> tuple[dict, list[dict]]:
    records, audits, common_train, common_mix = {}, [], None, None
    _require(root.is_dir(), f"GVAR root is not a directory: {root}")
    for variant in variants:
        train, mix, a = _train_contract(root, variant)
        audits.extend(a)
        if common_train is None:
            common_train, common_mix = train, mix
        else:
            for key in ("base_main_sha", "architecture_config", "detach_policy", "seed", "world_size",
                        "per_rank_batch", "global_batch", "train_original", "train_gntrans",
                        "train_steps_per_epoch", "frame_stride"):
                _same_value(_field(train, key, f"train/{variant}"), _field(common_train, key, "train/baseline"), f"training {key} vs baseline")
            for key in ("original_train_index_sha256", "gntrans_train_index_sha256", "original_seen_index_sha256",
                        "gntrans_seen_index_sha256", "train_fraction_each_domain", "eval_fraction_each_domain",
                        "gntrans_fused_background", "gntrans_object_depth_supervision"):
                _same_value(_field(mix, key, f"train/{variant}"), _field(common_mix, key, "train/baseline"), f"training index/supervision {key}")
        data = {split: _load_split(root, variant, split, epoch) for split in SPLITS}
        audits.extend(a for entry in data.values() for a in entry.audit)
        ckset = {data[split].protocol["checkpoint_sha256"] for split in SPLITS}
        _require(len(ckset) == 1, f"{variant} uses DIFFERENT checkpoints across three test splits: {ckset}")
        for split, entry in data.items():
            _same_value(entry.protocol["seed"], train["seed"], f"{variant}/{split} train/inference seed")
            _same_value(entry.protocol["gvar_config"]["variant"], train["gvar_config"]["variant"], f"{variant}/{split} train/inference variant")
            _same_value(entry.protocol["camera"], "realsense", f"{variant}/{split} camera")
            # Memory-chunk and activation-checkpoint flags are execution-only.
            semantic = lambda cfg: {k: v for k,v in cfg.items() if k not in ("action_chunk", "activation_checkpoint")}
            _same_value(semantic(entry.protocol["gvar_config"]), semantic(train["gvar_config"]),
                        f"{variant}/{split} train/inference reader physical config")
            _same_value(semantic(entry.protocol["gvar_config"]), semantic(data[SPLITS[0]].protocol["gvar_config"]),
                        f"{variant}/{split} per-split reader physical config")
        records[variant] = {"train": train, "mix": mix, "data": data, "ckpt_sha256": next(iter(ckset))}
    reference = records[variants[0]]["data"]
    for variant in variants[1:]:
        for split in SPLITS:
            p, q = reference[split].protocol, records[variant]["data"][split].protocol
            for key in ("frame_fingerprint", "frame_stride", "selected_samples", "batch_size", "dataset_root", "camera", "collision_thresh",
                        "collision_voxel_size", "collision_source", "network_geometry", "depth_assisted_dataset_preprocessing",
                        "completed_epoch", "seed"):
                _same_value(_field(q, key, f"eval/{variant}/{split}"), _field(p, key, f"eval/{variants[0]}/{split}"), f"paired {split}.{key} ({variant} vs {variants[0]})")
    return records, audits


def scene_metric(tensor: np.ndarray, metric: str) -> np.ndarray:
    """Return [30] probabilities: each scene has exactly 26 matched frames."""
    if metric == "AP":
        return tensor.mean(axis=(1, 2, 3), dtype=np.float64)
    if metric.startswith("AP_mu"):
        mu = float(metric[5:])
        j = FRICTIONS.index(mu)
        return tensor[:, :, :, j].mean(axis=(1, 2), dtype=np.float64)
    if metric.startswith("prefix_precision@"):
        k = int(metric.split("@", 1)[1])
        return tensor[:, :, k - 1, :].mean(axis=(1, 2), dtype=np.float64)
    raise KeyError(metric)


def boot_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    _require(n >= 200, "Bootstrap repetitions must be >=200")
    rng = np.random.default_rng(seed)
    return {s: rng.integers(0, 30, size=(n, 30), dtype=np.int16) for s in SPLITS}


def paired_metric(vectors: dict[str, np.ndarray], index: dict[str, np.ndarray], ci: float, split: str) -> dict:
    """Vectors are signed AP fraction differences for each split's 30 scenes."""
    if split == "Mean":
        delta = np.concatenate([vectors[s] for s in SPLITS]);
        draws = np.mean(np.stack([vectors[s][index[s]].mean(axis=1) for s in SPLITS], axis=0), axis=0)
    else:
        delta = vectors[split]; draws = delta[index[split]].mean(axis=1)
    lower, upper = np.quantile(draws, [(1-ci)/2, (1+ci)/2]) * 100
    wins = int(np.count_nonzero(delta > 1e-9))
    losses = int(np.count_nonzero(delta < -1e-9))
    return {"delta_pp": float(delta.mean() * 100), "ci_low_pp": float(lower), "ci_high_pp": float(upper),
            "scene_wins": wins, "scene_losses": losses, "scene_ties": int(len(delta)-wins-losses),
            "scene_count": int(len(delta))}


def build_tables(records: dict, variants: list[str], baseline: str, index: dict[str, np.ndarray], ci: float):
    paired_summary, paired_metrics, scenes, frames = [], [], [], []
    for variant in variants:
        for split in SPLITS:
            b = records[baseline]["data"][split].tensor
            a = records[variant]["data"][split].tensor
            scene_ids = canonical_scene_ids(split)
            bm, am = scene_metric(b, "AP"), scene_metric(a, "AP")
            for row_idx, scene_id in enumerate(scene_ids):
                r = {"variant": variant, "split": split, "scene_id": scene_id,
                     "baseline_AP": float(bm[row_idx]*100), "variant_AP": float(am[row_idx]*100),
                     "delta_AP_pp": float((am[row_idx]-bm[row_idx])*100)}
                for name in METRIC_NAMES[1:]:
                    xb, xa = scene_metric(b, name)[row_idx], scene_metric(a, name)[row_idx]
                    r[f"delta_{name}_pp"] = float((xa-xb)*100)
                scenes.append(r)
                fb = b[row_idx].mean(axis=(1,2),dtype=np.float64)
                fa = a[row_idx].mean(axis=(1,2),dtype=np.float64)
                for k, frame in enumerate(FRAMES):
                    frames.append({"variant": variant, "split": split, "scene_id": scene_id, "frame_id": frame,
                                   "baseline_AP": float(fb[k]*100), "variant_AP": float(fa[k]*100),
                                   "delta_AP_pp": float((fa[k]-fb[k])*100)})
        for metric in METRIC_NAMES:
            vectors = {split: scene_metric(records[variant]["data"][split].tensor, metric)
                              - scene_metric(records[baseline]["data"][split].tensor, metric) for split in SPLITS}
            for split in (*SPLITS, "Mean"):
                v = (np.mean([scene_metric(records[variant]["data"][s].tensor, metric).mean() for s in SPLITS])
                     if split=="Mean" else scene_metric(records[variant]["data"][split].tensor, metric).mean())*100
                base = (np.mean([scene_metric(records[baseline]["data"][s].tensor, metric).mean() for s in SPLITS])
                        if split=="Mean" else scene_metric(records[baseline]["data"][split].tensor, metric).mean())*100
                stats = paired_metric(vectors,index,ci,split)
                row = {"variant":variant,"baseline":baseline,"split":split,"metric":metric,
                       "baseline_pct":float(base),"variant_pct":float(v),**stats}
                paired_metrics.append(row)
                if metric=="AP": paired_summary.append(row)
    return paired_summary,paired_metrics,scenes,frames


def extra_contrasts(records: dict, contrasts: list[tuple[str, str]], index: dict[str, np.ndarray], ci: float) -> tuple[list[dict], list[dict]]:
    """Between non-baseline variants: volume vs volume_rel; slot vs volume; etc."""
    rows, scene_rows = [], []
    for reference, alternative in contrasts:
        for split in SPLITS:
            ref = records[reference]["data"][split].tensor
            alt = records[alternative]["data"][split].tensor
            rp, ap = scene_metric(ref, "AP"), scene_metric(alt, "AP")
            for i, sid in enumerate(canonical_scene_ids(split)):
                scene_rows.append({"reference": reference, "variant": alternative, "split": split,
                                   "scene_id": sid, "reference_AP": float(rp[i]*100),
                                   "variant_AP": float(ap[i]*100), "delta_AP_pp": float((ap[i]-rp[i])*100)})
        for metric in METRIC_NAMES:
            vectors = {sp: scene_metric(records[alternative]["data"][sp].tensor, metric)
                           - scene_metric(records[reference]["data"][sp].tensor, metric) for sp in SPLITS}
            for split in (*SPLITS, "Mean"):
                def metric_pct(v):
                    if split == "Mean":
                        return float(np.mean([scene_metric(records[v]["data"][sp].tensor, metric).mean() for sp in SPLITS]) * 100)
                    return float(scene_metric(records[v]["data"][split].tensor, metric).mean() * 100)
                rows.append({"reference": reference, "variant": alternative, "split": split, "metric": metric,
                             "reference_pct": metric_pct(reference), "variant_pct": metric_pct(alternative),
                             **paired_metric(vectors,index,ci,split)})
    return rows, scene_rows


def decide_contrasts(variants: list[str], requested: list[str] | None) -> list[tuple[str,str]]:
    """Fail when a requested pair is absent; automatic optional pairs never hide missing runs."""
    pairs = [(a,b) for a,b in (("slot","volume"),("volume","volume_rel"),("volume_fixed","volume"))
             if a in variants and b in variants]
    for item in requested or []:
        chunks = item.split(":")
        _require(len(chunks)==2 and all(chunks), f"--contrast expects REFERENCE:VARIANT, got {item!r}")
        a,b=chunks
        _require(a in variants and b in variants, f"--contrast {item}: both variants must be present in --variants")
        _require(a!=b, f"--contrast cannot compare a variant with itself: {item}")
        if (a,b) not in pairs:
            pairs.append((a,b))
    return pairs


def _csv(path: Path, rows: list[dict]) -> None:
    _require(bool(rows), f"Cannot write an empty table: {path}")
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader(); writer.writerows(rows)


def markdown(paired_summary: list[dict], paired_metrics: list[dict], scenes: list[dict], variants: list[str],baseline: str, nboot:int, ci:float,seed:int, epoch:int, root:Path, contrasts: list[dict])->str:
    by = {(r["variant"],r["split"]):r for r in paired_summary}
    out = [f"# GVAR scene-paired analysis — e{epoch}","",
           f"Source: `{root}`. Methods: Original GraspNet RealSense, each split 30 scenes × 26 frames (0,10,…,250), `[30,26,50,6]` official AP arrays.",
           f"Baseline `{baseline}`. Paired scene-cluster bootstrap within each split, {nboot:,} draws, random seed={seed}, {ci:.0%} percentile CI. Units = **AP percentage points**.","",
           "## Main AP: same scene/frame IDs; independently generated grasps","",
           "| Variant | Seen AP | Similar AP | Novel AP | Mean AP | ΔMean [CI] |", "|---|---:|---:|---:|---:|---|"]
    for v in variants:
        r=[by[(v,s)] for s in SPLITS]
        avg=by[(v,"Mean")]
        ci_str=("reference" if v==baseline else f"{avg['delta_pp']:+.3f} [{avg['ci_low_pp']:+.3f}, {avg['ci_high_pp']:+.3f}]")
        out.append(f"| {v} | {r[0]['variant_pct']:.3f} | {r[1]['variant_pct']:.3f} | {r[2]['variant_pct']:.3f} | {avg['variant_pct']:.3f} | {ci_str} |")
    out += ["", "## Paired AP changes by split", "",
            "| Variant | Split | ΔAP (pp) | Scene-bootstrap CI (pp) | Scene wins / ties / losses |",
            "|---|---|---:|---|---|"]
    for v in variants:
        if v==baseline: continue
        for s in (*SPLITS,"Mean"):
            r=by[(v,s)]; out.append(f"| {v} | {SPLIT_TITLE.get(s,s)} | {r['delta_pp']:+.3f} | [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}] | {r['scene_wins']}/{r['scene_ties']}/{r['scene_losses']} |")
    out += ["", "## Friction-specific paired AP changes (pp)", "",
            "| Variant | Split | μ=0.2 | μ=0.4 | μ=0.6 | μ=0.8 | μ=1.0 | μ=1.2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    lookup={(r["variant"],r["split"],r["metric"]):r for r in paired_metrics}
    for v in variants:
        if v==baseline:continue
        for s in SPLITS:
            values=[lookup[(v,s,f"AP_mu{mu:.1f}")]["delta_pp"] for mu in FRICTIONS]
            out.append("| "+" | ".join((v,SPLIT_TITLE[s],*(f"{x:+.3f}" for x in values)))+" |")
    out += ["", "## Prefix precision changes (pp; NOT candidate recall)", "",
            "| Variant | Split | @1 | @5 | @10 | @20 | @50 |", "|---|---|---:|---:|---:|---:|---:|"]
    for v in variants:
        if v==baseline:continue
        for s in SPLITS:
            vals=[lookup[(v,s,f"prefix_precision@{k}")]["delta_pp"] for k in PREFIXES]
            out.append("| "+" | ".join((v,SPLIT_TITLE[s],*(f"{x:+.3f}" for x in vals)))+" |")
    if contrasts:
        out += ["", "## Direct non-baseline comparisons (scene-paired)", "",
                f"| Reference → Variant | Split | ΔAP (pp) | {ci:.0%} scene-bootstrap CI (pp) | Scene wins / ties / losses |",
                "|---|---|---:|---|---|"]
        for r in contrasts:
            if r["metric"] != "AP": continue
            out.append(f"| {r['reference']} → {r['variant']} | {SPLIT_TITLE.get(r['split'],r['split'])} | {r['delta_pp']:+.3f} | [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}] | {r['scene_wins']}/{r['scene_ties']}/{r['scene_losses']} |")
    out += ["", "## Most changed Novel scenes (ranked by mean AP delta)", ""]
    for v in variants:
        if v==baseline:continue
        rows=[r for r in scenes if r["variant"]==v and r["split"]=="test_novel"]
        rows.sort(key=lambda x:x["delta_AP_pp"])
        worst=rows[:5]; best=list(reversed(rows[-5:]))
        out.append(f"**{v}** — worst: "+", ".join(f"{r['scene_id']:04d} ({r['delta_AP_pp']:+.2f})" for r in worst)
                   +"; best: "+", ".join(f"{r['scene_id']:04d} ({r['delta_AP_pp']:+.2f})" for r in best)+".")
    out += ["", "## Interpretation limits and next checks", "",
            "- The **same scene and frame IDs** are paired, but each model creates different physical grasps. AP/prefix changes **cannot be attributed solely to reranking**, and prefix precision is not candidate recall.",
            "- Percentile CIs cluster at the **scene** level (30 resampling units per split) and stratify three-split means. They **do not** cover training seeds, checkpoint selection, multiple comparisons, or scene-to-scene dependence.",
            "- Existing `volume_fixed` is optional only while untrained. To include it, require all three official splits and training/inference manifests; missing data must raise an error, not silently omit it.",
            "- For Novel, examine μ=0.4 and μ=0.8 separately before calling a variant robust. Differences whose CI includes zero are descriptive, not confirmed gains.",
            "- Check training step/runtime and e19 identity before extending the claim to different dataset sizes. Do not retrain or change grasp scoring to generate this report.",
            "", "## Generated artifacts", "",
            "`paired_summary.csv`, `paired_metrics.csv`, `scene_level.csv`, `frame_level.csv`, "
            "`cross_variant_contrasts.csv`, `cross_variant_scenes.csv`, `input_audit.csv`, and `gvar_scene_paired_audit.json`.", ""]
    return "\n".join(out)


def run(root: Path, output_dir: Path, variants: list[str], baseline: str, epoch: int, bootstrap: int, seed: int, ci: float, overwrite: bool=False, requested_contrasts: list[str] | None=None) -> dict:
    _require(0 < ci < 1, "CI must be in (0,1)")
    _require(epoch>=0,"Epoch must be nonnegative")
    _require(len(variants)==len(set(variants)), f"Duplicate variants supplied: {variants}")
    _require(baseline in variants, f"Baseline {baseline} absent from variants {variants}")
    _require(all(re.fullmatch(r"[a-z][a-z0-9_]*",x) for x in variants), "Invalid variant name")
    # Anchor all provenance comparisons to the explicitly named baseline.
    variants = [baseline] + [v for v in variants if v != baseline]
    records,audit=load_all(root,variants,epoch)  # ALL validation completes BEFORE output files
    idx=boot_indices(bootstrap,seed)
    paired_summary,paired_metrics,scenes,frames=build_tables(records,variants,baseline,idx,ci)
    contrast_pairs=decide_contrasts(variants, requested_contrasts)
    cross_rows,cross_scenes=extra_contrasts(records,contrast_pairs,idx,ci)
    content=markdown(paired_summary,paired_metrics,scenes,variants,baseline,bootstrap,ci,seed,epoch,root,cross_rows)
    output_dir=output_dir.resolve()
    if output_dir.exists():
        files={p.name for p in output_dir.iterdir()}
        _require(not files or (overwrite and files.issubset(OUTPUT_FILES)),
                 f"Refusing to overwrite existing/non-analysis output {output_dir}; choose fresh path or --overwrite for analysis-only outputs")
        if files:
            meta_path = output_dir / "gvar_scene_paired_audit.json"
            prior = _json(meta_path)
            for k, v in (("variants", variants), ("baseline", baseline), ("epoch", epoch),
                         ("direct_contrasts", [list(p) for p in contrast_pairs])):
                _same_value(_field(prior, k, str(meta_path)), v, f"overwrite existing analysis {k}; choose a new output directory")
    output_dir.mkdir(parents=True,exist_ok=True)
    _csv(output_dir/"paired_summary.csv",paired_summary)
    _csv(output_dir/"paired_metrics.csv",paired_metrics)
    _csv(output_dir/"scene_level.csv",scenes)
    _csv(output_dir/"frame_level.csv",frames)
    if cross_rows:
        _csv(output_dir/"cross_variant_contrasts.csv",cross_rows)
        _csv(output_dir/"cross_variant_scenes.csv",cross_scenes)
    _csv(output_dir/"input_audit.csv",audit)
    (output_dir/"REPORT.md").write_text(content,encoding="utf-8")
    meta={"tool_version":TOOL_VERSION,"root":str(root.resolve()),"output_dir":str(output_dir),
          "variants":variants,"baseline":baseline,"epoch":epoch,"bootstrap_draws":bootstrap,
          "bootstrap_seed":seed,"bootstrap_ci":ci,"bootstrap_unit":"scene, stratified by split, with replacement",
          "observations_per_scene":26,"scene_ids":{s:canonical_scene_ids(s) for s in SPLITS},
          "frame_ids":list(FRAMES),"friction_thresholds":list(FRICTIONS),"prefix_ranks":list(PREFIXES),
          "input_count":len(audit),"direct_contrasts":contrast_pairs,"checkpoint_sha256":{v:records[v]["ckpt_sha256"] for v in variants},
          "source_files":audit,"notes":["Original GraspNet only","AP points use percentage units", "prefix precision is not candidate recall", "Not a multi-seed comparison"]}
    (output_dir/"gvar_scene_paired_audit.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return {"rows":len(paired_summary),"scene_rows":len(scenes),"frame_rows":len(frames),"output_dir":str(output_dir),"mean_ap":{v:next(r["variant_pct"] for r in paired_summary if r["variant"]==v and r["split"]=="Mean") for v in variants}}


def main(argv:list[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root",type=Path,required=True,help="GVAR deployment results root containing train/ and eval/")
    p.add_argument("--output-dir",type=Path,required=True,help="New analysis-only directory; never overwrite train/eval")
    p.add_argument("--variants",nargs="+",default=["baseline","slot","volume","volume_rel"],help="Complete variants to compare; explicitly add volume_fixed only after its formal AP is available")
    p.add_argument("--baseline",default="baseline")
    p.add_argument("--epoch",type=int,default=19)
    p.add_argument("--bootstrap",type=int,default=20000)
    p.add_argument("--seed",type=int,default=20261009)
    p.add_argument("--ci",type=float,default=0.95)
    p.add_argument("--contrast",action="append",default=[],metavar="REFERENCE:VARIANT",
                   help="Extra direct comparison; automatic slot:volume, volume:volume_rel, and optional volume_fixed:volume")
    p.add_argument("--overwrite",action="store_true",help="Overwrite only previously generated analysis artifacts")
    args=p.parse_args(argv)
    try:
        report=run(args.root,args.output_dir,args.variants,args.baseline,args.epoch,args.bootstrap,args.seed,args.ci,args.overwrite,args.contrast)
    except (AuditError,OSError,KeyError,IndexError,ValueError) as exc:
        print(f"GVAR scene-paired analysis FAILED: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
