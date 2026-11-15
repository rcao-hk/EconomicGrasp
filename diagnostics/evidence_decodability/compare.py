"""Paired official GraspNet AP across evidence modes, seeds, and 3 test splits.

Scenes are the bootstrap unit. Seed variability is recorded separately; the
same frozen base does NOT make these independent network training replications.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import SCHEMA, sha256, write_json
SPLITS=("test_seen","test_similar","test_novel")


def _read_run(root,split):
    root=Path(root)
    mf=json.loads((root/"inference.json").read_text(encoding="utf-8"))
    ev=json.loads((root/"evaluation.json").read_text(encoding="utf-8"))
    if (mf.get("schema")!=SCHEMA or mf.get("kind")!="evidence-quality-inference"
            or not mf.get("complete") or mf.get("split")!=split or
            ev.get("status")!="complete" or ev.get("inference_sha256")!=sha256(root/"inference.json")):
        raise ValueError(f"Incomplete/nonmatching official AP run: {root}")
    ap_file=root/f"ap_{split}_realsense.npy"
    if sha256(ap_file)!=ev["ap_sha256"]:
        raise ValueError(f"Official AP changed: {ap_file}")
    ap=np.load(ap_file,allow_pickle=False)
    expected=(30,len(range(0,256,mf["interval"])),50,6)
    if ap.shape!=expected or not np.isfinite(ap).all():
        raise ValueError(f"Malformed AP array: {ap.shape}")
    identity={k:mf[k] for k in
              ("base_checkpoint_sha256","base_config_digest",
               "bound_config_sha256","runtime","interval",
               "snapshot_seed","collision","selection")}
    frame_banks={(r["scene"],r["anno"]):r["frozen_bank_sha256"]
                 for r in mf["records"]}
    if len(frame_banks)!=len(mf["records"]):
        raise ValueError("Duplicate frozen-frame bank identity")
    return mf,ap.mean(axis=(1,2,3))*100.,identity,frame_banks


def compare_runs(runs, bootstrap=5000,seed=20261115):
    if bootstrap<1:
        raise ValueError("Bootstrap must be positive")
    seeds=sorted(runs)
    if len(seeds)<1:
        raise ValueError("Empty run mapping")
    all_modes=sorted(set.union(*(set(v) for v in runs.values()))-{"native"})
    if not all_modes:
        raise ValueError("Require at least one learned P1 probe")
    raw={}
    contract=None
    expected_banks={}
    for s in seeds:
        names=runs[s]
        if set(names)!={"native",*all_modes}:
            raise ValueError("Every seed must have matched native and all same learned modes")
        raw[s]={}
        for mode in sorted(names):
            if set(names[mode])!=set(SPLITS):
                raise ValueError(f"Missing split in {s}/{mode}")
            vals=[]
            for split in SPLITS:
                manifest,ap,ident,banks=_read_run(names[mode][split],split)
                if contract is None:contract=ident
                elif contract!=ident:
                    raise ValueError(f"Base/geometry/frames/collision mismatch: {s}/{mode}/{split}")
                if split not in expected_banks:expected_banks[split]=banks
                elif expected_banks[split]!=banks:
                    raise ValueError(f"Frozen physical grasp bank changed: {s}/{mode}/{split}")
                vals.append(ap)
            raw[s][mode]=np.stack(vals,axis=0)  # [split=3, scene=30]
    rng=np.random.default_rng(seed)
    out={}
    for mode in all_modes:
        samples=np.stack([raw[s][mode]-raw[s]["native"] for s in seeds],axis=0)
        per_seed=samples.mean(axis=(1,2))
        for_split=samples.mean(axis=(0,2))
        conditional_scene=samples.mean(axis=0)  # [3,30] after fixed heads averaged
        boot=np.zeros(bootstrap,dtype=np.float64)
        for split in range(3):
            ids=rng.integers(0,30,size=(bootstrap,30))
            boot += conditional_scene[split][ids].mean(axis=1)/3.
        out[mode]=dict(
            delta_pp=float(samples.mean()),
            per_split_delta_pp=dict(zip(SPLITS,for_split.tolist())),
            per_seed_delta_pp={s:float(v) for s,v in zip(seeds,per_seed)},
            seed_sd_pp=float(np.std(per_seed,ddof=1)) if len(seeds)>1 else None,
            conditional_scene_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
            constraint="same frozen base / same Original frames / same native executed grasp geometry")
    return dict(schema=SCHEMA,kind="P1_evidence_paired_official_AP",
                fixed_base=True,exploratory_repeated_test=True,
                seed_count=len(seeds),seed_values=seeds,
                bootstrap_unit="paired scene within split, seed mean first",
                bootstrap_draws=bootstrap,base_contract=contract,modes=out,
                limitation="No independent backbone seeds or untouched Novel benchmark")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",required=True,
                   help="JSON seed -> native/mode -> test_seen/similar/novel -> inference root")
    p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261115)
    a=p.parse_args()
    runs=json.loads(Path(a.runs).read_text(encoding="utf-8"))
    result=compare_runs(runs,a.bootstrap,a.seed)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Comparison output must be new")
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/"summary.json",result)
    lines=["# P1 GVAR物理证据：完整官方AP配对对照", "",
           "同一冻结基座、相同动作和collision协议。场景内配对，三个split等权；训练seed差异单列。",
           "| 模式 | Mean ΔAP (pp) | 95%条件场景CI | 训练seed SD (pp) |",
           "|---|---:|---|---:|"]
    for mode,v in result["modes"].items():
        lines.append(f"| {mode} | {v['delta_pp']:+.4f} | "
                     f"[{v['conditional_scene_ci95_pp'][0]:+.4f},"
                     f"{v['conditional_scene_ci95_pp'][1]:+.4f}] | "
                     f"{v['seed_sd_pp']} |")
    (root/"RESULTS_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result["modes"],indent=2))


if __name__=="__main__":
    main()
