#!/usr/bin/env python3
"""Paired official GraspNet AP: RayRoPE Point vs Point+GVAR, Mixed-v2.

This computes the same overall mean used by compare_moge_rayrope.py and
samples scenes, not frames or grasp ranks, for uncertainty intervals.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from rayrope_gvar.runtime import VERSION,VARIANTS,atomic_json

SPLITS=("test_seen","test_similar","test_novel")
BUDGET_KEYS=("main_commit","main_files","model","gvar","loss","dataset_root",
             "label_folder","train_fraction","eval_fraction","train_frames",
             "val_frames","sampling_sha256","use_fuse_depth","gntrans_rgb_root",
             "mixed_depth_supervision","epochs","seed","batch_per_gpu",
             "world_size","effective_batch","optimizer","lr","weight_decay",
             "grad_clip","lr_schedule","code_sha256","dav2_sha256")


def scene_scores(arr,mu=None):
    if arr.shape!=(30,26,50,6) or arr.dtype.kind!="f" or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite [30,26,50,6] official AP, got {arr.shape}/{arr.dtype}")
    if arr.min()<-1e-6 or arr.max()>1.000001:
        raise ValueError("GraspNet AP values must be in [0,1]")
    if mu is None:return arr.mean((1,2,3))
    index={.4:1,.8:3}[float(mu)]
    return arr[...,index].mean((1,2))


def load_comparison(root,modes=("on","off")):
    results={};protocols={};primary=None
    for variant in VARIANTS:
        tp_path=root/variant/"train"/"protocol.json"
        tp=json.loads(tp_path.read_text())
        if tp["version"]!=VERSION or tp["variant"]!=variant or tp["partial_run"]:
            raise RuntimeError(f"{variant}: wrong/partial training protocol")
        if tp["epochs"]!=20 or tp["seed"]!=0 or tp["train_frames"]!=5200 or tp["val_frames"]!=780:
            raise RuntimeError(f"{variant}: training budget is not the matched Mixed-v2 contract")
        if tp["world_size"]!=3 or tp["batch_per_gpu"]!=3 or tp["effective_batch"]!=9:
            raise RuntimeError(f"{variant}: mismatched world/batch")
        if tp["mixed_depth_supervision"]["contract_version"]!=2:
            raise RuntimeError("Mixed-v2 depth supervision required")
        budget={key:tp[key] for key in BUDGET_KEYS}
        if primary is not None and budget!=primary:
            raise RuntimeError(f"{variant}: changed training source/config/sampler/optimizer")
        primary=budget;protocols[variant]=tp
        for mode in modes:
            for split in SPLITS:
                infer=root/variant/"test_native"/f"test_collision_{mode}"
                mp=json.loads((infer/split/"protocol.json").read_text())
                ap_root=infer/"official"/split
                info=json.loads((ap_root/"summary.json").read_text())
                ap=np.load(ap_root/"accuracy.npy",allow_pickle=False)
                if mp["training_protocol"]!=tp or mp["split"]!=split or mp["depth_bias_mm"]!=0:
                    raise RuntimeError(f"{variant}/{mode}/{split}: unmatched inference")
                if mp.get("experiment_variant")!=variant or mp["batch_size"]!=3:
                    raise RuntimeError(f"{variant}/{mode}/{split}: wrong variant/batch")
                if mp["collision_filter"]=="none" and mode!="off":
                    raise RuntimeError("Expected collision-on primary evaluation")
                if mp["checkpoint_epoch"]!=19 or int(info["checkpoint_epoch"])!=19:
                    raise RuntimeError("All official evaluations must use epoch19")
                if abs(float(info["mean_accuracy"])-float(scene_scores(ap).mean()))>1e-8:
                    raise RuntimeError(f"Summary/AP mismatch for {variant}/{mode}/{split}")
                results[(variant,mode,split)]=ap
    for mode in modes:
        for split in SPLITS:
            a=root/"rayrope_point"/"test_native"/f"test_collision_{mode}"/split/"protocol.json"
            b=root/"rayrope_point_gvar"/"test_native"/f"test_collision_{mode}"/split/"protocol.json"
            x=json.loads(a.read_text());y=json.loads(b.read_text())
            for key in ("schedule","dataset_root","split","evaluation_fraction","batch_size",
                        "num_shards","seed","max_frames","collision_filter","preprocessing"):
                if x[key]!=y[key]:raise RuntimeError(f"Unpaired inference: {mode}/{split}/{key}")
            if len(x["schedule"])!=780:raise RuntimeError("Expected 780 inference frames per split")
    return results,protocols


def analyze(root,output,bootstrap=50000,seed=20261014):
    if bootstrap<200:raise ValueError("bootstrap must be >=200")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refuse to replace existing comparison report: {output}")
    results,protocols=load_comparison(root)
    rng=np.random.default_rng(seed)
    rows=[];scenes=[]
    for collision in ("on","off"):
        for label in ("AP","AP_mu0.4","AP_mu0.8"):
            sample_fn=(lambda ar:scene_scores(ar)) if label=="AP" else (lambda ar:scene_scores(ar,float(label[5:])))
            difference={}
            for split in SPLITS:
                a=sample_fn(results[("rayrope_point",collision,split)])
                b=sample_fn(results[("rayrope_point_gvar",collision,split)])
                difference[split]=b-a
                for j,delta in enumerate(difference[split]):
                    scenes.append({"collision":collision,"metric":label,"split":split,
                                   "scene_index":j,"delta_pp":100*float(delta)})
            for split in (*SPLITS,"Mean"):
                if split=="Mean":
                    delta=sum(difference[s].mean() for s in SPLITS)/3
                    boot=sum(difference[s][rng.integers(0,30,size=(bootstrap,30))].mean(axis=1)
                             for s in SPLITS)/3
                else:
                    delta=difference[split].mean()
                    boot=difference[split][rng.integers(0,30,size=(bootstrap,30))].mean(axis=1)
                rows.append({"collision":collision,"metric":label,"split":split,
                             "delta_pp":100*float(delta),
                             "ci_low_pp":100*float(np.quantile(boot,.025)),
                             "ci_high_pp":100*float(np.quantile(boot,.975))})
    absolute=[]
    for variant in VARIANTS:
        for mode in ("on","off"):
            row={"variant":variant,"collision":mode}
            for split in SPLITS:row[split]=100*float(scene_scores(results[(variant,mode,split)]).mean())
            row["Mean"]=sum(row[s] for s in SPLITS)/3
            absolute.append(row)
    output.mkdir(parents=True,exist_ok=True)
    for name,data in (("absolute.csv",absolute),("paired_effects.csv",rows),("scene_effects.csv",scenes)):
        with (output/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    atomic_json(output/"report.json",{"training_protocols":protocols,"absolute":absolute,
        "paired_effects":rows,"bootstrap":bootstrap,"seed":seed,
        "warning":"Single training seed; scene-bootstrap cannot measure seed variability."})
    lines=["# Mixed-v2 Point RayRoPE vs Point+GVAR","","Collision-on is the primary AP protocol.",
           "Both models have fresh heads; 10% RealSense plus 10% GN-Trans, same dataset/seed/epochs.",
           "","| Variant | Collision | Seen | Similar | Novel | Mean |",
           "|---|---|---:|---:|---:|---:|"]
    for row in absolute:
        lines.append("| "+row["variant"]+" | "+row["collision"]+" | "+" | ".join(
            f"{row[s]:.3f}" for s in (*SPLITS,"Mean"))+" |")
    lines+=["","## GVAR minus Point (scene-paired 95% CI)", "",
            "| Collision | Metric | Split | ΔAP (pp) | CI (pp) |",
            "|---|---|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r['collision']} | {r['metric']} | {r['split']} | "
                     f"{r['delta_pp']:+.3f} | [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}] |")
    lines+=["","**Interpretation boundary:** combined model changes the decoded candidate set and",
            "ranking. This is a matched retraining architecture comparison, not same-grasp",
            "reranking; a scene CI does not include retraining seed variability. Both models",
            "use depth-assisted GraspNet preprocessing and original-sensor collision filtering."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"absolute":absolute,"effects":rows,"output":str(output)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--bootstrap",type=int,default=50000)
    p.add_argument("--seed",type=int,default=20261014)
    a=p.parse_args();out=a.output or a.root/"comparison"
    print(json.dumps(analyze(a.root,out,a.bootstrap,a.seed)["absolute"],indent=2))


if __name__=="__main__":main()
