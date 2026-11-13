"""P0/P1 one-time scorer confirmation on untouched GraspNet train-scene subset 90–99.

These scenes were used to train the FROZEN underlying GVAR backbone; they are
only disjoint from the NEW scorer's fit/validation. Not a novel-object test.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .core import QualityProbe, SCHEMA, frame_metrics, predictor, scene_equal_aggregate, sha
from .fit import load_entries, select_by_scene, read_frame


def confirm(a):
    data=torch.load(a.weights,map_location="cpu",weights_only=False)
    if data.get("schema")!=SCHEMA:
        raise ValueError("Wrong scorer schema")
    cfg=data["config"]
    if cfg.get("test_labels_used") is not False:
        raise ValueError("Test label leakage contract")
    if cfg["train_scenes"] != [0,80] or cfg["val_scenes"] != [80,90]:
        raise ValueError("Only preregistered 0:80 training / 80:90 model validation permitted")
    if cfg["strategy"]!=a.strategy:
        raise ValueError("Strategy does not match scorer training")
    entries,contract=load_entries(a.index,a.strategy)
    if contract["index_sha256"]!=cfg["cache_contract"]["index_sha256"]:
        raise ValueError("Confirmation cache differs from trained model's index")
    selected=select_by_scene(entries,90,100,cfg["frames_per_scene"])
    device=torch.device(a.device)
    model=QualityProbe(cfg["mode"],cfg["hidden"]).to(device)
    model.load_state_dict(data["model"],strict=True)
    model.eval().requires_grad_(False)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.inference_mode():
        for e in selected:
            x=read_frame(e);r=e["row"]
            z=predictor(model,x,device,seed=cfg["seed"]+int(r["scene"])*1009+int(r["anno"]))
            scores=z.sigmoid().detach().cpu().numpy()
            metrics=frame_metrics(scores,x)
            metrics.update(domain=r["domain"],scene=r["scene"],anno=r["anno"],
                           strategy=a.strategy,checkpoint_sha256=sha(a.weights))
            rows.append(metrics)
    summary=dict(schema=SCHEMA,scope="probe_confirmation_only",
                 underlying_base_saw_train_scenes=True,
                 scorer_train_scenes=[0,80],scorer_val_scenes=[80,90],
                 confirmation_scenes=[90,100],train_or_validation_reuse=False,
                 source_index_sha256=contract["index_sha256"],
                 checkpoint_sha256=sha(a.weights),
                 metrics=scene_equal_aggregate(rows),frames=len(rows))
    with (out/"per_frame.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary["metrics"],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",required=True)
    p.add_argument("--index",required=True)
    p.add_argument("--strategy",choices=("uniform","decision"),default="uniform")
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0")
    confirm(p.parse_args())


if __name__=="__main__":
    main()
