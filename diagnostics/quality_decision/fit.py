"""P0/P1 frozen-feature fitting with scene-disjoint probe splits.

The original GVAR/base checkpoint trained on all train scenes; only the NEW
quality probe is scene held out (not an unseen-scene backbone evaluation).
Official test labels are never used for model fit, val or checkpoint choice.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from rare_grasp.core import SCHEMA as RARE_SCHEMA, validate_cache
from rare_grasp.pipeline import load_index as load_rare_index
from .core import (SCHEMA, MODES, QualityProbe, digest, frame_metrics,
                   objective, predictor, scene_equal_aggregate, sha)


def load_entries(path: str, strategy: str) -> tuple[list[dict], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") == RARE_SCHEMA:
        index = load_rare_index(path, partition="train", allow_partial=False)
        entries=[dict(**e, strategy="uniform") for e in index["entries"]]
        contract={"origin":"RARE_MIX50_exact_action_cache",
                  "index_sha256":index["index_sha256"],
                  "base_config_digest":index["contract"]["config_digest"],
                  "source_checkpoint_sha256":
                    index["contract"]["config"]["checkpoint_sha256"],
                  "schema":RARE_SCHEMA}
    elif data.get("schema") == SCHEMA and data.get("kind") == "decision-cache-index":
        signature = data.pop("index_sha256", None)
        if signature != digest(data):
            raise ValueError("P1 cache index SHA mismatch")
        data["index_sha256"]=signature
        if not data.get("entries") or not data.get("base_checkpoint_sha256"):
            raise ValueError("Incomplete decision cache index")
        entries = list(data["entries"])
        contract={"origin":"P1_decision_cache",
                  "index_sha256":signature,
                  "base_config_digest":data["config_digest"],
                  "source_checkpoint_sha256":data["base_checkpoint_sha256"],
                  "schema":SCHEMA}
    else:
        raise ValueError("Unsupported cache index schema")
    if strategy not in ("uniform", "decision"):
        raise ValueError("Strategy must be uniform or decision")
    entries=[e for e in entries if e["strategy"] == strategy]
    if not entries:
        raise ValueError(f"No {strategy} entries in index")
    seen=set()
    for e in entries:
        row=e["row"]
        if row.get("split") != "train" or int(row["scene"]) not in range(100):
            raise ValueError("Only GraspNet train scene 0..99 permitted")
        key=(row["domain"], int(row["scene"]), int(row["anno"]))
        if key in seen:
            raise ValueError(f"Duplicate frame in strategy {strategy}: {key}")
        seen.add(key)
        if not Path(e["file"]).is_file() or sha(e["file"]) != e["sha256"]:
            raise ValueError(f"Cache artifact missing/modified: {e['file']}")
    return entries,contract


def select_by_scene(entries, start_scene, stop_scene, max_frames_per_scene):
    if not (0 <= start_scene < stop_scene <= 100):
        raise ValueError("Expected disjoint scene subset within train scene 0..99")
    groups={}
    for e in entries:
        row=e["row"]
        scene=int(row["scene"])
        if not start_scene <= scene < stop_scene:
            continue
        groups.setdefault((row["domain"], scene), []).append(e)
    if set(x[0] for x in groups) != {"original", "gntrans"}:
        raise ValueError("Both Original and GN-Trans domains required")
    result=[]
    for key in sorted(groups):
        options=sorted(groups[key],key=lambda e:int(e["row"]["anno"]))
        if max_frames_per_scene:
            # Deterministic approximately-uniform spread over the time samples.
            if len(options)<max_frames_per_scene:
                raise ValueError(f"Scene {key} only has {len(options)} frames")
            idx=np.linspace(0,len(options)-1,max_frames_per_scene,dtype=np.int64)
            result += [options[int(i)] for i in idx]
        else:
            result += options
    expected=(stop_scene-start_scene)*2
    if len(groups) != expected:
        raise ValueError(f"Expected {expected} domain-scene groups, found {len(groups)}")
    return result


def read_frame(entry):
    with np.load(entry["file"],allow_pickle=False) as z:
        x={k:z[k] for k in z.files}
    validate_cache(x)
    return x


@torch.no_grad()
def validate(model, entries, device, seed):
    model.eval()
    rows=[]
    for e in entries:
        x=read_frame(e)
        z=predictor(model,x,device,seed=seed+int(e["row"]["scene"])*1009+int(e["row"]["anno"]))
        metrics=frame_metrics(z.sigmoid().cpu().numpy(),x)
        metrics.update(domain=e["row"]["domain"],scene=int(e["row"]["scene"]),
                       anno=int(e["row"]["anno"]))
        rows.append(metrics)
    return scene_equal_aggregate(rows)


def fit(a):
    if a.epochs<1 or a.batch_frames<1 or a.learning_rate<=0 or a.weight_decay<0:
        raise ValueError("Invalid optimization hyperparameters")
    if a.mode not in MODES or a.pair_weight<0 or a.frames_per_scene<1:
        raise ValueError("Invalid probe/label budget")
    if a.val_scene_start<=a.train_scene_start or a.train_scene_end>a.val_scene_start or a.val_scene_end>100:
        raise ValueError("Scene validation must not overlap probe training")
    entries,contract=load_entries(a.index,a.strategy)
    train=select_by_scene(entries,a.train_scene_start,a.train_scene_end,a.frames_per_scene)
    valid=select_by_scene(entries,a.val_scene_start,a.val_scene_end,a.frames_per_scene)
    train_keys={(e["row"]["domain"],e["row"]["scene"],e["row"]["anno"]) for e in train}
    valid_keys={(e["row"]["domain"],e["row"]["scene"],e["row"]["anno"]) for e in valid}
    if train_keys&valid_keys:
        raise RuntimeError("Probe train/val frame leakage")
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty, new output directory")
    out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device(a.device)
    model=QualityProbe(a.mode,hidden=a.hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    config=dict(schema=SCHEMA,mode=a.mode,strategy=a.strategy,
                base_checkpoint_sha256=contract["source_checkpoint_sha256"],
                base_config_digest=contract["base_config_digest"],
                cache_contract=contract,index_path=str(Path(a.index).resolve()),
                train_scenes=[a.train_scene_start,a.train_scene_end],
                val_scenes=[a.val_scene_start,a.val_scene_end],
                frames_per_scene=a.frames_per_scene,
                train_frames=len(train),val_frames=len(valid),
                train_scene_count=len({e["row"]["scene"] for e in train}),
                val_scene_count=len({e["row"]["scene"] for e in valid}),
                seed=a.seed,hidden=a.hidden,epochs=a.epochs,
                pair_weight=a.pair_weight,pair_scope=a.pair_scope,
                batch_frames=a.batch_frames,max_pairs=a.max_pairs,
                learning_rate=a.learning_rate,weight_decay=a.weight_decay,
                trainable_parameters=sum(p.numel() for p in model.parameters()),
                test_labels_used=False,validation="probe-scene-held-out_only; base trained on all train scenes",
                selection="best validation quality MSE; no test AP selection")
    (out/"protocol.json").write_text(json.dumps(config,indent=2,ensure_ascii=False))
    best=float("inf")
    history=[]
    order_rng=np.random.default_rng(a.seed)
    for epoch in range(a.epochs+1):
        if epoch>0:
            model.train()
            order=order_rng.permutation(len(train))
            losses=[]
            for start in range(0,len(order),a.batch_frames):
                opt.zero_grad(set_to_none=True)
                frame_loss=[]
                for i in order[start:start+a.batch_frames]:
                    e=train[int(i)]
                    x=read_frame(e)
                    z=predictor(model,x,device,
                                seed=a.seed+epoch*1000003+int(e["row"]["scene"])*1009+int(e["row"]["anno"]))
                    counts=torch.as_tensor(x["count"],device=device,dtype=torch.long)
                    family=torch.as_tensor(x["family"],device=device,dtype=torch.long)
                    val,_=objective(z,counts,family,pair_weight=a.pair_weight,
                                    pair_scope=a.pair_scope,max_pairs=a.max_pairs,
                                    seed=a.seed+epoch*1000003+int(e["row"]["scene"])*1009+int(e["row"]["anno"]))
                    frame_loss.append(val)
                loss=torch.stack(frame_loss).mean()
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite quality loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                losses.append(float(loss.detach().cpu()))
        else:
            losses=[]
        metrics=validate(model,valid,device,a.seed)
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)) if losses else None,
                            val=metrics))
        payload=dict(schema=SCHEMA,epoch=epoch,model=model.state_dict(),
                     config=config,optimizer=opt.state_dict(),
                     train_state={"seed":a.seed,"torch_rng":torch.get_rng_state(),
                                  "numpy_rng":order_rng.bit_generator.state})
        def save(path):
            tmp=path.with_suffix(".tmp")
            torch.save(payload,tmp)
            os.replace(tmp,path)
        save(out/"last.pt")
        if metrics["mse"] < best:
            best=metrics["mse"];save(out/"best.pt")
            best_epoch=epoch
        (out/"history.json").write_text(json.dumps(history,indent=2))
        print(f"[probe {a.mode}/{a.strategy}] epoch={epoch}/{a.epochs} val={metrics}",flush=True)
    (out/"complete.json").write_text(json.dumps(
        dict(schema=SCHEMA,status="complete",best_epoch=best_epoch,best_val_mse=best,
             last_sha256=sha(out/"last.pt"),best_sha256=sha(out/"best.pt"),
             protocol_sha256=digest(config),test_labels_used=False),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",required=True)
    p.add_argument("--strategy",choices=("uniform","decision"),default="uniform")
    p.add_argument("--mode",choices=MODES,required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=20261113)
    p.add_argument("--train-scene-start",type=int,default=0)
    p.add_argument("--train-scene-end",type=int,default=80)
    p.add_argument("--val-scene-start",type=int,default=80)
    p.add_argument("--val-scene-end",type=int,default=90)
    p.add_argument("--frames-per-scene",type=int,default=4)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-frames",type=int,default=4)
    p.add_argument("--hidden",type=int,default=64)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--pair-weight",type=float,default=0.)
    p.add_argument("--pair-scope",choices=("image","family"),default="image")
    p.add_argument("--max-pairs",type=int,default=2048)
    fit(p.parse_args())


if __name__=="__main__":
    main()
