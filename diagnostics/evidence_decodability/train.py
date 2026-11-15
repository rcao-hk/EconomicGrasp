"""P1 fixed-budget physical evidence decodability: quality OR failure heads.

Exactly the same native executed actions per frame across all evidence modes,
with the same network parameter count, training scenes and optimization steps.
No 6D grasp backbone update, no official test labels and no test-AP selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.decision_supervision.core import BASE_SHA
from .core import (
    SCHEMA, MODES, TASKS, EvidenceProbe, check_bank, failure_class,
    failure_metrics, quality_stats, digest, sha256, write_json,
)
from .p0_match import load_manifest, read_bank


def _frame_subset(bank,row,old_count=32,new_count=32):
    """Deterministic same-frame 32 old + 32 new native actions, no Q-based sampling."""
    old=np.flatnonzero(np.asarray(bank["source_new"])==0)
    new=np.flatnonzero(np.asarray(bank["source_new"])==1)
    if len(old)<old_count or len(new)<new_count:
        raise ValueError(f"Missing matched candidate budget {row['domain']}/{row['scene']}/{row['anno']}: "
                         f"old={len(old)} new={len(new)}")
    key=f"{row['domain']}:{row['scene']}:{row['anno']}:P1-evidence-v1"
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")
    rng=np.random.default_rng(seed)
    ids=np.sort(np.concatenate([rng.choice(old,old_count,replace=False),
                                rng.choice(new,new_count,replace=False)]))
    if len(ids)!=64 or len(np.unique(ids))!=64:
        raise RuntimeError("Matched 64 executed-action budget failed")
    return {k:(v[ids] if isinstance(v,np.ndarray) and len(v)==len(bank["query_id"]) else v)
            for k,v in bank.items()}


def split_entries(manifest):
    grouped={"fit":[],"validation":[],"reused_audit":[]}
    for e in manifest["records"]:
        scene=int(e["row"]["scene"])
        stage="fit" if scene<80 else "validation" if scene<90 else "reused_audit"
        grouped[stage].append(e)
    expected={"fit":320,"validation":40,"reused_audit":40}
    if any(len(grouped[k])!=v for k,v in expected.items()):
        raise RuntimeError(f"Expected 320/40/40 train-scene frame split: "
                           f"{ {k:len(v) for k,v in grouped.items()} }")
    for name,rows in grouped.items():
        keys=[(r["row"]["domain"],int(r["row"]["scene"]),int(r["row"]["anno"]))
              for r in rows]
        if len(set(keys))!=len(keys):
            raise RuntimeError(f"Duplicate {name} frame")
    return grouped


def load_examples(entries):
    """Read each immutable frame once; RAM is not assumed sufficient for all 400."""
    examples=[]
    for e in entries:
        bank=read_bank(e)
        examples.append((e["row"],_frame_subset(bank,e["row"])))
    return examples


@torch.inference_mode()
def evaluate(model,entries,device):
    model.eval()
    rows=[]
    confusion=np.zeros((4,4),dtype=np.int64)
    for entry in entries:
        x=read_bank(entry)
        bank=_frame_subset(x,entry["row"])
        prediction=model(bank)
        row=entry["row"]
        if model.task=="quality":
            m=quality_stats(prediction.sigmoid().cpu().numpy(),
                            np.asarray(bank["quality_count"],dtype=float)/6.)
        else:
            truth=failure_class(bank["quality_count"],bank["pure_collision"],bank["empty"])
            m=failure_metrics(prediction.cpu().numpy(),truth)
            confusion+=np.asarray(m["confusion"],dtype=np.int64)
            m={k:v for k,v in m.items() if k!="confusion"}
        rows.append(dict(domain=row["domain"],scene=int(row["scene"]),
                         anno=int(row["anno"]),**m))
    if model.task=="quality":
        keys=("mse","pair_agreement","top10_quality","top10_zero")
        out={}
        for key in keys:
            by_domain=[]
            for d in ("original","gntrans"):
                per_scene=[]
                for scene in sorted({r["scene"] for r in rows if r["domain"]==d}):
                    values=[r[key] for r in rows if r["domain"]==d
                            and r["scene"]==scene and r[key] is not None]
                    if values:per_scene.append(np.mean(values))
                if per_scene:by_domain.append(np.mean(per_scene))
            out[key]=float(np.mean(by_domain)) if by_domain else None
    else:
        n=confusion.sum(axis=1);tp=np.diag(confusion)
        recall=np.divide(tp,n,out=np.full(4,np.nan,dtype=float),where=n>0)
        p=confusion.sum(axis=0)
        f1=np.divide(2*tp,2*tp+(n-tp)+(p-tp),
                     out=np.full(4,np.nan,dtype=float),
                     where=(2*tp+(n-tp)+(p-tp))>0)
        out=dict(class_recall=[float(v) if np.isfinite(v) else None for v in recall],
                 class_f1=[float(v) if np.isfinite(v) else None for v in f1],
                 balanced_accuracy=float(np.nanmean(recall)),
                 macro_f1=float(np.nanmean(f1)),
                 confusion=confusion.tolist())
    out["frames"]=len(rows)
    return out,rows


def _checkpoint(root,name,epoch,model,opt,protocol):
    path=root/name
    if path.exists():raise FileExistsError(path)
    tmp=root/(name+".tmp")
    torch.save(dict(schema=SCHEMA,kind="physical-evidence-probe",
                    epoch=epoch,model=model.state_dict(),
                    optimizer=opt.state_dict(),protocol=protocol),tmp)
    tmp.replace(path)


def fit(args):
    if args.mode not in MODES or args.task not in TASKS:
        raise ValueError("Unknown P1 task/evidence")
    if args.epochs<1 or args.batch_frames<1 or args.learning_rate<=0 or args.seed<0:
        raise ValueError("Invalid training settings")
    mf=load_manifest(args.cache_manifest)
    if mf["contract"]["base_checkpoint_sha256"]!=BASE_SHA:
        raise ValueError("Wrong GVAR base checkpoint")
    splits=split_entries(mf)
    first=read_bank(splits["fit"][0])
    hidden=check_bank(first)
    if args.mode=="reference_geometry":
        if any(not rec.get("has_reference") for rec in mf["records"]):
            raise ValueError("Privileged oracle mode requires every train/validation frame to have reference geometry")
    root=Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Use a new P1 training output: {root}")
    root.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device)
    model=EvidenceProbe(hidden,args.mode,args.task,width=args.width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    protocol=dict(schema=SCHEMA,base_checkpoint_sha256=BASE_SHA,
                  cache_manifest_sha256=mf["manifest_sha256"],
                  evidence_mode=args.mode,task=args.task,hidden_dim=hidden,
                  head_width=args.width,seed=args.seed,epochs=args.epochs,
                  train_frames=len(splits["fit"]),validation_frames=len(splits["validation"]),
                  audit_frames=len(splits["reused_audit"]),
                  scenes=dict(fit=[0,80],validation=[80,90],reused_audit=[90,100]),
                  native_executed_depth_only=True,actions_per_frame=64,
                  sampler="32 old + 32 new; fixed SHA frame seed, no labels used for selection",
                  optimizer="AdamW",learning_rate=args.learning_rate,
                  weight_decay=args.weight_decay,batch_frames=args.batch_frames,
                  trainable_params=sum(p.numel() for p in model.parameters()),
                  selected_epoch="fixed last",test_labels_used=False,
                  privileged_reference=args.mode=="reference_geometry",
                  code_sha256=sha256(__file__),
                  core_sha256=sha256(Path(__file__).with_name("core.py")))
    write_json(root/"protocol.json",protocol)
    order_rng=np.random.default_rng(args.seed)
    history=[]
    for epoch in range(args.epochs+1):
        if epoch:
            model.train()
            order=order_rng.permutation(len(splits["fit"]))
            losses=[]
            for start in range(0,len(order),args.batch_frames):
                opt.zero_grad(set_to_none=True)
                partial=[]
                for idx in order[start:start+args.batch_frames]:
                    entry=splits["fit"][int(idx)]
                    x=_frame_subset(read_bank(entry),entry["row"])
                    output=model(x)
                    if args.task=="quality":
                        target=torch.as_tensor(x["quality_count"],device=device,
                                               dtype=torch.float32)/6.
                        loss=10*F.mse_loss(output.sigmoid(),target)
                    else:
                        target=torch.as_tensor(failure_class(
                            x["quality_count"],x["pure_collision"],x["empty"]),
                            device=device,dtype=torch.long)
                        loss=F.cross_entropy(output,target)
                    partial.append(loss)
                step=torch.stack(partial).mean()
                if not torch.isfinite(step):
                    raise FloatingPointError("Nonfinite evidence probe loss")
                step.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                losses.append(float(step.detach().cpu()))
        else:losses=[]
        val,_=evaluate(model,splits["validation"],device)
        history.append(dict(epoch=epoch,validation=val,
                            train_loss=float(np.mean(losses)) if losses else None))
        print(f"[P1 {args.task}/{args.mode} seed={args.seed}] "
              f"epoch={epoch}/{args.epochs} val={val}",flush=True)
        if epoch==args.epochs:
            _checkpoint(root,"last.pt",epoch,model,opt,protocol)
    write_json(root/"history.json",history)
    audit,_=evaluate(model,splits["reused_audit"],device)
    write_json(root/"reused_audit.json",dict(
        metadata="Already inspected scenes; not independent new confirmation",
        metrics=audit))
    write_json(root/"complete.json",dict(
        schema=SCHEMA,status="complete",epoch=args.epochs,
        last_sha256=sha256(root/"last.pt"),
        protocol_sha256=digest(protocol),
        test_labels_used=False,
        validation_metrics=history[-1]["validation"],
        reused_audit_metrics=audit))


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--cache-manifest",required=True)
    a.add_argument("--mode",choices=MODES,required=True)
    a.add_argument("--task",choices=TASKS,default="quality")
    a.add_argument("--output",required=True)
    a.add_argument("--device",default="cuda:0")
    a.add_argument("--seed",type=int,default=20261113)
    a.add_argument("--epochs",type=int,default=10)
    a.add_argument("--batch-frames",type=int,default=4)
    a.add_argument("--learning-rate",type=float,default=1e-4)
    a.add_argument("--weight-decay",type=float,default=1e-4)
    a.add_argument("--width",type=int,default=64)
    fit(a.parse_args())


if __name__=="__main__":
    main()
