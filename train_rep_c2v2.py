#!/usr/bin/env python3
"""Train Rep-C2-v2 on full-path joint-error native/correction pairs.

Unlike Rep-C2-v1, the training labels here come from regenerated physical
anchors under joint depth errors. Frozen A1 fixed-0 supplies high-recall
correction proposals; the verifier learns whether each proposal is harmful,
equivalent, or beneficial relative to native.

Validation uses the already-produced full-path Seen split and fresh exact labels.
Similar/Novel are never used for threshold or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rep_a_common import (
    digest, exclusive_run, file_sha, load_torch, read_frame,
    save_json, save_torch, seed_for,
)
from rep_c2v2_common import (
    BENEFICIAL, CLASS_NAMES, VARIANTS, compact_from_source,
    frame_cache_path, full_query_gate_metrics, source_eval_path,
    source_files,
)
from rep_c2v2_model import RepC2V2Verifier, paired_initial_state


OBJECTIVE_VERSION="full_query_verifier_increment_oracle_gap_recovery_v1"


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root",required=True)
    p.add_argument("--train-cache-root",required=True)
    p.add_argument("--validation-source-root",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--reg-weight",type=float,default=.5)
    p.add_argument("--gain-weight",type=float,default=4.)
    p.add_argument("--dim",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--dropout",type=float,default=.1)
    p.add_argument("--seed",type=int,default=2028)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--val-cases",default="nominal,bias:-20,bias:20")
    p.add_argument("--val-move-limit",type=int,default=0,
                   help="Deprecated compatibility flag. Full-query calibration requires 0.")
    p.add_argument("--val-query-chunk",type=int,default=128,
                   help="Chunk moved proposals during full-query Seen validation")
    p.add_argument("--val-every",type=int,default=1)
    p.add_argument("--max-train-frames",type=int,default=0)
    p.add_argument("--max-val-files",type=int,default=0)
    p.add_argument("--progress-every",type=int,default=100)
    p.add_argument("--resume",action="store_true")
    return p


def load_train_examples(root, max_frames=0):
    files=sorted((Path(root)/"train").glob("scene_*/ann_*.npz"))
    if not files: raise FileNotFoundError(f"No Rep-C2-v2 train cache under {root}")
    return files[:max_frames] if max_frames>0 else files


def class_weights(files):
    counts=np.zeros(3,np.int64)
    examples=0
    for p in files:
        with np.load(p,allow_pickle=False) as z:
            y=np.asarray(z["target_class"],np.int64)
        counts += np.bincount(y,minlength=3)
        examples += len(y)
    if np.any(counts==0):
        raise RuntimeError(f"Missing Rep-C2-v2 target class: {dict(zip(CLASS_NAMES,counts.tolist()))}")
    w=1/np.sqrt(counts.astype(np.float64))
    w=w/w.mean()
    return counts,w.astype(np.float32),examples


def tensor_frame(frame, cache_frame, device):
    return {
        "image_feature":torch.from_numpy(np.asarray(cache_frame["image_feature"]).copy()).to(device).float(),
        "K":torch.from_numpy(np.asarray(cache_frame["K"]).copy()).to(device).float(),
        "actions":torch.from_numpy(np.asarray(frame["actions"]).copy()).to(device).float(),
        "probabilities":torch.from_numpy(np.asarray(frame["probabilities"]).copy()).to(device).float(),
        "offsets_mm":torch.from_numpy(np.asarray(frame["offsets_mm"]).copy()).to(device).float(),
        "native_score":torch.from_numpy(np.asarray(frame["original_native_score"]).copy()).to(device).float(),
    }


def forward_model(model,t,hw):
    return model(
        t["image_feature"],t["K"],t["actions"],t["probabilities"],
        t["offsets_mm"],t["native_score"],hw,
    )


@torch.no_grad()
def _predict_validation(model, ex, cache_frame, device, chunk):
    """Predict P(beneficial) for ALL A1 moved proposals, chunked for memory."""
    n=len(ex["delta_utility"])
    if n==0:
        return np.empty(0,np.float32)
    hw=cache_frame["depth"].shape[-2:]
    out=[]
    for start in range(0,n,chunk):
        sl=slice(start,min(start+chunk,n))
        sub={
            "actions":ex["actions"][:,sl],
            "probabilities":ex["probabilities"][:,sl],
            "offsets_mm":ex["offsets_mm"][sl],
            "original_native_score":ex["original_native_score"][sl],
        }
        t=tensor_frame(sub,cache_frame,device)
        pred=forward_model(model,t,hw)
        out.append(pred["class_logits"].softmax(-1)[:,BENEFICIAL].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def validation(models, cache_root, source_root, contract, cases, query_chunk, device, max_files=0):
    """Calibrate threshold on full-query Seen utility, not sampled proposals.

    Every A1 fixed-0 moved proposal in the source is scored. Queries where A1
    stays native remain in the denominator and contribute zero relative gain.
    Threshold selection directly maximizes verifier increment over A1 fixed-0.
    """
    for m in models.values():
        m.eval()
    files=source_files(source_root,"test_seen","joint",cases)
    if max_files>0:
        files=files[:max_files]
    records={v:[] for v in VARIANTS}
    for path in files:
        with np.load(path,allow_pickle=False) as z:
            payload={k:z[k] for k in z.files}
        ev=source_eval_path(source_root,path,"test_seen")
        if not ev.is_file():
            raise FileNotFoundError(ev)
        with np.load(ev,allow_pickle=False) as z:
            labels={k:z[k] for k in z.files}
        ex=compact_from_source(payload,labels)
        sid=int(np.asarray(payload["scene_id"]).reshape(()))
        aid=int(np.asarray(payload["anno_id"]).reshape(()))
        cf=read_frame(frame_cache_path(cache_root,"test_seen",sid,aid),contract)
        case=str(payload["case"])
        delta=np.asarray(ex["delta_utility"],np.float32)
        q_total=int(payload["actions"].shape[1])
        for v,m in models.items():
            p=_predict_validation(m,ex,cf,device,query_chunk)
            records[v].append((case,p,delta,q_total))

    result={}
    for v,rr in records.items():
        if not rr:
            raise RuntimeError(f"No validation records for {v}")
        thresholds=[]
        all_cases=sorted(set(c for c,_,_,_ in rr))
        for thr in np.linspace(0,1,101):
            by_case={}
            for case in all_cases:
                parts=[x for x in rr if x[0]==case]
                pp=np.concatenate([p for _,p,_,_ in parts]) if parts else np.empty(0,np.float32)
                dd=np.concatenate([d for _,_,d,_ in parts]) if parts else np.empty(0,np.float32)
                total_q=sum(q for *_,q in parts)
                by_case[case]=full_query_gate_metrics(pp,dd,thr,total_q)

            recoveries=[
                m.oracle_gap_recovery for m in by_case.values()
                if m.oracle_gap_recovery is not None
            ]
            macro_verified=float(np.mean([m.verified_gain for m in by_case.values()]))
            macro_a1=float(np.mean([m.a1_fixed0_gain for m in by_case.values()]))
            macro_oracle=float(np.mean([m.oracle_accept_gain for m in by_case.values()]))
            macro_increment=float(np.mean([m.verifier_increment for m in by_case.values()]))
            macro_gap=float(np.mean([m.oracle_gap for m in by_case.values()]))
            macro_recovery=float(np.mean(recoveries)) if recoveries else 0.
            macro_ret=float(np.mean([m.beneficial_retention for m in by_case.values()]))
            macro_rej=float(np.mean([m.harmful_rejection for m in by_case.values()]))
            macro_prec=float(np.mean([m.accept_precision for m in by_case.values()]))

            thresholds.append({
                "threshold":float(thr),
                "macro_full_query_utility_gain":macro_verified,
                "macro_a1_fixed0_gain":macro_a1,
                "macro_oracle_accept_gain":macro_oracle,
                "macro_verifier_increment":macro_increment,
                "macro_oracle_gap":macro_gap,
                "macro_oracle_gap_recovery":macro_recovery,
                "macro_beneficial_retention":macro_ret,
                "macro_harmful_rejection":macro_rej,
                "macro_accept_precision":macro_prec,
                "cases":{c:vars(m) for c,m in by_case.items()},
            })

        # Threshold objective: improve over A1 fixed-0 directly. Because the A1
        # baseline is threshold-independent, this is equivalent to maximizing
        # verified full-query utility, but makes the causal objective explicit.
        best=max(thresholds,key=lambda x:(
            x["macro_verifier_increment"],
            x["macro_beneficial_retention"],
            x["macro_harmful_rejection"],
            x["macro_accept_precision"],
        ))
        result[v]=(best,thresholds)
    return result

def main():
    args=parser().parse_args(); sys.argv=[sys.argv[0]]
    if args.epochs<1 or args.val_every<1 or args.reg_weight<0 or args.gain_weight<0:
        raise ValueError("Invalid Rep-C2-v2 training configuration")
    if args.val_move_limit!=0:
        raise ValueError(
            "Rep-C2-v2 full-query threshold calibration requires --val-move-limit 0. "
            "Proposal subsampling changes the validation distribution."
        )
    if args.val_query_chunk<1:
        raise ValueError("val-query-chunk must be positive")
    device=torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    train_protocol=json.loads((Path(args.train_cache_root)/"protocol.json").read_text())
    contract=train_protocol["cache_contract"]
    reader_init=load_torch(Path(args.cache_root)/"reader_init.pt")
    if reader_init["contract"]!=contract:
        raise RuntimeError("Rep-A cache and C2-v2 training cache contracts differ")
    train_files=load_train_examples(args.train_cache_root,args.max_train_frames)
    counts,cw_np,num_examples=class_weights(train_files)
    first_sid=int(train_files[0].parent.name.split("_")[-1])
    first_aid=int(train_files[0].stem.split("_")[-1])
    sample=read_frame(
        frame_cache_path(args.cache_root,"train",first_sid,first_aid),
        contract,
    )
    image_channels=int(sample["image_feature"].shape[0])
    del sample,reader_init

    val_cases=tuple(x.strip() for x in args.val_cases.split(",") if x.strip())
    source_protocol=json.loads((Path(args.validation_source_root)/"protocol.json").read_text())
    if source_protocol.get("query_limit",None) is None:
        raise RuntimeError("Unrecognized full-path validation source")
    if int(source_protocol["query_limit"])!=0:
        raise RuntimeError(
            "Rep-C2-v2 full-query threshold objective requires a validation source "
            "with query_limit=0. Use the formal full-path root."
        )
    config={k:v for k,v in vars(args).items() if k not in (
        "device","output_dir","resume","progress_every","cache_root",
        "train_cache_root","validation_source_root"
    )}
    signature=digest({
        "experiment":"Rep-C2-v2","objective_version":OBJECTIVE_VERSION,
        "config":config,"contract":contract,
        "train_protocol":train_protocol,
        "validation_source_protocol":source_protocol,
        "train_files":[(str(p.relative_to(args.train_cache_root)),p.stat().st_size,p.stat().st_mtime_ns) for p in train_files],
    })

    base_state=paired_initial_state(image_channels,args.dim,args.heads,args.dropout)
    models={
        v:RepC2V2Verifier(image_channels,args.dim,args.heads,args.dropout,v).to(device)
        for v in VARIANTS
    }
    for m in models.values(): m.load_state_dict(base_state,strict=True)
    opts={v:torch.optim.AdamW(models[v].parameters(),lr=args.lr,weight_decay=args.weight_decay) for v in VARIANTS}
    class_w=torch.from_numpy(cw_np).to(device)

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    latest=out/"checkpoint_latest.pt"
    start=0; history=[]
    best={v:(-float("inf"),-float("inf"),-float("inf"),-float("inf")) for v in VARIANTS}
    if latest.exists():
        if not args.resume: raise FileExistsError(f"{latest}; use --resume")
        ck=load_torch(latest)
        if ck["signature"]!=signature: raise RuntimeError("Rep-C2-v2 training contract changed")
        for v in VARIANTS:
            models[v].load_state_dict(ck["models"][v]); opts[v].load_state_dict(ck["optimizers"][v])
        start=ck["epoch"]+1; history=ck["history"]; best={k:tuple(x) for k,x in ck["best"].items()}
        del ck

    save_json(out/"protocol.json",{
        "experiment":"Rep-C2-v2","signature":signature,
        "objective_version":OBJECTIVE_VERSION,"cache_contract":contract,
        "train_cache_protocol":train_protocol,
        "validation_source_root":str(Path(args.validation_source_root).resolve()),
        "validation_cases":list(val_cases),
        "validation_role":"test_seen / validation_seen only; Similar/Novel never tune threshold/checkpoint",
        "threshold_objective":"maximize macro verifier increment over A1 fixed-0 using ALL Stage-1 queries",
        "checkpoint_objective":"maximize macro oracle-gap recovery, then verifier increment",
        "variants":list(VARIANTS),
        "variant_semantics":{
            "score":"A1 native/proposal score profile only",
            "rgb":"score + paired pre-enhancer RGB grasp-region evidence",
            "rgb_only":"paired pre-enhancer RGB grasp-region evidence only",
        },
        "target":"3-way harmful/equivalent/beneficial + auxiliary delta-utility regression",
        "class_counts_harm_equiv_benefit":counts.tolist(),
        "class_weights_inverse_sqrt":cw_np.tolist(),
        "num_training_examples":num_examples,
        "config":config,
    })

    with exclusive_run(out/".train.lock"):
        for epoch in range(start,args.epochs):
            for m in models.values(): m.train()
            order=list(train_files)
            random.Random(seed_for(args.seed,"c2v2-order",epoch)).shuffle(order)
            sums={v:{"total":0.,"ce":0.,"reg":0.} for v in VARIANTS}; used=0; examples=0
            for i,path in enumerate(order):
                with np.load(path,allow_pickle=False) as z: fr={k:z[k] for k in z.files}
                if str(fr["signature"])!=digest(train_protocol):
                    raise RuntimeError(f"Mixed Rep-C2-v2 mining protocol: {path}")
                n=len(fr["target_class"])
                if n==0: continue
                sid=int(fr["scene_id"]); aid=int(fr["anno_id"])
                cf=read_frame(frame_cache_path(args.cache_root,"train",sid,aid),contract)
                t=tensor_frame(fr,cf,device); hw=cf["depth"].shape[-2:]
                target=torch.from_numpy(np.asarray(fr["target_class"],np.int64)).to(device)
                delta=torch.from_numpy(np.asarray(fr["delta_utility"],np.float32)).to(device)
                sample_w=1.+args.gain_weight*delta.abs()
                for v in VARIANTS:
                    opts[v].zero_grad(set_to_none=True)
                    pred=forward_model(models[v],t,hw)
                    ce=F.cross_entropy(pred["class_logits"],target,weight=class_w,reduction="none")
                    ce=(ce*sample_w).mean()
                    reg=F.smooth_l1_loss(pred["delta"],delta,reduction="none")
                    reg=(reg*sample_w).mean()
                    loss=ce+args.reg_weight*reg
                    if not torch.isfinite(loss): raise FloatingPointError(f"{v} non-finite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(models[v].parameters(),5.,error_if_nonfinite=True)
                    opts[v].step()
                    sums[v]["total"]+=float(loss.detach()); sums[v]["ce"]+=float(ce.detach()); sums[v]["reg"]+=float(reg.detach())
                used+=1; examples+=n
                if (i+1)%max(1,args.progress_every)==0:
                    print(f"[REP-C2-v2] epoch={epoch} frames={i+1}/{len(order)} examples={examples}",flush=True)
                del fr,cf,t,target,delta

            row={"epoch":epoch,"train_used_frames":used,"train_examples":examples,
                 "train_loss":{v:{k:x/max(used,1) for k,x in sums[v].items()} for v in VARIANTS}}
            do_val=((epoch+1)%args.val_every==0 or epoch==args.epochs-1)
            if do_val:
                vr=validation(
                    models,args.cache_root,args.validation_source_root,contract,
                    val_cases,args.val_query_chunk,device,args.max_val_files
                )
                row["val"]={}
                for v,(bv,sweep) in vr.items():
                    row["val"][v]=bv
                    # Checkpoint objective: close the largest fraction of the
                    # C1 oracle accept/reject headroom. This normalizes across
                    # nominal/corruption cases with different raw A1 gains.
                    key=(
                        bv["macro_oracle_gap_recovery"],
                        bv["macro_verifier_increment"],
                        bv["macro_beneficial_retention"],
                        bv["macro_harmful_rejection"],
                    )
                    if key>best[v]:
                        best[v]=key
                        save_torch(out/f"checkpoint_{v}.pt",{
                            "experiment":"Rep-C2-v2","variant":v,
                            "model":models[v].state_dict(),"threshold":bv["threshold"],
                            "epoch":epoch,"signature":signature,"contract":contract,
                            "image_channels":image_channels,"dim":args.dim,"heads":args.heads,
                            "dropout":args.dropout,"val":bv,"config":config,
                        })
                        save_json(out/f"threshold_sweep_{v}.json",sweep)
            history.append(row)
            save_torch(latest,{
                "experiment":"Rep-C2-v2","epoch":epoch,"signature":signature,
                "models":{v:models[v].state_dict() for v in VARIANTS},
                "optimizers":{v:opts[v].state_dict() for v in VARIANTS},
                "history":history,"best":best,
            })
            save_json(out/"metrics.json",history)
            print(json.dumps(row,sort_keys=True),flush=True)
    print(f"[REP-C2-v2] finished: {out}")


if __name__=="__main__":
    main()
