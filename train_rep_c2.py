#!/usr/bin/env python3
"""Train Rep-C2 evidence verifiers on top of frozen A1 proposals.

All three evidence variants see the same A1 proposals and exact-action labels.
They use identical 160-D MLPs; unavailable evidence blocks are zeroed.

Training target: whether the A1 fixed-0 proposed correction has strictly higher
exact utility than native. The verifier is only trained on queries where A1
actually proposes a move. Validation chooses an acceptance threshold on Seen
using nominal + in-support structured depth perturbations. Held-out thresholds
are never retuned.
"""
from __future__ import annotations

import argparse, json, sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from rep_a_common import (
    check_runtime_sources, choose, digest, exclusive_run, list_frames,
    read_frame, save_json, save_torch, seed_for, tensors,
    training_case, perturb_depth, file_sha,
)
from rep_followup_common import load_scorer
from rep_c2_model import EVIDENCE, RepC2Verifier, apply_evidence, build_pair_evidence


VAL_CASES = (
    "nominal","bias:-10","bias:10","bias:-20","bias:20",
    "scale:-0.03","scale:0.03","smooth:10",
)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root",required=True)
    p.add_argument("--a1-checkpoint",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-train-frames",type=int,default=0)
    p.add_argument("--max-val-frames",type=int,default=0)
    p.add_argument("--max-bias-mm",type=float,default=20.)
    p.add_argument("--max-scale",type=float,default=.03)
    p.add_argument("--smooth-mm",type=float,default=10.)
    p.add_argument("--nominal-prob",type=float,default=.25)
    p.add_argument("--nominal-val-weight",type=float,default=2.)
    p.add_argument("--gain-weight",type=float,default=6.)
    p.add_argument("--progress-every",type=int,default=100)
    p.add_argument("--resume",action="store_true")
    return p


def gate_stats(records, threshold, nominal_weight=2.):
    total_q=0; utility_sum=0.; weighted_obj=0.; obj_weight=0.
    accepted=benefit_acc=harm_acc=benefit_total=harm_total=0
    nominal_q=0; nominal_gain=0.
    for r in records:
        p=r["gate"]; proposal=r["proposal"]; z=r["zero"]; u=r["utility"]; valid_move=r["move"]
        q=np.arange(len(proposal))
        accept=valid_move & (p >= threshold)
        selected=np.where(accept,proposal,z)
        gain=u[selected,q]-u[z]
        n=len(q); total_q+=n; utility_sum+=float(u[selected,q].sum())
        w=nominal_weight if r["case"]=="nominal" else 1.
        weighted_obj += w*float(u[selected,q].mean()); obj_weight += w
        true_gain=u[proposal,q]-u[z]
        benefit=valid_move & (true_gain>1e-7)
        harm=valid_move & (true_gain<-1e-7)
        accepted += int(accept.sum())
        benefit_acc += int((accept&benefit).sum()); harm_acc += int((accept&harm).sum())
        benefit_total += int(benefit.sum()); harm_total += int(harm.sum())
        if r["case"]=="nominal":
            nominal_q += n; nominal_gain += float(gain.sum())
    return {
        "threshold":float(threshold),
        "objective":weighted_obj/max(obj_weight,1e-9),
        "selected_utility":utility_sum/max(total_q,1),
        "accept_rate":accepted/max(total_q,1),
        "beneficial_retention":benefit_acc/max(benefit_total,1),
        "harmful_accept_rate":harm_acc/max(harm_total,1),
        "harmful_rejection":1-harm_acc/max(harm_total,1),
        "accept_precision":benefit_acc/max(accepted,1),
        "nominal_utility_gain":nominal_gain/max(nominal_q,1),
        "num_queries":total_q,
        "beneficial_proposals":benefit_total,
        "harmful_proposals":harm_total,
    }


def tune_threshold(records, nominal_weight):
    sweep=[gate_stats(records,t,nominal_weight) for t in np.linspace(0,1,51)]
    best=max(sweep,key=lambda x:(x["objective"],x["nominal_utility_gain"],-x["harmful_accept_rate"],-x["accept_rate"]))
    return best,sweep


@torch.no_grad()
def collect_validation(a1, verifier, evidence, paths, contract, device, seed):
    records=[]
    verifier.eval()
    for path in paths:
        d=read_frame(path,contract)
        t=tensors(d,device)
        t["valid"]=torch.as_tensor(d["valid"],device=device,dtype=torch.bool)
        t["offsets_mm"]=torch.as_tensor(d["offsets_mm"],device=device,dtype=torch.float32)
        t["zero_index"]=int(d["zero_index"])
        t["native_score"]=torch.as_tensor(d["native_score"],device=device,dtype=torch.float32)
        for case in VAL_CASES:
            s=seed_for(seed,"c2-val",int(d["scene_id"]),int(d["anno_id"]),case)
            dep,_=perturb_depth(t["depth"],case,s)
            prob=a1(t,dep).sigmoid()
            pu=prob.mean(-1).cpu().numpy()
            proposal_np=choose(pu,d["valid"],int(d["zero_index"]),0.)
            proposal=torch.as_tensor(proposal_np,device=device)
            feat=build_pair_evidence(t,dep,prob,proposal)
            gate=verifier(apply_evidence(feat,evidence)).sigmoid().cpu().numpy()
            records.append({
                "case":case,"gate":gate,"proposal":proposal_np,
                "zero":int(d["zero_index"]),"utility":d["utility"],"move":proposal_np!=int(d["zero_index"])
            })
        del d,t
    return records


def main():
    args=parser().parse_args()
    # Project model modules import utils.arguments, whose legacy parser runs
    # at import time. Hide Rep-C2 CLI flags before load_scorer() lazily imports
    # Rep-A/CVA modules; this matches train_rep_a.py.
    sys.argv=[sys.argv[0]]
    if args.epochs<1 or args.nominal_val_weight<=0 or args.gain_weight<0:
        raise ValueError("Invalid Rep-C2 configuration")
    check_runtime_sources(args.cache_root)
    device=torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    a1,meta=load_scorer(args.a1_checkpoint,args.cache_root,device)
    if meta["variant"]!="A1":
        raise ValueError("Rep-C2 requires an A1 checkpoint")
    a1.requires_grad_(False)

    train_paths=list_frames(args.cache_root,"train",args.max_train_frames)
    val_paths=list_frames(args.cache_root,"test_seen",args.max_val_frames)
    contract=meta["contract"]
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(args).items() if k not in ("device","output_dir","resume","progress_every","cache_root","a1_checkpoint")}
    signature=digest({
        "experiment":"Rep-C2","config":config,"contract":contract,
        "a1_sha":file_sha(args.a1_checkpoint),
        "train":[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in train_paths],
        "val":[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in val_paths],
    })

    # Identical architecture and initialization across evidence variants.
    base=RepC2Verifier().to(device)
    base_state={k:v.detach().clone() for k,v in base.state_dict().items()}
    models={e:RepC2Verifier().to(device) for e in EVIDENCE}
    for m in models.values(): m.load_state_dict(base_state,strict=True)
    opts={e:torch.optim.AdamW(models[e].parameters(),lr=args.lr,weight_decay=args.weight_decay) for e in EVIDENCE}

    latest=out/"checkpoint_latest.pt"
    start=0; history=[]; best={e:(-float("inf"),-float("inf")) for e in EVIDENCE}
    if latest.exists():
        if not args.resume: raise FileExistsError(f"{latest}; use --resume")
        ck=torch.load(latest,map_location="cpu",weights_only=False)
        if ck["signature"]!=signature: raise RuntimeError("Rep-C2 training contract changed")
        for e in EVIDENCE:
            models[e].load_state_dict(ck["models"][e]); opts[e].load_state_dict(ck["optimizers"][e])
        start=ck["epoch"]+1; history=ck["history"]; best={k:tuple(v) for k,v in ck["best"].items()}
        del ck

    save_json(out/"protocol.json",{
        "experiment":"Rep-C2","signature":signature,"cache_contract":contract,
        "a1_checkpoint":str(Path(args.a1_checkpoint).resolve()),"config":config,
        "evidence_variants":list(EVIDENCE),
        "proposal":"A1 fixed-0 candidate; verifier only decides accept vs native",
        "target":"strict exact-utility improvement of A1 proposal over native",
        "validation_cases":list(VAL_CASES),
        "equal_classifier_capacity":"all variants use the same 160-D MLP; absent evidence blocks are zero",
        "no_privileged_input":"features use A1 scores, predicted depth, gripper geometry, visibility/gradient/roughness only",
    })

    with exclusive_run(out/".train.lock"):
        for epoch in range(start,args.epochs):
            for m in models.values(): m.train()
            order=list(train_paths); random.Random(seed_for(args.seed,"c2-order",epoch)).shuffle(order)
            loss_sum={e:0. for e in EVIDENCE}; used=0
            for i,path in enumerate(order):
                d=read_frame(path,contract)
                t=tensors(d,device)
                t["valid"]=torch.as_tensor(d["valid"],device=device,dtype=torch.bool)
                t["offsets_mm"]=torch.as_tensor(d["offsets_mm"],device=device,dtype=torch.float32)
                t["zero_index"]=int(d["zero_index"])
                t["native_score"]=torch.as_tensor(d["native_score"],device=device,dtype=torch.float32)
                s=seed_for(args.seed,"c2-train",epoch,int(d["scene_id"]),int(d["anno_id"]))
                case=training_case(s,args.max_bias_mm,args.max_scale,args.smooth_mm,args.nominal_prob)
                dep,_=perturb_depth(t["depth"],case,s)
                with torch.no_grad():
                    prob=a1(t,dep).sigmoid()
                    proposal_np=choose(prob.mean(-1).cpu().numpy(),d["valid"],int(d["zero_index"]),0.)
                    proposal=torch.as_tensor(proposal_np,device=device)
                    feat=build_pair_evidence(t,dep,prob,proposal)
                    q=np.arange(len(proposal_np)); z=int(d["zero_index"])
                    gain=d["utility"][proposal_np,q]-d["utility"][z]
                    move=proposal_np!=z
                if move.any():
                    ids=torch.as_tensor(np.flatnonzero(move),device=device)
                    target=torch.as_tensor((gain[move]>1e-7).astype(np.float32),device=device)
                    weight=torch.as_tensor(1.+args.gain_weight*np.abs(gain[move]),device=device,dtype=torch.float32)
                    for e in EVIDENCE:
                        opts[e].zero_grad(set_to_none=True)
                        logits=models[e](apply_evidence(feat[ids],e))
                        loss=(F.binary_cross_entropy_with_logits(logits,target,reduction="none")*weight).mean()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(models[e].parameters(),5.,error_if_nonfinite=True)
                        opts[e].step()
                        loss_sum[e]+=float(loss.detach())
                    used+=1
                if (i+1)%max(1,args.progress_every)==0:
                    print(f"[REP-C2] epoch={epoch} frames={i+1}/{len(order)} used={used}",flush=True)
                del d,t,dep,prob,feat
            epoch_row={"epoch":epoch,"train_used_frames":used,"train_loss":{e:loss_sum[e]/max(used,1) for e in EVIDENCE},"val":{}}
            for e in EVIDENCE:
                records=collect_validation(a1,models[e],e,val_paths,contract,device,args.seed)
                val,sweep=tune_threshold(records,args.nominal_val_weight)
                epoch_row["val"][e]=val
                key=(val["objective"],val["nominal_utility_gain"])
                if key>best[e]:
                    best[e]=key
                    save_torch(out/f"checkpoint_{e}.pt",{
                        "experiment":"Rep-C2","evidence":e,"model":models[e].state_dict(),
                        "threshold":val["threshold"],"epoch":epoch,"signature":signature,
                        "contract":contract,"a1_checkpoint":str(Path(args.a1_checkpoint).resolve()),
                        "a1_sha":file_sha(args.a1_checkpoint),"config":config,"val":val,
                    })
                    save_json(out/f"threshold_sweep_{e}.json",sweep)
            history.append(epoch_row)
            payload={
                "experiment":"Rep-C2","epoch":epoch,"signature":signature,
                "models":{e:models[e].state_dict() for e in EVIDENCE},
                "optimizers":{e:opts[e].state_dict() for e in EVIDENCE},
                "best":best,"history":history,
            }
            save_torch(latest,payload); save_json(out/"metrics.json",history)
            print(json.dumps(epoch_row,sort_keys=True),flush=True)
    print(f"[REP-C2] finished: {out}")


if __name__=="__main__":
    main()
