#!/usr/bin/env python3
"""Evaluate Rep-C2-v2 on the formal full-path source without retuning.

The source root supplies regenerated candidates, frozen A1 fixed-0 proposals and
fresh exact labels. Rep-C2-v2 only accepts/rejects those proposals. Thresholds
come from Seen validation checkpoints.

When SOURCE_ROOT has QUERY_LIMIT=0, this script also emits complete GraspNet
.dumpy arrays for official AP. Scores are matched A1 scores: accepted corrections
use the A1 selected-hypothesis score; rejected proposals use A1 zero/native score.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from rep_a_common import atomic_file, cdf_targets, digest, exclusive_run, load_torch, read_frame, save_json
from rep_c2v2_common import (
    BENEFICIAL, VARIANTS, compact_from_source, frame_cache_path,
    official_dump_path, row_index, scorer_index, source_eval_path, source_files,
)
from rep_c2v2_model import RepC2V2Verifier
from rep_followup_common import checked_manifest


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",required=True)
    p.add_argument("--cache-root",required=True)
    p.add_argument("--c2v2-dir",required=True)
    p.add_argument("--output-root",required=True)
    p.add_argument("--split",choices=("test_seen","test_similar","test_novel"),required=True)
    p.add_argument("--cases",default="nominal,bias:-20,bias:20")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--query-chunk",type=int,default=128)
    p.add_argument("--max-files",type=int,default=0)
    p.add_argument("--overwrite-dumps",action="store_true")
    return p


def load_models(folder,device):
    models={}; meta={}
    for v in VARIANTS:
        ck=load_torch(Path(folder)/f"checkpoint_{v}.pt")
        if ck.get("experiment")!="Rep-C2-v2" or ck.get("variant")!=v:
            raise RuntimeError(f"Bad Rep-C2-v2 checkpoint for {v}")
        m=RepC2V2Verifier(
            ck["image_channels"],ck["dim"],ck["heads"],ck["dropout"],v
        ).to(device)
        m.load_state_dict(ck["model"],strict=True); m.eval()
        models[v]=m
        meta[v]={
            "threshold":float(ck["threshold"]),"epoch":int(ck["epoch"]),
            "signature":ck["signature"],"contract":ck["contract"],
        }
    contracts={x["contract"] for x in meta.values()}
    if len(contracts)!=1: raise RuntimeError("C2-v2 checkpoints use different cache contracts")
    return models,meta,next(iter(contracts))


def subset_tensor(ex,cf,ids,device):
    return {
        "image_feature":torch.from_numpy(np.asarray(cf["image_feature"]).copy()).to(device).float(),
        "K":torch.from_numpy(np.asarray(cf["K"]).copy()).to(device).float(),
        "actions":torch.from_numpy(np.asarray(ex["actions"][:,ids]).copy()).to(device).float(),
        "probabilities":torch.from_numpy(np.asarray(ex["probabilities"][:,ids]).copy()).to(device).float(),
        "offsets_mm":torch.from_numpy(np.asarray(ex["offsets_mm"][ids]).copy()).to(device).float(),
        "native_score":torch.from_numpy(np.asarray(ex["original_native_score"][ids]).copy()).to(device).float(),
    }


@torch.no_grad()
def predict(model,ex,cf,device,chunk):
    n=len(ex["delta_utility"])
    if n==0: return np.empty(0,np.float32),np.empty(0,np.float32)
    probs=[]; deltas=[]; hw=cf["depth"].shape[-2:]
    for start in range(0,n,chunk):
        ids=np.arange(start,min(start+chunk,n))
        t=subset_tensor(ex,cf,ids,device)
        out=model(
            t["image_feature"],t["K"],t["actions"],t["probabilities"],
            t["offsets_mm"],t["native_score"],hw
        )
        probs.append(out["class_logits"].softmax(-1)[:,BENEFICIAL].cpu().numpy())
        deltas.append(out["delta"].cpu().numpy())
    return np.concatenate(probs),np.concatenate(deltas)


def metrics_for_frame(payload,labels,ex,accept):
    q_total=payload["actions"].shape[1]
    delta=np.asarray(ex["delta_utility"],np.float64)
    benefit=delta>1e-7; harm=delta<-1e-7; equiv=~(benefit|harm)
    accepted_delta=np.where(accept,delta,0.)
    return {
        "num_queries":int(q_total),
        "num_proposals":int(len(delta)),
        "accept_count":int(accept.sum()),
        "beneficial_proposals":int(benefit.sum()),
        "harmful_proposals":int(harm.sum()),
        "equivalent_proposals":int(equiv.sum()),
        "utility_gain":float(accepted_delta.sum()/q_total),
        "a1_fixed0_gain":float(delta.sum()/q_total),
        "oracle_accept_gain":float(np.maximum(delta,0).sum()/q_total),
        "accept_rate":float(accept.sum()/q_total),
        "proposal_rate":float(len(delta)/q_total),
        "beneficial_retention":float((accept&benefit).sum()/max(1,benefit.sum())),
        "harmful_rejection":float(1-(accept&harm).sum()/max(1,harm.sum())),
        "accept_precision":float((accept&benefit).sum()/max(1,accept.sum())),
    }


def aggregate(rows):
    nq=sum(r["num_queries"] for r in rows)
    out={"num_frames":len(rows),"num_queries":nq}
    for k in ("num_proposals","accept_count","beneficial_proposals","harmful_proposals","equivalent_proposals"):
        out[k]=int(sum(r[k] for r in rows))
    for k in ("utility_gain","a1_fixed0_gain","oracle_accept_gain","accept_rate","proposal_rate"):
        out[k]=sum(r[k]*r["num_queries"] for r in rows)/nq
    out["beneficial_retention"]=sum(
        r["beneficial_retention"]*r["beneficial_proposals"] for r in rows
    )/max(1,out["beneficial_proposals"])
    out["harmful_rejection"]=sum(
        r["harmful_rejection"]*r["harmful_proposals"] for r in rows
    )/max(1,out["harmful_proposals"])
    out["accept_precision"]=sum(
        r["accept_precision"]*r["accept_count"] for r in rows
    )/max(1,out["accept_count"])
    return out


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    import io
    buf=io.StringIO(newline=""); w=csv.DictWriter(buf,fieldnames=keys,restval="")
    w.writeheader(); w.writerows(rows)
    with atomic_file(path) as f: f.write(buf.getvalue().encode())


def write_dump(path,array,overwrite=False):
    if path.exists() and not overwrite:
        old=np.load(path,allow_pickle=False)
        if old.shape!=array.shape or not np.allclose(old,array,atol=1e-6,rtol=0):
            raise RuntimeError(f"Existing C2-v2 dump differs: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with atomic_file(path) as f: np.save(f,array.astype(np.float32),allow_pickle=False)


def main():
    args=parser().parse_args(); sys.argv=[sys.argv[0]]
    if args.query_chunk<1: raise ValueError("query-chunk must be positive")
    device=torch.device(args.device)
    models,meta,contract=load_models(args.c2v2_dir,device)
    cases=tuple(x.strip() for x in args.cases.split(",") if x.strip())
    source_protocol=json.loads((Path(args.source_root)/"protocol.json").read_text())
    protocol=dict(source_protocol)
    protocol["rep_c2v2"]={
        "source_root":str(Path(args.source_root).resolve()),
        "source_protocol_digest":digest(source_protocol),
        "checkpoint_signatures":{v:meta[v]["signature"] for v in VARIANTS},
        "thresholds":{v:meta[v]["threshold"] for v in VARIANTS},
        "proposal":"A1 fixed_0",
        "decision":"accept proposal iff validation-calibrated P(beneficial) >= threshold",
        "score_policy":"matched A1 score: selected-hypothesis if accepted, zero/native-hypothesis if rejected",
        "evidence":"score baseline vs independent pre-enhancer RGB pair evidence",
    }
    root=Path(args.output_root); root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root/".protocol.lock",wait=True):
        checked_manifest(root,protocol)

    files=source_files(args.source_root,args.split,"joint",cases)
    if args.max_files>0: files=files[:args.max_files]
    rows=[]
    for fi,path in enumerate(files):
        with np.load(path,allow_pickle=False) as z: payload={k:z[k] for k in z.files}
        ev=source_eval_path(args.source_root,path,args.split)
        if not ev.is_file(): raise FileNotFoundError(ev)
        with np.load(ev,allow_pickle=False) as z: labels={k:z[k] for k in z.files}
        ex=compact_from_source(payload,labels)
        sid=int(np.asarray(payload["scene_id"]).reshape(()))
        aid=int(np.asarray(payload["anno_id"]).reshape(()))
        case=str(payload["case"]); mode=str(payload["mode"])
        cf=read_frame(frame_cache_path(args.cache_root,args.split,sid,aid),contract)

        # Source row/score definitions are fixed once for every verifier.
        row=row_index(payload["output_methods"],payload["output_policies"],"A1","fixed_0")
        si=scorer_index(payload["scorer_names"],"A1")
        z=int(payload["zero_index"]); q_all=np.arange(payload["actions"].shape[1])
        proposal_all=np.asarray(payload["selected"][row],np.int64)
        zero_score=np.asarray(payload["probabilities"][si,z],np.float32).mean(-1)
        proposal_score=np.asarray(payload["rank_scores"][row],np.float32)

        for v,m in models.items():
            pbenef,pdelta=predict(m,ex,cf,device,args.query_chunk)
            accept=pbenef>=meta[v]["threshold"]
            selected=np.full(len(q_all),z,np.int64)
            score=zero_score.copy()
            if len(accept):
                pos=np.asarray(ex["query_pos"],np.int64)
                take=pos[accept]
                selected[take]=proposal_all[take]
                score[take]=proposal_score[take]
            rec={
                "variant":v,"split":args.split,"case":case,"mode":mode,
                "scene_id":sid,"anno_id":aid,"threshold":meta[v]["threshold"],
                "mean_pred_delta":float(pdelta.mean()) if len(pdelta) else 0.,
                **metrics_for_frame(payload,labels,ex,accept),
            }
            rows.append(rec)

            # Full-query source can be evaluated officially. Pilot roots still
            # get useful exact metrics, but are not mislabeled as official AP.
            if int(source_protocol.get("query_limit",-1))==0:
                out=payload["actions"][selected,q_all].copy()
                out[:,0]=score
                method=f"C2v2_{v}"
                dest=official_dump_path(
                    root,method,mode,case,args.split,
                    source_protocol["camera"],sid,aid
                )
                write_dump(dest,out,args.overwrite_dumps)

        if (fi+1)%50==0:
            print(f"[REP-C2-v2 TEST] {args.split} {fi+1}/{len(files)}",flush=True)

    split_dir=root/"test"/args.split; split_dir.mkdir(parents=True,exist_ok=True)
    write_csv(split_dir/"per_frame.csv",rows)
    summary=[]
    for v in VARIANTS:
        for case in cases:
            rr=[r for r in rows if r["variant"]==v and r["case"]==case]
            if rr:
                summary.append({
                    "variant":v,"split":args.split,"case":case,
                    "threshold":meta[v]["threshold"],**aggregate(rr)
                })
    write_csv(split_dir/"comparison.csv",summary)
    save_json(split_dir/"summary.json",{
        "split":args.split,"source_query_limit":source_protocol.get("query_limit"),
        "formal_dumps_emitted":int(source_protocol.get("query_limit",-1))==0,
        "results":summary,
    })
    print(f"[REP-C2-v2 TEST] wrote {split_dir/'comparison.csv'}")


if __name__=="__main__":
    main()
