"""P0/P1: compare frozen GVAR feature decodability and decision-aware scoring.

P0 input modes: base_geometry, score64, reader256, reader_delta, score_reader.
P1 objectives: mse, rare, decision. All use identical train/validation scene
splits and the SAME saved exact executed-action labels. Test scenes NEVER fit.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import torch

from rare_grasp.pipeline import (arrays, atomic_npz, load_index, protocol_rows,
                                 check_model_matches_manifest)
from rare_grasp.core import (candidate_metrics,digest,frame_seed,physical_ids,
                             select_grasps,sha256,write_json)
from diagnostics.action_decision.feature_cache import (gather_reader,
                                                       load_feature_index)
from diagnostics.action_decision.model import (MODES,OBJECTIVES,ActionQualityProbe,
                                              make_input,quality_losses)

SCHEMA="action-decision-probe-v1"


def load_training_entries(source_index, feature_index, *, smoke=False):
    rare=load_index(source_index,"train",allow_partial=smoke)
    added=load_feature_index(feature_index,allow_partial=smoke)
    if added["source_train_index_sha256"]!=rare["index_sha256"]:
        raise RuntimeError("GVAR features and quality labels belong to different cache indices")
    if added["base_contract"]["config_digest"]!=rare["contract"]["config_digest"]:
        raise RuntimeError("Frozen GVAR checkpoint identity mismatch")
    lookup={}
    for x in added["entries"]:
        k=(x["row"]["domain"],int(x["row"]["scene"]),int(x["row"]["anno"]))
        if k in lookup:raise RuntimeError("Duplicate reader feature frame")
        lookup[k]=x
    combined=[]
    for x in rare["entries"]:
        r=x["row"];k=(r["domain"],int(r["scene"]),int(r["anno"]))
        y=lookup.get(k)
        if y is None:
            if smoke:continue
            raise RuntimeError(f"Missing reader feature frame: {k}")
        if y["source"]!=x:
            raise RuntimeError("Reader/label cache index entry changed")
        combined.append(dict(row=r,rare=x,reader=y))
    if not combined:raise RuntimeError("No matching frames")
    return rare,added,combined


def read_one(e, device):
    src=arrays(e["rare"])
    path=Path(e["reader"]["reader_file"])
    if sha256(path)!=e["reader"]["reader_sha256"]:
        raise RuntimeError(f"Reader feature SHA changed: {path}")
    with np.load(path,allow_pickle=False) as f:added={k:f[k] for k in f.files}
    return make_input(src,added,device)


def _scene_mean(records,metric):
    groups=defaultdict(list)
    for r in records:
        if r.get(metric) is not None:
            groups[(r["domain"],r["scene"])].append(r[metric])
    domains=sorted({k[0] for k in groups})
    if not domains:return None
    return float(np.mean([np.mean([np.mean(v) for (d,_),v in groups.items()
                                   if d==domain]) for domain in domains]))


def validate(model,entries,device):
    records=[]
    model.eval()
    with torch.inference_mode():
        for e in entries:
            data=read_one(e,device)
            z=model(data);s=z.sigmoid().detach().cpu().numpy()
            cnt=data["count"].cpu().numpy()
            families=np.asarray(data["family"].cpu())
            unique=np.unique(families)
            if len(unique)*4!=len(cnt):raise RuntimeError("Incomplete depth families")
            scores=s.reshape(-1,4);quality=cnt.reshape(-1,4)
            info=candidate_metrics(scores,quality)
            info["mse"]=float(np.mean((s-cnt/6.0)**2))
            info["top1_quality"]=float(quality[np.arange(len(unique)),scores.argmax(1)]
                                        [np.argmax(scores.max(1))]/6.0)
            info.update(domain=e["row"]["domain"],scene=int(e["row"]["scene"]),
                        anno=int(e["row"]["anno"]))
            records.append(info)
    keys=("selected_q","oracle_q","regret","oracle_at2","depth_pair_agreement",
          "mse","top1_quality")
    return {k:_scene_mean(records,k) for k in keys},records


def fit(a):
    rare,added,entries=load_training_entries(a.index,a.feature_index,smoke=a.smoke)
    if not a.smoke and (len(entries)!=25600 or
                        rare["contract"]["config"].get("mix_training_fraction")!=0.5):
        raise ValueError("Formal P0/P1 uses complete MIX50 exact-action caches")
    train=[e for e in entries if int(e["row"]["scene"])<90]
    val=[e for e in entries if 90<=int(e["row"]["scene"])<100]
    if a.smoke:
        train=train[:a.limit_train or 4];val=val[:a.limit_val or 2]
    else:
        expected_train,expected_val=23040,2560
        if len(train)!=expected_train or len(val)!=expected_val:
            raise RuntimeError("Train=90 scenes, validation=10 distinct scenes per domain required")
        for domain in ("original","gntrans"):
            if (sum(e["row"]["domain"]==domain for e in train)!=11520 or
                sum(e["row"]["domain"]==domain for e in val)!=1280):
                raise ValueError("Per-domain scene-balanced split is broken")
    if not train or not val:raise ValueError("No scene-held-out training or validation data")
    if a.epochs<1 or a.batch_frames<1 or a.lr<=0 or a.pair_budget<1 or a.hidden<1:
        raise ValueError("Invalid scorer settings")
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Use a new run directory")
    out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    dev=torch.device(a.device)
    model=ActionQualityProbe(a.feature_mode,a.hidden).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    rng=np.random.default_rng(a.seed)
    arguments={k:v for k,v in vars(a).items() if k!="func"}
    contract=dict(schema=SCHEMA,train_source_sha256=rare["index_sha256"],
                  reader_index_sha256=added["index_sha256"],
                  bound_config_digest=rare["contract"]["config_digest"],
                  base_checkpoint_sha256=added["base_contract"]["base_sha256"],
                  scene_split="train_0000-0089, validation_0090-0099; both domains",
                  base_seen_validation_scenes=True,
                  training=arguments,weight_selection="fixed last epoch",
                  training_labels="existing actual executed grasp six-mu CAD labels",
                  parameter_count=sum(x.numel() for x in model.parameters()))
    write_json(out/"protocol.json",contract)
    hist=[];updates=0
    def save(epoch):
        p=out/"last.pt"
        tmp=out/"last.pt.tmp"
        torch.save(dict(schema=SCHEMA,epoch=epoch,feature_mode=a.feature_mode,
                        objective=a.objective,hidden=a.hidden,
                        state_dict=model.state_dict(),protocol=contract,
                        optimizer=opt.state_dict(),updates=updates,
                        np_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),tmp)
        os.replace(tmp,p)
    metrics,_=validate(model,val,dev)
    hist.append(dict(epoch=0,updates=updates,validation=metrics))
    save(0)
    for epoch in range(1,a.epochs+1):
        model.train()
        order=rng.permutation(len(train))
        stats=[]
        for start in range(0,len(order),a.batch_frames):
            opt.zero_grad(set_to_none=True);losses=[]
            for j in order[start:start+a.batch_frames]:
                e=train[int(j)];data=read_one(e,dev)
                logits=model(data)
                pair_seed=frame_seed(e["row"],a.seed+epoch*100003)
                mse,pair,info=quality_losses(logits,data,a.objective,
                                              a.pair_budget,pair_seed,a.top_k)
                loss=10.0*mse+(a.pair_weight*pair if a.objective!="mse" else 0.0)
                losses.append(loss)
                stats.append((float(mse.detach()),float(pair.detach()),
                              info["used"],info["conflict"]))
            total=torch.stack(losses).mean()
            if not torch.isfinite(total):raise FloatingPointError("Nonfinite scorer loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            opt.step();updates+=1
        metrics,_=validate(model,val,dev)
        arr=np.asarray(stats,dtype=float)
        hist.append(dict(epoch=epoch,updates=updates,
                         train_mean_mse=float(arr[:,0].mean()),
                         train_mean_pair=float(arr[:,1].mean()),
                         train_mean_used_pairs=float(arr[:,2].mean()),
                         train_mean_conflict_pairs=float(arr[:,3].mean()),
                         validation=metrics))
        save(epoch);write_json(out/"history.json",hist)
        print(f"[P0/P1 {a.feature_mode}/{a.objective}] "
              f"epoch={epoch}/{a.epochs}, validation={metrics}",flush=True)
    write_json(out/"complete.json",dict(status="complete",smoke_only=a.smoke,
               checkpoint_sha256=sha256(out/"last.pt"),updates=updates,
               protocol_sha256=digest(contract)))


def _capture_full(runtime,row,seed):
    reader=runtime.model.kview_grasp_module.decoder.reader
    if reader is None:raise RuntimeError("GVAR reader unavailable")
    saved={}
    def pre(_m,args):
        if saved:raise RuntimeError("Repeated reader hook")
        saved["before"]=args[0].detach()
    def post(_m,args,y):
        saved["after"]=y.detach()
    hpre=reader.register_forward_pre_hook(pre)
    hpost=reader.register_forward_hook(post)
    try:
        snap=runtime.snapshot(row,seed)
        if set(saved)!={"before","after"}:raise RuntimeError("Missing full GVAR latent")
        q=np.repeat(np.arange(1024,dtype=np.int64),4)
        d=np.tile(np.arange(4,dtype=np.int64),1024)
        sample=dict(family=q,depth=d,
                    native_angle=np.repeat(snap["angles"],4),
                    features=snap["features"].reshape(-1,64),
                    base_logit=snap["base_logit"].reshape(-1),
                    grasps=snap["grasps"].reshape(-1,17))
        selected=gather_reader((saved["before"],saved["after"]),snap,sample)
        return snap,sample,selected
    finally:
        hpost.remove();hpre.remove();saved.clear()


def inference(a):
    from rare_grasp.runtime import Runtime
    path=Path(a.head)
    if not path.is_file():raise FileNotFoundError(path)
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint.get("schema")!=SCHEMA:
        raise ValueError("Not a P0/P1 action quality checkpoint")
    tr=checkpoint["protocol"]["training"]
    if checkpoint["epoch"]!=tr["epochs"] or tr["smoke"]:
        raise ValueError("Only completed fixed-final non-smoke heads are formally evaluable")
    p,rows=protocol_rows(a.protocol,"test",a.domain,a.split)
    if a.limit:rows=rows[:a.limit]
    rt=Runtime(a.config,a.dataset_root,a.gntrans_root,a.device)
    check_model_matches_manifest(rt.config,p)
    if rt.config_digest!=checkpoint["protocol"]["bound_config_digest"]:
        raise ValueError("Head and live model checkpoint/config differ")
    if rt.config["checkpoint_sha256"]!=checkpoint["protocol"]["base_checkpoint_sha256"]:
        raise ValueError("Frozen base checkpoint SHA mismatch")
    head=ActionQualityProbe(checkpoint["feature_mode"],checkpoint["hidden"]).to(rt.device)
    head.load_state_dict(checkpoint["state_dict"],strict=True)
    head.eval()
    policies=a.policies.split(",")
    if not policies or any(x not in ("native","both","depth_only","rank_only") for x in policies):
        raise ValueError("Unknown inference policy")
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use new inference directory")
    root.mkdir(parents=True,exist_ok=True)
    records=[]
    for n,row in enumerate(rows):
        snap,source,added=_capture_full(rt,row,a.seed)
        inp=make_input(source,added,rt.device)
        with torch.inference_mode():
            scores=head(inp).sigmoid().cpu().numpy().reshape(1024,4)
        bank=snap["grasps"]
        for policy in policies:
            action=select_grasps(bank,scores,policy)
            before=len(action)
            if not a.no_collision:
                action=rt.sensor_collision_filter(row,action)
            file=root/policy/f"scene_{row['scene']:04d}"/"realsense"/f"{row['anno']:04d}.npy"
            file.parent.mkdir(parents=True,exist_ok=True)
            np.save(file,action)
            records.append(dict(row=row,policy=policy,path=str(file.relative_to(root)),
                                prefilter_count=before,postfilter_count=len(action),
                                sha256=sha256(file)))
        print(f"[P0/P1 inference] {n+1}/{len(rows)} {row['scene']}/{row['anno']}",flush=True)
    write_json(root/"inference_manifest.json",dict(
        schema=SCHEMA,status="complete",frames=len(rows),smoke_only=bool(a.limit),
        config_digest=rt.config_digest,base_sha256=rt.config["checkpoint_sha256"],
        ranker_sha256=sha256(path),domain=a.domain,split=a.split,seed=a.seed,
        policies=policies,records=records,
        angle="frozen_native_winner",width="native_per_depth_frozen",
        collision="none" if a.no_collision else "original_sensor:.01voxel,.01threshold,.05approach",
        scorer=checkpoint["feature_mode"],objective=checkpoint["objective"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="cmd",required=True)
    t=sp.add_parser("fit")
    t.add_argument("--index",required=True)
    t.add_argument("--feature-index",required=True)
    t.add_argument("--output",required=True)
    t.add_argument("--feature-mode",required=True,choices=MODES)
    t.add_argument("--objective",required=True,choices=OBJECTIVES)
    t.add_argument("--pair-budget",type=int,default=2048)
    t.add_argument("--top-k",type=int,default=16)
    t.add_argument("--pair-weight",type=float,default=.5)
    t.add_argument("--epochs",type=int,default=20)
    t.add_argument("--batch-frames",type=int,default=4)
    t.add_argument("--hidden",type=int,default=64)
    t.add_argument("--lr",type=float,default=1e-4)
    t.add_argument("--weight-decay",type=float,default=1e-4)
    t.add_argument("--seed",type=int,default=0)
    t.add_argument("--device",default="cuda:0")
    t.add_argument("--smoke",action="store_true")
    t.add_argument("--limit-train",type=int,default=0)
    t.add_argument("--limit-val",type=int,default=0)
    t.set_defaults(func=fit)
    i=sp.add_parser("infer")
    for name in ("config","protocol","dataset-root","head","output"):
        i.add_argument("--"+name,required=True)
    i.add_argument("--gntrans-root",default="")
    i.add_argument("--domain",choices=("original","gntrans"),default="original")
    i.add_argument("--split",choices=("test_seen","test_similar","test_novel"),required=True)
    i.add_argument("--device",default="cuda:0")
    i.add_argument("--policies",default="both,depth_only,rank_only")
    i.add_argument("--seed",type=int,default=0)
    i.add_argument("--limit",type=int,default=0)
    i.add_argument("--no-collision",action="store_true")
    i.set_defaults(func=inference)
    a=p.parse_args();a.func(a)


if __name__=="__main__":
    main()
