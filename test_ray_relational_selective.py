#!/usr/bin/env python3
"""Held-out exact-action test for Cross-Center Relational Selective Correction.

The Stage-1 RGB grasp network is frozen. For every selected native image-FPS
query, the same K=7 camera-ray centers are reread coherently. The relational
selector jointly observes all K frozen center features, first decides whether to
leave native, then conditionally ranks alternative centers. CAD/DexNet labels are
used only after selection for diagnosis.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root",required=True)
    p.add_argument("--checkpoint_path",required=True,help="Frozen Stage-1 checkpoint")
    p.add_argument("--selector_checkpoint",required=True)
    p.add_argument("--output_dir",required=True)
    p.add_argument("--split",default="test_similar",choices=("test_seen","test_similar","test_novel"))
    p.add_argument("--camera",default="realsense")
    p.add_argument("--sample_interval",type=float,default=0.1)
    p.add_argument("--max_samples",type=int,default=0)
    p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--num_points",type=int,default=20000)
    p.add_argument("--min_depth",type=float,default=0.2)
    p.add_argument("--max_depth",type=float,default=1.0)
    p.add_argument("--bin_num",type=int,default=256)
    p.add_argument("--pose_depth_mode",default="global_film",choices=("none","global_film","ray_gravity_film"))
    p.add_argument("--offsets_mm",default="-40,-20,-10,0,10,20,40")
    p.add_argument("--query_eval_num",type=int,default=128)
    p.add_argument("--query_eval_mode",default="topk_uniform",choices=("all","topk","uniform","topk_uniform"))
    p.add_argument("--move_threshold",type=float,default=None)
    p.add_argument("--fc_mode",default="reuse_contacts",choices=("reuse_contacts","official"))
    p.add_argument("--verify_n",type=int,default=0)
    p.add_argument("--noop_check_samples",type=int,default=2)
    p.add_argument("--noop_atol",type=float,default=5e-5)
    p.add_argument("--profile_timing",action="store_true")
    p.add_argument("--save_candidate_rows",action="store_true")
    return p.parse_args()


ARGS=parse_args()
sys.argv=[sys.argv[0]]

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
from utils.arguments import cfgs
from utils.cva_center_decoupling import assert_native_reread_equivalent, rerun_cdf_with_read_center
from utils.ray_bestofk_diagnostic import (
    build_ray_center_hypotheses,
    friction_utility,
    gather_kn,
    parse_offsets_mm,
    select_exact_oracle,
    select_raw_score,
)
from utils.ray_pairwise_selector import extract_action_conditioned_features
from utils.ray_relational_selective import (
    CrossCenterRelationalSelective,
    compose_relational_tokens,
    select_relational_correction,
)


def configure_cfg():
    cfgs.use_top4_view_infer=False
    cfgs.kview_mode="A1"
    cfgs.kview_k=1
    cfgs.use_cdf=True
    cfgs.use_obs_depth=False
    cfgs.pose_depth_mode=ARGS.pose_depth_mode


def load_stage1(model,path):
    ckpt=torch.load(path,map_location="cpu")
    state=ckpt["model_state_dict"] if isinstance(ckpt,dict) and "model_state_dict" in ckpt else ckpt
    if not isinstance(state,Mapping):
        raise TypeError("Checkpoint does not contain a state dict.")
    result=model.load_state_dict(state,strict=False)
    optional=("rgb_geometry_diagnostics.",)
    missing=[k for k in result.missing_keys if not k.startswith(optional)]
    unexpected=[k for k in result.unexpected_keys if not k.startswith(optional)]
    if missing or unexpected:
        raise RuntimeError(f"Stage-1 checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def subset_indices(total,interval,max_samples):
    stride=max(1,int(round(1.0/interval)))
    out=[]
    for start in range(0,total,256):
        out.extend(range(start,min(start+256,total),stride))
    return out[:max_samples] if max_samples>0 else out


def move_batch(batch,device):
    for k,v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k]=v.to(device,non_blocking=False)
        elif isinstance(v,(list,tuple)):
            raise TypeError(f"Unexpected list-valued key {k!r}.")
    return batch


def uniform_pick(indices,count):
    if count<=0 or count>=len(indices): return indices
    pos=torch.round(torch.linspace(0,len(indices)-1,steps=count,device=indices.device)).long()
    pos=torch.unique(pos,sorted=True)
    if len(pos)<count:
        used=torch.zeros(len(indices),dtype=torch.bool,device=indices.device); used[pos]=True
        fill=torch.nonzero(~used,as_tuple=False).squeeze(1)[:count-len(pos)]
        pos=torch.sort(torch.cat((pos,fill))).values
    return indices.index_select(0,pos[:count])


def select_queries(native,count,mode):
    n=native.shape[0]; ids=torch.arange(n,device=native.device)
    if count<=0 or count>=n or mode=="all": return ids
    score=native[:,0]
    if mode=="topk": return torch.argsort(score,descending=True,stable=True)[:count]
    if mode=="uniform": return uniform_pick(ids,count)
    n_top=count//2
    ranked=torch.argsort(score,descending=True,stable=True)
    return torch.cat((ranked[:n_top],uniform_pick(ranked[n_top:],count-n_top)))


def success(f,threshold):
    f=np.asarray(f,dtype=np.float32)
    return np.isfinite(f)&(f>0.0)&(f<=threshold+1e-6)


def evaluate_grid(evaluator,scene_id,anno_id,grasps,valid):
    K,N,_=grasps.shape
    flat=grasps.reshape(K*N,17); vf=valid.reshape(K*N); ids=np.flatnonzero(vf)
    t0=time.perf_counter(); r=evaluator.evaluate(scene_id,anno_id,flat[ids]); elapsed=time.perf_counter()-t0
    friction=np.full(K*N,np.nan,np.float32)
    assigned=np.full(K*N,-1,np.int64)
    coll=np.full(K*N,-1,np.int8)
    pure=np.full(K*N,-1,np.int8)
    empty=np.full(K*N,-1,np.int8)
    friction[ids]=r.friction; assigned[ids]=r.assigned_obj
    coll[ids]=r.collision_or_empty.astype(np.int8)
    pure[ids]=r.pure_collision.astype(np.int8); empty[ids]=r.empty.astype(np.int8)
    return {
        "friction":friction.reshape(K,N),
        "assigned_obj":assigned.reshape(K,N),
        "collision_or_empty":coll.reshape(K,N),
        "pure_collision":pure.reshape(K,N),
        "empty":empty.reshape(K,N),
    },{
        "eval_sec":elapsed,
        "collision_sec":float(r.stats.get("collision_sec",0.0)),
        "force_closure_sec":float(r.stats.get("force_closure_sec",0.0)),
    }


def selected_metrics(mats,k):
    f=gather_kn(mats["friction"],k)
    return {
        "friction":f,
        "utility":friction_utility(f),
        "success04":success(f,0.4),
        "success08":success(f,0.8),
        "collision_or_empty":gather_kn(mats["collision_or_empty"],k),
        "pure_collision":gather_kn(mats["pure_collision"],k),
        "empty":gather_kn(mats["empty"],k),
        "assigned_obj":gather_kn(mats["assigned_obj"],k),
    }


def write_csv(path,rows):
    if not rows: return
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    configure_cfg()
    offsets=parse_offsets_mm(ARGS.offsets_mm)
    zero=[i for i,x in enumerate(offsets) if abs(x)<1e-9]
    if len(zero)!=1: raise ValueError("Exactly one zero offset required.")
    zero_k=zero[0]
    out=Path(ARGS.output_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt=torch.load(ARGS.selector_checkpoint,map_location=device)
    if ckpt.get("selector_type")!="cross_center_relational_selective_v1":
        raise RuntimeError(f"Unexpected selector_type={ckpt.get('selector_type')!r}")
    selector=CrossCenterRelationalSelective(
        token_dim=int(ckpt["token_dim"]),
        d_model=int(ckpt["d_model"]),
        nhead=int(ckpt["nhead"]),
        num_layers=int(ckpt["num_layers"]),
        ff_dim=int(ckpt["ff_dim"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    selector.load_state_dict(ckpt["model_state_dict"]); selector.eval()
    token_mean=ckpt["token_mean"].to(device).float()
    token_std=ckpt["token_std"].to(device).float()
    move_threshold=float(ckpt.get("move_threshold",0.5) if ARGS.move_threshold is None else ARGS.move_threshold)

    dataset=GraspNetMultiDataset(
        ARGS.dataset_root,split=ARGS.split,camera=ARGS.camera,num_points=ARGS.num_points,
        remove_outlier=True,augment=False,load_label=False,use_gt_depth=False,
        min_depth=ARGS.min_depth,max_depth=ARGS.max_depth,bin_num=ARGS.bin_num,
    )
    indices=subset_indices(len(dataset),ARGS.sample_interval,ARGS.max_samples)
    loader=DataLoader(
        Subset(dataset,indices),batch_size=1,shuffle=False,num_workers=ARGS.num_workers,
        collate_fn=collate_fn,pin_memory=False,persistent_workers=(ARGS.num_workers>0),
    )
    model=economicgrasp_dpt_student(
        min_depth=ARGS.min_depth,max_depth=ARGS.max_depth,bin_num=ARGS.bin_num,
        is_training=False,use_obs_depth=False,pose_depth_mode=ARGS.pose_depth_mode,
        use_cdf=True,vis_dir=None,
    ).to(device)
    load_stage1(model,ARGS.checkpoint_path); model.eval()
    evaluator=ExactGraspNetActionEvaluator(
        ARGS.dataset_root,ARGS.camera,split=ARGS.split,
        fc_mode=ARGS.fc_mode,verify_n=ARGS.verify_n,strict=True,
    )

    rows=[]; sample_rows=[]; timing=defaultdict(float); noop_max=defaultdict(float)
    learned_hist=Counter(); raw_hist=Counter(); oracle_hist=Counter()
    cand_file=None; cand_writer=None
    if ARGS.save_candidate_rows:
        cand_file=gzip.open(out/"per_candidate.csv.gz","wt",newline="")
        fields=["split","scene_id","anno_id","query_id","offset_mm","valid","raw_score",
                "selector_logit","delta_pred","utility","friction","selected_learned",
                "selected_raw","selected_oracle"]
        cand_writer=csv.DictWriter(cand_file,fieldnames=fields); cand_writer.writeheader()

    try:
        for local_i,batch in enumerate(loader):
            batch=move_batch(batch,device)
            batch["cva_export_angle_feature"]=False
            batch["cva_compute_diagnostics"]=False
            batch["geometry_compute_diagnostics"]=False
            t0=time.perf_counter()
            with torch.inference_mode():
                ep=model(batch)
                native_pred=pred_decode_center_view_angle(ep,use_cdf=True)[0]
            native_sec=time.perf_counter()-t0
            native_xyz=ep["kview_base_xyz_graspable"].float()
            token_idx=ep["kview_base_token_sel_idx"].long()
            H,W=ep["depth_map_used_for_geometry"].shape[-2:]
            centers,valid_center=build_ray_center_hypotheses(
                native_xyz,token_idx,ep["K"],(H,W),offsets,ARGS.min_depth,ARGS.max_depth
            )
            qidx=select_queries(native_pred,ARGS.query_eval_num,ARGS.query_eval_mode)

            grasps_k=[]; sel_k=[]; mean_k=[]
            t1=time.perf_counter()
            with torch.inference_mode():
                for k in range(len(offsets)):
                    epk,grouped=rerun_cdf_with_read_center(model,ep,read_center=centers[k],output_center=centers[k])
                    if local_i<ARGS.noop_check_samples and k==zero_k:
                        m=assert_native_reread_equivalent(ep,epk,atol=ARGS.noop_atol)
                        replay=pred_decode_center_view_angle(epk,use_cdf=True)[0]
                        m["noop_decoded_max_abs"]=float((replay-native_pred).abs().max().item())
                        if m["noop_decoded_max_abs"]>ARGS.noop_atol:
                            raise RuntimeError("No-op decoded replay failed.")
                        for name,val in m.items():
                            noop_max[name]=max(noop_max[name],float(val))
                    pred=native_pred if k==zero_k else pred_decode_center_view_angle(epk,use_cdf=True)[0]
                    sel,mean,_=extract_action_conditioned_features(grouped,epk)
                    grasps_k.append(pred.index_select(0,qidx))
                    sel_k.append(sel[0].index_select(0,qidx))
                    mean_k.append(mean[0].index_select(0,qidx))
            reread_sec=time.perf_counter()-t1

            grasps=torch.stack(grasps_k).detach().cpu().numpy().astype(np.float32)
            sel=torch.stack(sel_k); mean=torch.stack(mean_k)
            raw=torch.from_numpy(grasps[:,:,0]).to(device)
            offs=torch.as_tensor(offsets,device=device,dtype=sel.dtype)
            tokens=compose_relational_tokens(sel,mean,raw,offs,zero_k)
            valid=valid_center[:,0].index_select(1,qidx).detach().cpu().numpy().astype(bool)
            valid_nk=torch.from_numpy(valid.T).to(device)
            if tokens.shape[-1]!=selector.token_dim:
                raise RuntimeError(f"Relational token mismatch {tokens.shape[-1]} vs {selector.token_dim}")
            with torch.inference_mode():
                rel=selector((tokens-token_mean)/token_std.clamp_min(1e-5),valid_nk,zero_k)
            learned_k,best_alt_k,move_prob=select_relational_correction(
                rel["gate_logit"],rel["selector_logits"],valid_nk,zero_k,move_threshold
            )
            selector_logits=rel["selector_logits"].detach().cpu().numpy()
            delta_pred=rel["delta_pred"].detach().cpu().numpy()

            scene_id=int(batch["scene_idx"].reshape(-1)[0]); anno_id=int(batch["anno_idx"].reshape(-1)[0])
            mats,et=evaluate_grid(evaluator,scene_id,anno_id,grasps,valid)
            utility=friction_utility(mats["friction"]); raw_score=grasps[:,:,0]
            raw_k=select_raw_score(raw_score,valid)
            oracle_k=select_exact_oracle(utility,raw_score,valid)
            native_k=np.full(len(learned_k),zero_k,np.int64)
            mets={name:selected_metrics(mats,k) for name,k in (
                ("native",native_k),("raw",raw_k),("learned",learned_k),("oracle",oracle_k)
            )}
            qids=qidx.detach().cpu().numpy().astype(np.int64)
            for j,qid in enumerate(qids):
                native_u=float(mets["native"]["utility"][j]); learned_u=float(mets["learned"]["utility"][j])
                row={
                    "split":ARGS.split,"scene_id":scene_id,"anno_id":anno_id,"query_id":int(qid),
                    "move_threshold":move_threshold,"move_probability":float(move_prob[j]),
                    "learned_offset_mm":float(offsets[int(learned_k[j])]),
                    "best_alt_offset_mm":float(offsets[int(best_alt_k[j])]),
                    "raw_offset_mm":float(offsets[int(raw_k[j])]),
                    "oracle_offset_mm":float(offsets[int(oracle_k[j])]),
                    "learned_utility_delta":learned_u-native_u,
                }
                for name,karr in (("native",native_k),("raw",raw_k),("learned",learned_k),("oracle",oracle_k)):
                    m=mets[name]
                    for key in ("utility","success04","success08","collision_or_empty","pure_collision","empty","assigned_obj"):
                        row[f"{name}_{key}"]=float(m[key][j]) if key=="utility" else int(m[key][j])
                row["learned_rescue08"]=int((not row["native_success08"]) and row["learned_success08"])
                row["learned_harm08"]=int(row["native_success08"] and (not row["learned_success08"]))
                row["raw_rescue08"]=int((not row["native_success08"]) and row["raw_success08"])
                row["raw_harm08"]=int(row["native_success08"] and (not row["raw_success08"]))
                row["learned_matches_oracle"]=int(learned_k[j]==oracle_k[j])
                row["raw_matches_oracle"]=int(raw_k[j]==oracle_k[j])
                rows.append(row)
                learned_hist[float(offsets[int(learned_k[j])])]+=1
                raw_hist[float(offsets[int(raw_k[j])])]+=1
                oracle_hist[float(offsets[int(oracle_k[j])])]+=1

                if cand_writer:
                    for k,o in enumerate(offsets):
                        cand_writer.writerow({
                            "split":ARGS.split,"scene_id":scene_id,"anno_id":anno_id,"query_id":int(qid),
                            "offset_mm":float(o),"valid":int(valid[k,j]),"raw_score":float(raw_score[k,j]),
                            "selector_logit":float(selector_logits[j,k]),"delta_pred":float(delta_pred[j,k]),
                            "utility":float(utility[k,j]),"friction":float(mats["friction"][k,j]),
                            "selected_learned":int(k==learned_k[j]),"selected_raw":int(k==raw_k[j]),
                            "selected_oracle":int(k==oracle_k[j]),
                        })

            sample={"split":ARGS.split,"scene_id":scene_id,"anno_id":anno_id,"num_queries":len(qids),
                    "native_forward_sec":native_sec,"reread_sec":reread_sec,"exact_eval_sec":et["eval_sec"],
                    "move_rate":float((learned_k!=zero_k).mean()),"move_probability_mean":float(move_prob.mean())}
            for name in mets:
                sample[f"{name}_utility"]=float(mets[name]["utility"].mean())
                sample[f"{name}_success08"]=float(mets[name]["success08"].mean())
            sample_rows.append(sample)
            timing["native_forward_sec"]+=native_sec; timing["reread_sec"]+=reread_sec
            for k,v in et.items(): timing[k]+=float(v)
            if local_i%20==0:
                print(
                    f"[REL-TEST] {local_i+1}/{len(indices)} scene={scene_id:04d} anno={anno_id:04d} "
                    f"native08={sample['native_success08']:.3f} learned08={sample['learned_success08']:.3f} "
                    f"oracle08={sample['oracle_success08']:.3f} move={sample['move_rate']:.3f}",
                    flush=True,
                )
    finally:
        if cand_file: cand_file.close()

    write_csv(out/"per_query.csv",rows); write_csv(out/"per_sample_summary.csv",sample_rows)
    def agg(name):
        return {metric:float(np.mean([r[f"{name}_{metric}"] for r in rows])) for metric in
                ("utility","success04","success08","collision_or_empty","pure_collision","empty")}
    summary={
        "protocol":"cross-center relational selective correction v1",
        "split":ARGS.split,"num_samples":len(sample_rows),"num_queries":len(rows),
        "move_threshold":move_threshold,
        "selector_checkpoint":os.path.abspath(ARGS.selector_checkpoint),
        "stage1_checkpoint":os.path.abspath(ARGS.checkpoint_path),
        "noop_replay_max":dict(noop_max),
        "aggregate":{name:agg(name) for name in ("native","raw","learned","oracle")},
        "learned_rescue08":float(np.mean([r["learned_rescue08"] for r in rows])),
        "learned_harm08":float(np.mean([r["learned_harm08"] for r in rows])),
        "raw_rescue08":float(np.mean([r["raw_rescue08"] for r in rows])),
        "raw_harm08":float(np.mean([r["raw_harm08"] for r in rows])),
        "learned_matches_oracle":float(np.mean([r["learned_matches_oracle"] for r in rows])),
        "raw_matches_oracle":float(np.mean([r["raw_matches_oracle"] for r in rows])),
        "move_rate":float(np.mean([float(r["learned_offset_mm"])!=0.0 for r in rows])),
        "move_probability_mean":float(np.mean([r["move_probability"] for r in rows])),
        "offset_hist":{
            "learned":{str(k):int(v) for k,v in sorted(learned_hist.items())},
            "raw":{str(k):int(v) for k,v in sorted(raw_hist.items())},
            "oracle":{str(k):int(v) for k,v in sorted(oracle_hist.items())},
        },
        "timing_total_sec":{k:float(v) for k,v in timing.items()},
    }
    summary["gaps"]={
        "learned_minus_native_utility":summary["aggregate"]["learned"]["utility"]-summary["aggregate"]["native"]["utility"],
        "learned_minus_native_success08":summary["aggregate"]["learned"]["success08"]-summary["aggregate"]["native"]["success08"],
        "oracle_minus_learned_utility":summary["aggregate"]["oracle"]["utility"]-summary["aggregate"]["learned"]["utility"],
        "oracle_minus_learned_success08":summary["aggregate"]["oracle"]["success08"]-summary["aggregate"]["learned"]["success08"],
    }
    headroom=summary["aggregate"]["oracle"]["utility"]-summary["aggregate"]["native"]["utility"]
    summary["utility_headroom_recovery"]=(
        summary["gaps"]["learned_minus_native_utility"]/headroom if abs(headroom)>1e-12 else float("nan")
    )
    with (out/"summary.json").open("w") as f:
        json.dump(summary,f,indent=2,sort_keys=True)
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
