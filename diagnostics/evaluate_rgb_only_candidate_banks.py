#!/usr/bin/env python3
"""Exact evaluator diagnostics for V1/V2 fixed-seed or fixed-action candidate dumps."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator

def _write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise RuntimeError("No evaluator rows were produced")
    fields=sorted({k for r in rows for k in r})
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def _success(friction,threshold):
    friction=np.asarray(friction,dtype=np.float64)
    return (friction>0)&(friction<=float(threshold))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sample-manifest",required=True)
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--camera",default="realsense")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--split",default="test")
    args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    evaluator=ExactGraspNetActionEvaluator(
        args.dataset_root,camera=args.camera,split=args.split,device=args.device
    )
    with open(args.sample_manifest,newline="",encoding="utf-8") as f:
        manifest=list(csv.DictReader(f))
    if not manifest: raise ValueError("sample manifest is empty")
    candidate_rows=[]; summary_rows=[]
    oracle_root=out/"oracle_rerank"
    for i,row in enumerate(manifest):
        path=Path(row["candidate_bank_path"])
        if not path.exists(): raise FileNotFoundError(path)
        data=np.load(path,allow_pickle=False)
        pred=np.asarray(data["pred"],dtype=np.float32)
        scene=int(row["scene_id"]); anno=int(row["anno_id"])
        result=evaluator.evaluate(scene,anno,pred)
        friction=np.asarray(result["friction"],dtype=np.float32).reshape(-1)
        collision=np.asarray(result["pure_collision"],dtype=bool).reshape(-1)
        collision_or_empty=np.asarray(result["collision_or_empty"],dtype=bool).reshape(-1)
        empty=np.asarray(result["empty"],dtype=bool).reshape(-1)
        assigned=np.asarray(result["assigned_obj"],dtype=np.int64).reshape(-1)
        if not (len(pred)==len(friction)==len(collision)==len(empty)==len(assigned)):
            raise RuntimeError(f"Evaluator output length mismatch for {path}")
        score_order=np.argsort(-pred[:,0],kind="stable")
        oracle_key=np.where(friction>0,friction,np.inf)
        oracle_order=np.argsort(oracle_key,kind="stable")
        for rank,idx in enumerate(score_order):
            candidate_rows.append({
              "split":row.get("split",""),"variant":row["variant"],"scene_id":scene,
              "anno_id":anno,"candidate_id":int(idx),"rank":int(rank+1),
              "pred_score":float(pred[idx,0]),"eval_friction_score":float(friction[idx]),
              "pure_collision":int(collision[idx]),"collision_or_empty":int(collision_or_empty[idx]),
              "empty":int(empty[idx]),"assigned_obj_id":int(assigned[idx]),
              "success_04":int(_success(friction[idx:idx+1],.4)[0]),
              "success_08":int(_success(friction[idx:idx+1],.8)[0]),
              "oracle_rank":int(np.where(oracle_order==idx)[0][0]+1),
            })
        for k in (1,10,50):
            top=score_order[:min(k,len(score_order))]
            summary_rows.append({
              "split":row.get("split",""),"variant":row["variant"],"scene_id":scene,
              "anno_id":anno,"k":k,"candidate_count":len(pred),
              "topk_pure_collision_rate":float(collision[top].mean()),
              "topk_collision_or_empty_rate":float(collision_or_empty[top].mean()),
              "topk_empty_rate":float(empty[top].mean()),
              "topk_success_04":float(_success(friction[top],.4).mean()),
              "topk_success_08":float(_success(friction[top],.8).mean()),
            })
        oracle_pred=pred[oracle_order].copy()
        oracle_friction=friction[oracle_order]
        oracle_pred[:,0]=np.where(oracle_friction>0,-oracle_friction,-10.0)
        scene_name=f"scene_{scene:04d}"
        od=oracle_root/row["variant"]/scene_name/args.camera
        od.mkdir(parents=True,exist_ok=True)
        np.save(od/f"{anno:04d}.npy",oracle_pred)
        if i%50==0: print(f"[EVAL] {i+1}/{len(manifest)}",flush=True)
    _write_csv(out/"ranked_candidate_rows.csv",candidate_rows)
    _write_csv(out/"topk_exact_evaluator_summary.csv",summary_rows)
    print(f"[DONE] rows={len(candidate_rows)} samples={len(manifest)} oracle_dump={oracle_root}")
    print("Run the repository's official eval.py separately on each prediction dump and oracle_rerank/<variant>.")

if __name__=="__main__": main()
