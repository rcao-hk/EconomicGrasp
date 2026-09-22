#!/usr/bin/env python3
"""Rep-C3: full-path robustness under predictor-shaped and material RGB errors.

Two nontrivial stress families are supported:

1) residual_full/local: use GT depth ONLY to construct a stress field from the
   model's own prediction residual. For alpha>0:
       D_stress = D_pred + alpha * residual
   "full" keeps the residual bias; "local" removes its per-frame median first.
   GT is never passed to Stage-1/A0/A1 as evidence.

2) material: replace RGB with a paired material-augmented image while retaining
   the original scene geometry/crop. Stage-1 is rerun from the augmented RGB, so
   BOTH image features and predicted depth are generated from that same image.
   This avoids impossible feature/depth pairings.

New physical grasps are regenerated and saved for fresh CAD/DexNet evaluation.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch

from rep_a_common import (
    SPLITS, digest, exclusive_run, file_sha, frame_identity, list_frames,
    save_json, save_npz
)
from rep_followup_common import checked_manifest, load_scorer, memory_status, release_memory, source_digest
from rep_fullpath_runtime import expand_actions, load_stage1, parse_offsets
from infer_rep_fullpath import query_indices, score_queries, dump_outputs


def parse_residual_case(text):
    kind,val=text.split(":",1)
    if kind not in ("residual_full","residual_local"):
        raise ValueError(text)
    alpha=float(val)
    if not np.isfinite(alpha) or alpha<0:
        raise ValueError(text)
    return kind,alpha


def residual_stress(pred,gt,case):
    kind,alpha=parse_residual_case(case)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/GT shape mismatch: {pred.shape} vs {gt.shape}")
    valid=torch.isfinite(pred)&torch.isfinite(gt)&(pred>.01)&(gt>.01)
    if not bool(valid.any()): raise ValueError("No valid GT/pred overlap")
    resid=pred-gt
    field=resid.clone()
    center=pred.new_tensor(0.)
    if kind=="residual_local":
        center=resid[valid].median()
        field=torch.where(valid,resid-center,torch.zeros_like(resid))
    else:
        field=torch.where(valid,resid,torch.zeros_like(resid))
    changed=torch.where(valid,pred+alpha*field,pred)
    changed=torch.where(valid,changed.clamp_min(.01),changed)
    delta=(changed-pred)[valid]
    return changed,{
        "stress_kind":kind,"alpha":alpha,
        "source_residual_bias_mm":float(resid[valid].mean())*1000,
        "removed_center_mm":float(center)*1000,
        "depth_rms_mm":float(delta.square().mean().sqrt())*1000,
        "depth_bias_mm":float(delta.mean())*1000,
    }


def depth_gt_stats(depth,gt):
    valid=torch.isfinite(depth)&torch.isfinite(gt)&(depth>.01)&(gt>.01)
    if not bool(valid.any()):
        return {"gt_mae_mm":None,"gt_rmse_mm":None,"gt_bias_mm":None,"gt_valid_fraction":0.}
    e=(depth-gt)[valid]
    return {
        "gt_mae_mm":float(e.abs().mean())*1000,
        "gt_rmse_mm":float(e.square().mean().sqrt())*1000,
        "gt_bias_mm":float(e.mean())*1000,
        "gt_valid_fraction":float(valid.float().mean()),
    }


def material_path(root,scene,camera,aid):
    root=Path(root)
    candidates=[
        root/"scenes"/scene/camera/"rgb"/f"{aid:04d}.png",
        root/scene/camera/"rgb"/f"{aid:04d}.png",
        root/"scenes"/scene/"rgb"/f"{aid:04d}.png",
        root/scene/"rgb"/f"{aid:04d}.png",
        root/scene/f"{aid:04d}.png",
    ]
    for p in candidates:
        if p.is_file(): return p
    raise FileNotFoundError(
        f"No paired material RGB for {scene}/{aid:04d}; tried: "
        + ", ".join(map(str,candidates))
    )


def make_batch(item,device,collate_fn):
    allowed=("img","K","camera_pose_vec","camera_gravity_vec","scene_idx","anno_idx","token_valid_mask")
    batch=collate_fn([{k:item[k] for k in allowed if k in item}])
    batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
    batch.update(cva_export_angle_feature=False,cva_compute_diagnostics=False,geometry_compute_diagnostics=False)
    return batch


def case_key_local(case):
    return case.replace(":","_").replace("-","m").replace("+","p").replace(".","p")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--stage1-checkpoint",required=True)
    p.add_argument("--cache-root",required=True)
    p.add_argument("--output-root",required=True)
    p.add_argument("--split",choices=tuple(SPLITS)[1:],required=True)
    p.add_argument("--scorer",action="append",default=[],help="NAME=checkpoint")
    p.add_argument("--residual-cases",default="residual_full:0.5,residual_full:1.0,residual_local:0.5,residual_local:1.0")
    p.add_argument("--material",action="append",default=[],help="NAME=ROOT, repeatable")
    p.add_argument("--camera",default="realsense")
    p.add_argument("--pose-depth-mode",default="global_film")
    p.add_argument("--offsets-mm",default="-40,-20,-10,0,10,20,40")
    p.add_argument("--query-limit",type=int,default=64)
    p.add_argument("--score-query-chunk",type=int,default=64)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--shard-id",type=int,default=0)
    p.add_argument("--num-shards",type=int,default=1)
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--replay-atol",type=float,default=5e-5)
    p.add_argument("--min-host-free-gib",type=float,default=4.)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--repair-corrupt",action="store_true")
    args=p.parse_args(); sys.argv=[sys.argv[0]]

    residual_cases=[x.strip() for x in args.residual_cases.split(",") if x.strip()]
    for c in residual_cases: parse_residual_case(c)
    materials={}
    for item in args.material:
        name,root=item.split("=",1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+",name) or name in materials:
            raise ValueError(f"Bad material name: {name}")
        materials[name]=root
    cases=["nominal"]+residual_cases+[f"material:{n}" for n in materials]

    manifest=check_runtime_sources(args.cache_root)
    if file_sha(args.stage1_checkpoint)!=manifest["stage1_sha256"]:
        raise RuntimeError("Rep-C3 requires the same Stage-1 checkpoint as Rep-A")
    if manifest["camera"]!=args.camera or manifest["pose_depth_mode"]!=args.pose_depth_mode:
        raise RuntimeError("Camera/pose mode differs from Rep-A cache")

    specs={}
    for item in args.scorer:
        name,path=item.split("=",1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+",name) or name in specs: raise ValueError(name)
        specs[name]=path
    offsets=parse_offsets(args.offsets_mm)
    protocol={
        "version":1,"experiment":"Rep-C3","camera":args.camera,
        "query_limit":args.query_limit,"cases":cases,"modes":["c3"],
        "offsets_mm":offsets.tolist(),"stage1_sha":manifest["stage1_sha256"],
        "scorers":{n:file_sha(p) for n,p in specs.items()},
        "feature_precision":manifest["feature_dtype"],
        "residual_semantics":"D_stress=D_pred+alpha*(D_pred-D_gt); local removes per-frame residual median first",
        "gt_role":"GT depth constructs stress/evaluation metadata only; never model evidence",
        "material_roots":materials,
        "material_semantics":"paired RGB replaces original RGB before Stage-1; depth/features are recomputed from the same augmented image",
        "scope":"Full Stage-1 regeneration + K-ray rescore; new actions require fresh CAD/DexNet labels",
        "code":source_digest(("infer_rep_c3.py","rep_fullpath_runtime.py","infer_rep_fullpath.py","rep_a_model.py","rep_b_model.py")),
    }
    signature=digest(protocol)
    root=Path(args.output_root); root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root/".protocol.lock",wait=True): checked_manifest(root,protocol)

    paths=list_frames(args.cache_root,args.split)
    scenes=sorted({frame_identity(x)[0] for x in paths})
    owned=set(scenes[args.shard_id::args.num_shards])
    paths=[x for x in paths if frame_identity(x)[0] in owned]
    if args.max_frames>0: paths=paths[:args.max_frames]

    from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    device=torch.device(args.device)
    stage,replay,use_fuse=load_stage1(args.stage1_checkpoint,device,args.pose_depth_mode)
    scorers={n:load_scorer(p,args.cache_root,device) for n,p in specs.items()}
    dataset=GraspNetMultiDataset(
        args.dataset_root,split=args.split,camera=args.camera,num_points=20000,
        remove_outlier=True,augment=False,load_label=False,use_gt_depth=False,
        use_fuse_depth=use_fuse,min_depth=.2,max_depth=1.,bin_num=256
    )
    index={(int(str(s).split("_")[-1]),i%256):i for i,s in enumerate(dataset.scene_list())}

    def emit(ep,feature,active_depth,gt,sid,aid,case,folder,shift_stats):
        ids_all=pred_decode_center_view_angle(ep,use_cdf=True)[0].cpu().numpy()
        ids=query_indices(ids_all,args.query_limit); native=ids_all[ids]
        action,valid,zero=expand_actions(native,offsets)
        h,w=active_depth.shape[-2:]
        raw=feature.to(torch.float16 if manifest["feature_dtype"]=="float16" else torch.float32).float()
        tokens=ep["kview_base_token_sel_idx"][0][torch.as_tensor(ids,device=device)]
        data=dict(
            image_feature=raw,depth=active_depth[0],K=current_batch["K"][0],
            objectness=ep["objectness_score"][0].reshape(2,h,w),
            graspness=ep["graspness_score"][0].reshape(1,h,w),
            actions=torch.from_numpy(action).to(device),valid=torch.from_numpy(valid).to(device),
            offsets_mm=torch.from_numpy(offsets).to(device),zero_index=zero,token_ids=tokens
        )
        methods=["stage1_native"]; policies=["native"]; margins=[0.]
        chosen=[np.full(len(ids),zero,np.int64)]; scores=[native[:,0]]; probs=[]
        for name,(model,meta) in scorers.items():
            prob=score_queries(model,data,args.score_query_chunk)
            probs.append(prob)
            for policy,margin in (("fixed_0",0.),("val_selected",meta["margin"])):
                from rep_a_common import choose
                sel=choose(prob.mean(-1),valid,zero,margin)
                methods.append(name); policies.append(policy); margins.append(margin)
                chosen.append(sel); scores.append(prob.mean(-1)[sel,np.arange(len(ids))])
        payload=dict(
            signature=np.array(signature),scene_id=np.array(sid),anno_id=np.array(aid),
            split=np.array(args.split),mode=np.array("c3"),case=np.array(case),
            actions=action,valid=valid,offsets_mm=offsets,zero_index=np.array(zero),
            query_ids=ids,token_ids=tokens.cpu().numpy(),original_native_score=native[:,0],
            output_methods=np.asarray(methods),output_policies=np.asarray(policies),
            output_margins=np.asarray(margins),selected=np.stack(chosen),rank_scores=np.stack(scores),
            scorer_names=np.asarray(list(scorers)),
            probabilities=np.stack(probs) if probs else np.empty((0,*valid.shape,6),np.float32),
            replay_max_abs=np.array(current_noop),num_stage1_queries=np.array(len(ids_all)),
            depth_rms_mm=np.array(float(shift_stats.get("depth_rms_mm",0.))),
            depth_bias_mm=np.array(float(shift_stats.get("depth_bias_mm",0.))),
            anchor_depth_rms_mm=np.array(float(shift_stats.get("depth_rms_mm",0.))),
            reader_depth_rms_mm=np.array(float(shift_stats.get("depth_rms_mm",0.))),
            gt_mae_mm=np.array(float(shift_stats["gt_mae_mm"])),
            gt_rmse_mm=np.array(float(shift_stats["gt_rmse_mm"])),
            gt_bias_mm=np.array(float(shift_stats["gt_bias_mm"])),
        )
        out=folder/f"ann_{aid:04d}_c3_{case_key_local(case)}.npz"
        save_npz(out,payload)
        dump_outputs(root,payload,"c3",case,args.split,args.camera,sid,aid)
        return out

    completed=0
    for src in paths:
        sid,aid=frame_identity(src); scene=f"scene_{sid:04d}"
        folder=root/"inference"/args.split/scene; folder.mkdir(parents=True,exist_ok=True)
        expected=[folder/f"ann_{aid:04d}_c3_{case_key_local(c)}.npz" for c in cases]
        if args.resume and all(x.exists() for x in expected):
            continue
        memory_status(args.min_host_free_gib)
        idx=index[(sid,aid)]
        original_path=dataset.colorpath[idx]
        item=dataset[idx]
        gt=torch.as_tensor(item["gt_depth_m"],device=device,dtype=torch.float32)[None,None]
        current_batch=make_batch(item,device,collate_fn)
        ep0=replay.capture(current_batch)
        depth0=replay.nominal_depth.detach().clone()
        feature0=replay.image_feature.detach()
        check=replay.run(current_batch,depth0,depth0)
        native0=pred_decode_center_view_angle(ep0,use_cdf=True)[0].cpu().numpy()
        rebuilt=pred_decode_center_view_angle(check,use_cdf=True)[0].cpu().numpy()
        current_noop=float(np.max(np.abs(native0-rebuilt)))
        if current_noop>args.replay_atol: raise RuntimeError(f"C3 nominal replay failed {sid}/{aid}: {current_noop}")

        if not (args.resume and expected[0].exists()):
            st=depth_gt_stats(depth0,gt)
            st.update(depth_rms_mm=0.,depth_bias_mm=0.)
            emit(ep0,feature0,depth0,gt,sid,aid,"nominal",folder,st)

        # Predictor-shaped spatial errors on the ORIGINAL RGB/features.
        for case,path in zip(residual_cases,expected[1:1+len(residual_cases)]):
            if args.resume and path.exists(): continue
            pert,meta_err=residual_stress(depth0,gt,case)
            ep=replay.run(current_batch,pert,pert)
            st=depth_gt_stats(pert,gt); st.update(meta_err)
            emit(ep,feature0,pert,gt,sid,aid,case,folder,st)
            del ep,pert

        # Paired material RGB. Monkeypatch only this item's RGB path; original
        # depth/seg/GT determine exactly the same crop/K and physical scene.
        for name,mat_root in materials.items():
            case=f"material:{name}"
            path=folder/f"ann_{aid:04d}_c3_{case_key_local(case)}.npz"
            if args.resume and path.exists(): continue
            mpath=material_path(mat_root,scene,args.camera,aid)
            dataset.colorpath[idx]=str(mpath)
            try:
                mitem=dataset[idx]
            finally:
                dataset.colorpath[idx]=original_path
            if not np.allclose(mitem["K"],item["K"],atol=1e-6,rtol=0):
                raise RuntimeError("Material pair changed crop/intrinsics")
            current_batch=make_batch(mitem,device,collate_fn)
            mep=replay.capture(current_batch)
            mdepth=replay.nominal_depth.detach().clone()
            mfeat=replay.image_feature.detach()
            st=depth_gt_stats(mdepth,gt)
            diff=(mdepth-depth0)
            st.update(depth_rms_mm=float(diff.square().mean().sqrt())*1000,
                      depth_bias_mm=float(diff.mean())*1000)
            current_noop=0.
            emit(mep,mfeat,mdepth,gt,sid,aid,case,folder,st)
            del mitem,mep,mdepth,mfeat
        dataset.colorpath[idx]=original_path
        replay.clear(); release_memory(); completed+=1
        print(f"[REP-C3] {args.split} {sid}/{aid} completed={completed}",flush=True)

    save_json(root/f"infer_{args.split}_shard{args.shard_id}.json",{
        "signature":signature,"completed_new_frames":completed,"selected_frames":len(paths),
        "shard":args.shard_id,"num_shards":args.num_shards,
        "seen_role":"validation_seen" if args.split=="test_seen" else "held_out"
    })


if __name__=="__main__":
    main()
