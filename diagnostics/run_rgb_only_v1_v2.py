#!/usr/bin/env python3
"""V1/V2 inference runner for the RGB-only four-validation suite.

V1 performs geometry counterfactuals at a fixed image-seed bank.
V2 performs GVAR role-mask readout ablations at a fixed physical action bank.
The runner uses one sample per forward to keep intervention identity explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from graspnetAPI import GraspGroup

def _parse_flags():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--validation",choices=("v1","v2"),required=True)
    p.add_argument("--validation_output_root",required=True)
    p.add_argument("--validation_max_samples",type=int,default=0)
    p.add_argument("--validation_seed",type=int,default=20261123)
    p.add_argument("--validation_role_masks",nargs="*",default=None)
    a,remaining=p.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    return a

V=_parse_flags()
from utils.arguments import cfgs
from models.grasp_designs import design_options,validate_checkpoint_design
from dataset.graspnet_dataset import GraspNetMultiDataset,collate_fn
from models.economicgrasp_bip3d import economicgrasp_dpt,pred_decode_center_view_angle
from utils.label_generation import batch_viewpoint_params_to_matrix
from diagnostics.rgb_only_validation_geometry import ROLE_MASKS,normalize_role_mask,assert_fixed_action_bank_identity


def _load_model(device):
    if not cfgs.multi_modal or not bool(getattr(cfgs,"use_cdf",False)):
        raise RuntimeError("V1/V2 require --multi_modal --use_cdf.")
    if bool(getattr(cfgs,"use_obs_depth",False)) or bool(getattr(cfgs,"use_gt_depth",False)):
        raise RuntimeError("V1/V2 require RGB-only predicted geometry; disable observed/GT depth.")
    if str(getattr(cfgs,"kview_mode","A1")).upper()!="A1":
        raise RuntimeError("V1/V2 require kview_mode=A1 so each seed has one view query.")
    if bool(getattr(cfgs,"use_top4_view_infer",False)):
        raise RuntimeError("V1/V2 require use_top4_view_infer=False for fixed query identity.")
    model=economicgrasp_dpt(
        **design_options(cfgs),
        min_depth=cfgs.min_depth,max_depth=cfgs.max_depth,bin_num=cfgs.bin_num,
        is_training=False,use_obs_depth=False,
        pose_depth_mode=str(getattr(cfgs,"pose_depth_mode","none")),
        camera_pose_key=str(getattr(cfgs,"camera_pose_key","camera_pose_vec")),
        camera_gravity_key=str(getattr(cfgs,"camera_gravity_key","camera_gravity_vec")),
        pose_hidden_dim=int(getattr(cfgs,"pose_hidden_dim",64)),
        ray_gravity_hidden_dim=int(getattr(cfgs,"ray_gravity_hidden_dim",64)),
        ray_gravity_mid_dim=int(getattr(cfgs,"ray_gravity_mid_dim",32)),
        use_cdf=True,vis_dir=None,vis_every=1000,
    ).to(device)
    ck=torch.load(cfgs.checkpoint_path,map_location="cpu",weights_only=False)
    validate_checkpoint_design(model,ck)
    state=ck["model_state_dict"] if isinstance(ck,dict) and "model_state_dict" in ck else ck
    result=model.load_state_dict(state,strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")
    model.eval()
    digest=hashlib.sha256(Path(cfgs.checkpoint_path).read_bytes()).hexdigest()
    return model,digest


def _build_dataset():
    if not cfgs.test_mode:
        raise ValueError("--test_mode is required.")
    ds=GraspNetMultiDataset(
        cfgs.dataset_root,split=cfgs.test_mode,camera=cfgs.camera,
        num_points=cfgs.num_point,remove_outlier=True,augment=False,load_label=False,
    )
    total=len(ds)
    fraction=float(getattr(cfgs,"sample_interval",1.0))
    if fraction<=0: raise ValueError("sample_interval must be positive")
    if fraction>=1: indices=list(range(total))
    else:
        stride=max(1,int(round(1.0/fraction)))
        indices=[]
        for start in range(0,total,256):
            indices.extend(range(start,min(start+256,total),stride))
    if V.validation_max_samples>0:
        indices=indices[:V.validation_max_samples]
    return ds,indices


def _move_batch(batch,device):
    for k,v in batch.items():
        if isinstance(v,(list,tuple)):
            raise TypeError(f"Unexpected list-valued inference key {k!r}; use load_label=False")
        if torch.is_tensor(v): batch[k]=v.to(device,non_blocking=False)
    return batch


def _make_batch(ds,index,device):
    np.random.seed(V.validation_seed+int(index))
    torch.manual_seed(V.validation_seed+int(index))
    batch=collate_fn([ds[int(index)]])
    return _move_batch(batch,device)


def _sample_meta(ds,index):
    scene=str(ds.scenename[index])
    scene_id=int(scene.split("_")[-1])
    anno_id=int(ds.frameid[index])
    return scene,scene_id,anno_id


def _common_seed_bank(base_ep,batch,model):
    idx=base_ep.get("kview_base_token_sel_idx",base_ep.get("token_sel_idx"))
    if idx is None: raise KeyError("Model did not export kview_base_token_sel_idx/token_sel_idx")
    idx=idx[0].long()
    pred=base_ep["depth_map_pred"][0,0].reshape(-1)
    gt=batch["gt_depth_m"]
    if gt.dim()==4: gt=gt[:,0]
    elif gt.dim()==2: gt=gt.unsqueeze(0)
    gt=gt[0].reshape(-1)
    valid=torch.isfinite(pred)&(pred>model.min_depth)&(pred<model.max_depth)
    valid &= torch.isfinite(gt)&(gt>model.min_depth)&(gt<model.max_depth)
    keep=valid.index_select(0,idx)
    fixed=idx[keep]
    if fixed.numel()<16:
        raise RuntimeError(f"Only {fixed.numel()} common-valid seeds; need at least 16")
    return fixed


def _save_prediction(root,variant,scene,camera,anno,pred,meta):
    out=Path(root)/variant/scene/camera
    out.mkdir(parents=True,exist_ok=True)
    pred_np=pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    GraspGroup(pred_np).save_npy(str(out/f"{anno:04d}.npy"))
    sidecar=out/f"{anno:04d}_candidate_bank.npz"
    np.savez_compressed(sidecar,pred=pred_np,**meta)
    return str(out/f"{anno:04d}.npy"),str(sidecar)


def _decode(ep):
    pred=pred_decode_center_view_angle(ep,use_cdf=True)
    if len(pred)!=1: raise RuntimeError(f"Expected batch size 1, got {len(pred)}")
    return pred[0]


def _run_v1(model,ds,indices,device,root,ckpt_sha):
    conditions=[
      ("baseline_pred_pred",{"evidence_source":"pred","action_source":"pred","center_z_source":"action"},False),
      ("A_gt_evidence_pred_action",{"evidence_source":"gt","action_source":"pred","center_z_source":"action"},True),
      ("B1_pred_evidence_gt_ray_center",{"evidence_source":"pred","action_source":"pred","center_z_source":"gt_ray_shift"},True),
      ("B2_pred_evidence_gt_full_action",{"evidence_source":"pred","action_source":"gt","center_z_source":"action"},True),
      ("oracle_gt_evidence_gt_action",{"evidence_source":"gt","action_source":"gt","center_z_source":"action"},True),
    ]
    records=[]
    for n,index in enumerate(indices):
        batch=_make_batch(ds,index,device)
        scene,scene_id,anno=_sample_meta(ds,index)
        with torch.inference_mode():
            first=model(dict(batch))
            fixed=_common_seed_bank(first,batch,model)
        for variant,override,oracle in conditions:
            run=dict(batch)
            run["cva_geometry_override"]=dict(override)
            run["cva_fixed_token_sel_idx"]=fixed.unsqueeze(0)
            with torch.inference_mode():
                ep=model(run)
                pred=_decode(ep)
            if pred.shape[0]!=fixed.numel():
                raise RuntimeError(f"{variant}: candidate count {pred.shape[0]} != fixed seeds {fixed.numel()}")
            token=ep["token_sel_idx"][0].detach().cpu().numpy()
            pred_path,bank_path=_save_prediction(
                root,variant,scene,cfgs.camera,anno,pred,
                {"candidate_id":np.arange(pred.shape[0],dtype=np.int64),
                 "token_sel_idx":token,"fixed_seed_count":np.asarray(fixed.numel()),
                 "scene_id":np.asarray(scene_id),"anno_id":np.asarray(anno),
                 "variant":np.asarray(variant),"oracle":np.asarray(oracle),
                 "evidence_source":np.asarray(override["evidence_source"]),
                 "action_source":np.asarray(override["action_source"]),
                 "center_z_source":np.asarray(override["center_z_source"])},
            )
            records.append({"split":cfgs.test_mode,"scene_id":scene_id,"anno_id":anno,
              "variant":variant,"oracle":int(oracle),"fixed_seed_count":int(fixed.numel()),
              "candidate_count":int(pred.shape[0]),"prediction_path":pred_path,
              "candidate_bank_path":bank_path,"checkpoint_sha256":ckpt_sha})
        if n%20==0: print(f"[V1] samples={n+1}/{len(indices)}",flush=True)
    _write_manifest(Path(root)/"sample_manifest.csv",records)
    print(f"[V1 DONE] samples={len(indices)} output={root}")


def _cdf_utility(ep):
    cdf=ep["grasp_cdf_pred_angle_depth"][0]
    centers=ep["xyz_graspable"][0]
    Q=int(centers.shape[0])
    A=int(getattr(cfgs,"num_angle",12))
    D=int(getattr(cfgs,"num_depth",4))
    if cdf.ndim!=4:
        raise RuntimeError(f"Expected 4D per-batch CDF tensor, got {tuple(cdf.shape)}")
    if tuple(cdf.shape[-2:])==(A,D):
        if cdf.shape[1]==Q:       # [T,Q,A,D]
            utility=torch.sigmoid(cdf).mean(dim=0)
        elif cdf.shape[0]==Q:     # [Q,T,A,D]
            utility=torch.sigmoid(cdf).mean(dim=1)
        else:
            raise RuntimeError(f"Cannot identify CDF query axis in {tuple(cdf.shape)} for Q={Q}")
    elif tuple(cdf.shape[1:3])==(A,D) and cdf.shape[0]==Q:  # [Q,A,D,T]
        utility=torch.sigmoid(cdf).mean(dim=-1)
    else:
        raise RuntimeError(f"Unsupported CDF layout {tuple(cdf.shape)} for A={A}, D={D}, Q={Q}")
    if tuple(utility.shape)!=(Q,A,D):
        raise RuntimeError(f"CDF utility shape {tuple(utility.shape)} != {(Q,A,D)}")
    return utility


def _select_width(width,depth_idx,q,angle_idx,Q,A,D):
    if tuple(width.shape)==(D,Q,A):
        return width[depth_idx,q,angle_idx]
    if tuple(width.shape)==(Q,A,D):
        return width[q,angle_idx,depth_idx]
    if tuple(width.shape)==(Q,D,A):
        return width[q,depth_idx,angle_idx]
    raise RuntimeError(f"Unsupported grasp width layout {tuple(width.shape)}")


def _build_fixed_action_bank(ep,device):
    utility=_cdf_utility(ep)
    width=ep["grasp_width_pred_angle_depth"][0]
    Q,A,D=utility.shape
    q=torch.arange(Q,device=device)
    flat=utility.reshape(Q,A*D).argmax(dim=-1)
    angle_idx=torch.div(flat,D,rounding_mode="floor")
    depth_idx=flat%D
    width_pred=_select_width(width,depth_idx,q,angle_idx,Q,A,D)
    width_m=(1.2*width_pred/10.0).clamp(0.0,float(cfgs.grasp_max_width))
    insertion=(depth_idx.to(width_pred.dtype)+1.0)*0.01
    centers=ep["xyz_graspable"][0]
    view_xyz=ep["grasp_top_view_xyz"][0]
    approaching=-view_xyz
    angle=angle_idx.to(width_pred.dtype)*(np.pi/float(A))
    rotations=batch_viewpoint_params_to_matrix(approaching,angle)
    height=torch.full_like(width_m,0.02)
    obj_id=torch.full_like(width_m,-1.0)
    pred=torch.cat((utility[q,angle_idx,depth_idx,None],
                    width_m[:,None],height[:,None],insertion[:,None],
                    rotations.reshape(Q,9),centers,obj_id[:,None]),dim=1)
    return {
      "pred":pred,"candidate_id":q,"angle_idx":angle_idx,"depth_idx":depth_idx,
      "centers":centers,"view_xyz":view_xyz,"rotations":rotations,
      "width_m":width_m,"insertion_depth_m":insertion,
      "token_sel_idx":ep["token_sel_idx"][0],"shape":(Q,A,D),
    }


def _make_fixed_bank_prediction(ep,bank):
    utility=_cdf_utility(ep)
    q=bank["candidate_id"].long()
    a=bank["angle_idx"].long()
    d=bank["depth_idx"].long()
    score=utility[q,a,d]
    pred=bank["pred"].clone()
    pred[:,0]=score
    return pred


def _run_v2(model,ds,indices,device,root,ckpt_sha):
    if not bool(getattr(cfgs,"enable_gvar",False)):
        raise RuntimeError("V2 requires --enable_gvar and a GVAR checkpoint")
    mask_names=V.validation_role_masks or [
      "all","contact_closing","clearance_approach",
      "only_left_contact","only_right_contact","only_closing",
      "only_finger_body","only_palm","only_approach",
      "leave_left_contact","leave_right_contact","leave_closing",
      "leave_finger_body","leave_palm","leave_approach",
    ]
    unknown=[x for x in mask_names if x not in ROLE_MASKS]
    if unknown: raise ValueError(f"Unknown role masks: {unknown}")
    records=[]
    for n,index in enumerate(indices):
        batch=_make_batch(ds,index,device)
        scene,scene_id,anno=_sample_meta(ds,index)
        with torch.inference_mode():
            first=dict(batch)
            first["cva_gvar_role_mask"]=torch.as_tensor(ROLE_MASKS["all"],device=device,dtype=torch.bool)
            base0=model(first)
            fixed=_common_seed_bank(base0,batch,model)
            base_run=dict(batch)
            base_run["cva_fixed_token_sel_idx"]=fixed.unsqueeze(0)
            base_run["cva_gvar_role_mask"]=torch.as_tensor(ROLE_MASKS["all"],device=device,dtype=torch.bool)
            base_ep=model(base_run)
            bank=_build_fixed_action_bank(base_ep,device)
            reference=bank["pred"].detach().cpu().numpy()
        for name in mask_names:
            mask=normalize_role_mask(name)
            run=dict(batch)
            run["cva_fixed_token_sel_idx"]=fixed.unsqueeze(0)
            run["cva_gvar_role_mask"]=torch.as_tensor(mask,device=device,dtype=torch.bool)
            with torch.inference_mode():
                ep=model(run)
                pred=_make_fixed_bank_prediction(ep,bank)
            # Reader masks are downstream of seed/view selection. A mismatch
            # means this is no longer a fixed action-bank experiment.
            for key in ("kview_base_token_sel_idx","token_sel_idx"):
                if key in ep and key in base_ep and not torch.equal(ep[key],base_ep[key]):
                    raise AssertionError(f"{name}: {key} changed across role masks")
            if not torch.allclose(ep["xyz_graspable"],base_ep["xyz_graspable"],atol=1e-6,rtol=0):
                raise AssertionError(f"{name}: action centers changed across role masks")
            if not torch.allclose(ep["grasp_top_view_xyz"],base_ep["grasp_top_view_xyz"],atol=1e-6,rtol=0):
                raise AssertionError(f"{name}: selected views changed across role masks")
            pred_np=pred.detach().cpu().numpy()
            assert_fixed_action_bank_identity(reference,pred_np)
            pred_path,bank_path=_save_prediction(
                root,name,scene,cfgs.camera,anno,pred,
                {"candidate_id":bank["candidate_id"].detach().cpu().numpy(),
                 "token_sel_idx":bank["token_sel_idx"].detach().cpu().numpy(),
                 "angle_idx":bank["angle_idx"].detach().cpu().numpy(),
                 "depth_idx":bank["depth_idx"].detach().cpu().numpy(),
                 "role_mask":mask.astype(np.uint8),"fixed_action_bank":np.asarray(True),
                 "scene_id":np.asarray(scene_id),"anno_id":np.asarray(anno),
                 "variant":np.asarray(name),"oracle":np.asarray(False)},
            )
            records.append({"split":cfgs.test_mode,"scene_id":scene_id,"anno_id":anno,
              "variant":name,"oracle":0,"fixed_seed_count":int(fixed.numel()),
              "candidate_count":int(pred.shape[0]),"fixed_action_bank":1,
              "prediction_path":pred_path,"candidate_bank_path":bank_path,
              "checkpoint_sha256":ckpt_sha})
        if n%20==0: print(f"[V2] samples={n+1}/{len(indices)}",flush=True)
    _write_manifest(Path(root)/"sample_manifest.csv",records)
    print(f"[V2 DONE] samples={len(indices)} output={root}")


def _write_manifest(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not records: raise RuntimeError("No validation samples were processed")
    fields=sorted({k for row in records for k in row})
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)


def main():
    if cfgs.collision_thresh>0:
        raise RuntimeError("V1/V2 must save unfiltered candidates; set --collision_thresh 0")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model,ckpt_sha=_load_model(device)
    ds,indices=_build_dataset()
    root=Path(V.validation_output_root); root.mkdir(parents=True,exist_ok=True)
    config={"validation":V.validation,"split":cfgs.test_mode,"camera":cfgs.camera,
      "dataset_root":cfgs.dataset_root,"checkpoint":str(cfgs.checkpoint_path),
      "checkpoint_sha256":ckpt_sha,"git_branch":"exp/rgb-only-four-validation-suite",
      "sample_count":len(indices),"seed":V.validation_seed,"oracle_conditions":"V1 only"}
    with open(root/"run_config.json","w",encoding="utf-8") as f: json.dump(config,f,indent=2)
    if V.validation=="v1": _run_v1(model,ds,indices,device,root,ckpt_sha)
    else: _run_v2(model,ds,indices,device,root,ckpt_sha)

if __name__=="__main__": main()
