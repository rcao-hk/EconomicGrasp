#!/usr/bin/env python3
"""V3: zero-shot camera/crop observation-parameterization consistency test.

Run once per separately trained RayRoPE position encoding checkpoint:
  --rayrope_position_type pixel2d
  --rayrope_position_type point3d
  --rayrope_position_type ray

GT depth is used only to select corresponding physical probe points. It is not
passed as the model's geometry input.
"""
from __future__ import annotations
import argparse,csv,hashlib,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from graspnetAPI import GraspGroup

def _parse_flags():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--v3_output_root",required=True)
    p.add_argument("--v3_transform",choices=("all","zoom","asym_crop","roll"),default="all")
    p.add_argument("--v3_probe_count",type=int,default=256)
    p.add_argument("--v3_topk",type=int,default=50)
    p.add_argument("--v3_roll_degrees",nargs="*",type=float,default=[15.,-15.,30.,-30.])
    p.add_argument("--v3_zoom_scale",type=float,default=1.25)
    p.add_argument("--v3_crop_fraction",type=float,default=.85)
    p.add_argument("--v3_seed",type=int,default=20261123)
    p.add_argument("--v3_max_samples",type=int,default=0)
    a,remaining=p.parse_known_args(); sys.argv=[sys.argv[0],*remaining]
    return a
V=_parse_flags()
from utils.arguments import cfgs
from models.grasp_designs import design_options,validate_checkpoint_design
from dataset.graspnet_dataset import GraspNetMultiDataset,collate_fn
from models.economicgrasp_bip3d import economicgrasp_dpt,pred_decode_center_view_angle
from diagnostics.rgb_only_validation_geometry import (
    make_zoom_homography,make_asymmetric_crop_homography,make_camera_roll_homography,
    transform_intrinsics,warp_array,unproject_pixels,project_points,match_topk_actions,
)

def _load_model(device):
    if not bool(getattr(cfgs,"enable_point_rayrope",False)):
        raise RuntimeError("V3 requires --enable_point_rayrope")
    if str(getattr(cfgs,"rayrope_position_type","ray")) not in ("ray","pixel2d","point3d"):
        raise ValueError("Unsupported --rayrope_position_type")
    if str(getattr(cfgs,"kview_mode","A1")).upper()!="A1" or bool(getattr(cfgs,"use_top4_view_infer",False)):
        raise RuntimeError("V3 requires kview_mode=A1 and use_top4_view_infer=False")
    model=economicgrasp_dpt(
        **design_options(cfgs),min_depth=cfgs.min_depth,max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,is_training=False,use_obs_depth=False,
        pose_depth_mode=str(getattr(cfgs,"pose_depth_mode","none")),
        camera_pose_key=str(getattr(cfgs,"camera_pose_key","camera_pose_vec")),
        camera_gravity_key=str(getattr(cfgs,"camera_gravity_key","camera_gravity_vec")),
        pose_hidden_dim=int(getattr(cfgs,"pose_hidden_dim",64)),
        ray_gravity_hidden_dim=int(getattr(cfgs,"ray_gravity_hidden_dim",64)),
        ray_gravity_mid_dim=int(getattr(cfgs,"ray_gravity_mid_dim",32)),
        use_cdf=bool(getattr(cfgs,"use_cdf",True)),vis_dir=None,vis_every=1000,
    ).to(device)
    ck=torch.load(cfgs.checkpoint_path,map_location="cpu",weights_only=False)
    validate_checkpoint_design(model,ck)
    state=ck["model_state_dict"] if isinstance(ck,dict) and "model_state_dict" in ck else ck
    model.load_state_dict(state,strict=True); model.eval()
    digest=hashlib.sha256(Path(cfgs.checkpoint_path).read_bytes()).hexdigest()
    return model,digest

def _build_dataset():
    ds=GraspNetMultiDataset(cfgs.dataset_root,split=cfgs.test_mode,camera=cfgs.camera,
        num_points=cfgs.num_point,remove_outlier=True,augment=False,load_label=False)
    frac=float(getattr(cfgs,"sample_interval",1.0))
    stride=1 if frac>=1 else max(1,int(round(1./frac)))
    indices=[]
    for start in range(0,len(ds),256): indices.extend(range(start,min(start+256,len(ds)),stride))
    if V.v3_max_samples>0: indices=indices[:V.v3_max_samples]
    return ds,indices

def _move(batch,device):
    for k,v in batch.items():
        if isinstance(v,(list,tuple)): raise TypeError(f"Unexpected list-valued key {k}")
        if torch.is_tensor(v): batch[k]=v.to(device)
    return batch

def _make_batch(ds,index,device):
    np.random.seed(V.v3_seed+int(index)); torch.manual_seed(V.v3_seed+int(index))
    return _move(collate_fn([ds[int(index)]]),device)

def _meta(ds,index):
    scene=str(ds.scenename[index]); return scene,int(scene.split("_")[-1]),int(ds.frameid[index])

def _warp_chw(tensor,H,interpolation):
    arr=tensor[0].detach().cpu().numpy().transpose(1,2,0)
    out=warp_array(arr,H,arr.shape[:2],interpolation=interpolation)
    return torch.from_numpy(out.transpose(2,0,1).copy()).to(tensor.device,dtype=tensor.dtype).unsqueeze(0)

def _warp_depth_tensor(tensor,H,interpolation="nearest"):
    if tensor.dim()==4:
        arr=tensor[0,0].detach().cpu().numpy()
        out=warp_array(arr,H,arr.shape,interpolation=interpolation)
        return torch.from_numpy(out.copy()).to(tensor.device,dtype=tensor.dtype)[None,None]
    if tensor.dim()==3:
        arr=tensor[0].detach().cpu().numpy()
        out=warp_array(arr,H,arr.shape,interpolation=interpolation)
        return torch.from_numpy(out.copy()).to(tensor.device,dtype=tensor.dtype)[None]
    raise ValueError(f"Unsupported depth tensor shape {tuple(tensor.shape)}")

def _transform_batch(batch,kind,value):
    out=dict(batch); img=batch["img"]; _,_,h,w=img.shape
    K=batch["K"][0].detach().cpu().numpy().astype(np.float64)
    Rz=np.eye(3,dtype=np.float64)
    if kind=="zoom":
        H=make_zoom_homography(h,w,float(value)); Knew=transform_intrinsics(K,H)
    elif kind=="asym_crop":
        frac=float(value); cw=w*frac; ch=h*frac
        x0=.11*w; y0=.07*h
        x1=min(w,x0+cw); y1=min(h,y0+ch)
        if x1-x0<.5*w or y1-y0<.5*h: raise ValueError("Asymmetric crop too small")
        H=make_asymmetric_crop_homography(h,w,(x0,y0,x1,y1)); Knew=transform_intrinsics(K,H)
    elif kind=="roll":
        H,Rz=make_camera_roll_homography(K,float(value)); Knew=K
    else: raise ValueError(kind)
    out["img"]=_warp_chw(img,H,"linear")
    for key in ("gt_depth_m","sensor_depth_m","depth_map","depth"):
        if key in batch and torch.is_tensor(batch[key]) and batch[key].dim() in (3,4):
            if batch[key].shape[-2:]==(h,w): out[key]=_warp_depth_tensor(batch[key],H,"nearest")
    out["K"]=torch.as_tensor(Knew,device=img.device,dtype=batch["K"].dtype).unsqueeze(0)
    if kind=="roll":
        for key in ("camera_pose_vec","camera_gravity_vec"):
            if key in batch and torch.is_tensor(batch[key]) and batch[key].shape[-1]==3:
                out[key]=batch[key]@torch.as_tensor(Rz,device=img.device,dtype=batch[key].dtype)
        for key in ("point_clouds","coordinates_for_voxel"):
            if key in batch and torch.is_tensor(batch[key]) and batch[key].shape[-1]==3:
                out[key]=batch[key]@torch.as_tensor(Rz,device=img.device,dtype=batch[key].dtype)
    return out,H,Rz,Knew

def _sample_features(feat,uv):
    c,h,w=feat.shape[1:]
    uv=torch.as_tensor(uv,device=feat.device,dtype=feat.dtype)
    grid=torch.stack((2.*uv[:,0]/max(w-1,1)-1.,2.*uv[:,1]/max(h-1,1)-1.),dim=-1)
    grid=grid.reshape(1,1,-1,2)
    sampled=F.grid_sample(feat,grid,mode="bilinear",padding_mode="zeros",align_corners=True)
    return sampled[0,:,0,:].transpose(0,1)

def _probe_points(batch,count,seed,min_depth,max_depth):
    gt=batch["gt_depth_m"]
    if gt.dim()==4: gt=gt[:,0]
    depth=gt[0].detach().cpu().numpy()
    h,w=depth.shape
    valid=np.isfinite(depth)&(depth>min_depth)&(depth<max_depth)
    ys,xs=np.nonzero(valid)
    if len(xs)<16: raise RuntimeError("Too few valid GT pixels for physical feature probes")
    rng=np.random.default_rng(seed)
    keep=np.sort(rng.choice(len(xs),size=min(int(count),len(xs)),replace=False))
    uv=np.stack((xs[keep],ys[keep]),-1).astype(np.float64)
    z=depth[ys[keep],xs[keep]].astype(np.float64)
    K=batch["K"][0].detach().cpu().numpy()
    xyz=unproject_pixels(uv,z,K)
    return uv,xyz,K,(h,w)

def _transform_probe_uv(uv,xyz,H,Rz,Knew,kind):
    if kind=="roll":
        xyz_new=xyz@Rz
        return project_points(xyz_new,Knew)
    uvh=np.concatenate((uv,np.ones((len(uv),1))),axis=1)@H.T
    return uvh[:,:2]/uvh[:,2:]

def _decode(ep):
    out=pred_decode_center_view_angle(ep,use_cdf=bool(getattr(cfgs,"use_cdf",True)))
    if len(out)!=1: raise RuntimeError("V3 requires batch size 1")
    return out[0].detach().cpu().numpy()

def _unroll_predictions(pred,Rz,kind):
    pred=np.asarray(pred,dtype=np.float32).copy()
    if kind=="roll":
        pred[:,13:16]=pred[:,13:16]@Rz.T
        rot=pred[:,4:13].reshape(-1,3,3)
        pred[:,4:13]=(Rz[None,:,:]@rot).reshape(-1,9)
    return pred

def _save_pred(root,transform,scene,camera,anno,pred):
    out=Path(root)/"predictions"/transform/scene/camera; out.mkdir(parents=True,exist_ok=True)
    GraspGroup(pred).save_npy(str(out/f"{anno:04d}.npy"))

def _transform_specs():
    specs=[]
    if V.v3_transform in ("all","zoom"): specs.append(("zoom",V.v3_zoom_scale))
    if V.v3_transform in ("all","asym_crop"): specs.append(("asym_crop",V.v3_crop_fraction))
    if V.v3_transform in ("all","roll"):
        specs.extend(("roll",d) for d in V.v3_roll_degrees)
    return specs

def main():
    if cfgs.collision_thresh>0: raise RuntimeError("V3 should save unfiltered predictions; set collision_thresh=0")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model,ckpt_sha=_load_model(device); ds,indices=_build_dataset()
    root=Path(V.v3_output_root); root.mkdir(parents=True,exist_ok=True)
    records=[]
    position_type=str(getattr(cfgs,"rayrope_position_type","ray"))
    for n,index in enumerate(indices):
        batch=_make_batch(ds,index,device); scene,scene_id,anno=_meta(ds,index)
        with torch.inference_mode():
            ep0=model(dict(batch)); pred0=_decode(ep0)
        _save_pred(root,"identity",scene,cfgs.camera,anno,pred0)
        uv,xyz,K,(h,w)=_probe_points(batch,V.v3_probe_count,V.v3_seed+int(index),cfgs.min_depth,cfgs.max_depth)
        feat0=_sample_features(ep0["img_feat_dpt"],uv)
        for kind,value in _transform_specs():
            transformed,H,Rz,Knew=_transform_batch(batch,kind,value)
            with torch.inference_mode():
                ep1=model(transformed); pred1=_decode(ep1)
            uv1=_transform_probe_uv(uv,xyz,H,Rz,Knew,kind)
            inside=(uv1[:,0]>=0)&(uv1[:,0]<w-1)&(uv1[:,1]>=0)&(uv1[:,1]<h-1)
            if inside.sum()<8: raise RuntimeError(f"{kind} leaves too few corresponding physical probes")
            f0=feat0[torch.as_tensor(inside,device=device)]
            f1=_sample_features(ep1["img_feat_dpt"],uv1[inside])
            cos=F.cosine_similarity(f0,f1,dim=-1).detach().cpu().numpy()
            pred1_common=_unroll_predictions(pred1,Rz,kind)
            consistency=match_topk_actions(pred0,pred1_common,topk=V.v3_topk)
            transform_name=f"{kind}_{value:g}"
            _save_pred(root,transform_name,scene,cfgs.camera,anno,pred1_common)
            records.append({
              "split":cfgs.test_mode,"scene_id":scene_id,"anno_id":anno,
              "position_type":position_type,"transform":kind,"transform_value":float(value),
              "physical_probe_count":int(inside.sum()),
              "feature_cosine_mean":float(np.mean(cos)),
              "feature_cosine_median":float(np.median(cos)),
              "feature_cosine_p10":float(np.percentile(cos,10)),
              **consistency,"checkpoint_sha256":ckpt_sha,
            })
        if n%20==0: print(f"[V3] samples={n+1}/{len(indices)} encoding={position_type}",flush=True)
    if not records: raise RuntimeError("No V3 samples were processed")
    fields=sorted({k for r in records for k in r})
    with open(root/"observation_invariance_metrics.csv","w",newline="",encoding="utf-8") as f:
        wri=csv.DictWriter(f,fieldnames=fields); wri.writeheader(); wri.writerows(records)
    print(f"[V3 DONE] samples={len(indices)} encoding={position_type} output={root}")

if __name__=="__main__": main()
