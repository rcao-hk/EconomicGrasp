#!/usr/bin/env python3
"""Export RGB-only predicted/GT depth maps and masks for V4 scale diagnostics.

The dataset adapter follows run_rgb_only_v4_depth_override.py. If no full
resolution/cropped segmentation map is present in the sample, foreground
fallback is valid GT depth and background is empty; this is explicitly marked
and is unsuitable for support-plane fitting without an external mask.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib,inspect,json,sys
from pathlib import Path
import numpy as np
import torch

def _parse_flags():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--export_output_root",required=True)
    p.add_argument("--export_dataset_type",choices=("graspnet","gntrans","gc6d"),default="graspnet")
    p.add_argument("--export_dataset_module",default="")
    p.add_argument("--export_dataset_class",default="")
    p.add_argument("--export_dataset_kwargs_json",default="{}")
    p.add_argument("--export_dataset_name",default="")
    p.add_argument("--export_max_samples",type=int,default=0)
    p.add_argument("--export_seed",type=int,default=20261123)
    a,remaining=p.parse_known_args(); sys.argv=[sys.argv[0],*remaining]
    return a
A=_parse_flags()
from utils.arguments import cfgs
from models.grasp_designs import design_options,validate_checkpoint_design
from models.economicgrasp_bip3d import economicgrasp_dpt
from dataset.graspnet_dataset import collate_fn


def _dataset():
    if A.export_dataset_module: module_name=A.export_dataset_module
    elif A.export_dataset_type=="gc6d": module_name="dataset.gc6d_dataset"
    else: module_name="dataset.graspnet_dataset"
    if A.export_dataset_class: class_name=A.export_dataset_class
    elif A.export_dataset_type=="graspnet": class_name="GraspNetMultiDataset"
    elif A.export_dataset_type=="gntrans": class_name="GraspNetTransDataset"
    else: class_name="GraspClutter6DDataset"
    module=importlib.import_module(module_name)
    if not hasattr(module,class_name):
        raise AttributeError(f"{module_name}.{class_name} missing; provide explicit export_dataset_module/class")
    cls=getattr(module,class_name)
    extra=json.loads(A.export_dataset_kwargs_json)
    values={"root":cfgs.dataset_root,"root_dir":cfgs.dataset_root,"dataset_root":cfgs.dataset_root,
      "data_root":cfgs.dataset_root,"path":cfgs.dataset_root,"camera":cfgs.camera,
      "camera_name":cfgs.camera,"split":cfgs.test_mode,"mode":cfgs.test_mode,
      "num_points":cfgs.num_point,"num_point":cfgs.num_point,"remove_outlier":True,
      "augment":False,"load_label":False,"use_gt_depth":False,
      "min_depth":cfgs.min_depth,"max_depth":cfgs.max_depth}
    values.update(extra); sig=inspect.signature(cls.__init__); params=list(sig.parameters.values())[1:]
    kwargs={}; missing=[]
    for param in params:
      if param.kind in (inspect.Parameter.VAR_POSITIONAL,inspect.Parameter.VAR_KEYWORD): continue
      if param.name in values and values[param.name] is not None: kwargs[param.name]=values[param.name]
      elif param.default is inspect.Parameter.empty: missing.append(param.name)
    if missing: raise TypeError(f"Missing dataset constructor args {missing}; signature={sig}; use export_dataset_kwargs_json")
    return cls(**kwargs)


def _sample_meta(ds,index,sample):
    scene=str(ds.scenename[index]) if hasattr(ds,"scenename") else None
    anno=int(ds.frameid[index]) if hasattr(ds,"frameid") else None
    if scene is None:
      for k in ("scene_name","scene","scenename"):
        if k in sample: scene=str(sample[k]); break
    if anno is None:
      for k in ("anno_id","anno_idx","frame_id","frameid"):
        if k in sample: anno=int(sample[k]); break
    if scene is None:
      for k in ("scene_idx","scene_id"):
        if k in sample: scene=f"scene_{int(sample[k]):04d}"; break
    if anno is None: anno=int(index%256)
    if scene is None: raise KeyError("Dataset adapter must expose scene name or scene_idx")
    digits="".join(c for c in scene if c.isdigit())
    return scene,int(digits[-4:]),int(anno)


def _segmentation_mask(sample,batch,hw,valid):
    h,w=hw
    for source in (sample,batch):
      for key in ("seg_map","segmentation_map","label_map","segmentation","seg","mask"):
        if key not in source: continue
        value=source[key]
        if torch.is_tensor(value): value=value.detach().cpu().numpy()
        arr=np.asarray(value)
        while arr.ndim>2 and arr.shape[0]==1: arr=arr[0]
        if arr.ndim==3 and arr.shape[-1] in (1,3): arr=arr[...,0]
        if arr.shape==(h,w):
          return (arr>0)&valid,f"segmentation:{key}"
    return valid.copy(),"fallback:valid_gt_depth"


def main():
    if not cfgs.multi_modal: raise RuntimeError("Depth export expects the RGB-only multi-modal model")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    cksha=hashlib.sha256(Path(cfgs.checkpoint_path).read_bytes()).hexdigest()
    ds=_dataset()
    frac=float(getattr(cfgs,"sample_interval",1.0)); stride=1 if frac>=1 else max(1,int(round(1./frac)))
    indices=[]
    for start in range(0,len(ds),256): indices.extend(range(start,min(start+256,len(ds)),stride))
    if A.export_max_samples>0: indices=indices[:A.export_max_samples]
    root=Path(A.export_output_root); root.mkdir(parents=True,exist_ok=True)
    depth_root=root/"depth"; mask_root=root/"masks"
    rows=[]
    for n,index in enumerate(indices):
      np.random.seed(A.export_seed+int(index)); torch.manual_seed(A.export_seed+int(index))
      sample=ds[int(index)]; batch=collate_fn([sample])
      for k,v in batch.items():
        if isinstance(v,(list,tuple)): raise TypeError(f"Unexpected list-valued key {k}")
        if torch.is_tensor(v): batch[k]=v.to(device)
      with torch.inference_mode(): ep=model(dict(batch))
      if "gt_depth_m" not in batch: raise KeyError("Dataset must provide gt_depth_m for V4 depth diagnostics")
      pred=ep["depth_map_pred"][0,0].detach().cpu().numpy().astype(np.float32)
      gt=batch["gt_depth_m"]
      if gt.dim()==4: gt=gt[0,0]
      elif gt.dim()==3: gt=gt[0]
      gt=gt.detach().cpu().numpy().astype(np.float32)
      if pred.shape!=gt.shape: raise ValueError(f"pred/GT depth shapes differ: {pred.shape}/{gt.shape}")
      K=batch["K"][0].detach().cpu().numpy().astype(np.float64)
      valid=np.isfinite(gt)&(gt>=cfgs.min_depth)&(gt<=cfgs.max_depth)
      fg,mask_source=_segmentation_mask(sample,batch,pred.shape,valid)
      bg=valid&(~fg)
      scene,scene_id,anno=_sample_meta(ds,index,sample)
      ddir=depth_root/A.export_dataset_type/cfgs.test_mode
      mdir=mask_root/A.export_dataset_type/cfgs.test_mode
      ddir.mkdir(parents=True,exist_ok=True); mdir.mkdir(parents=True,exist_ok=True)
      stem=f"scene{scene_id:04d}_anno{anno:04d}"
      pp=ddir/f"{stem}_pred.npy"; gp=ddir/f"{stem}_gt.npy"
      fp=mdir/f"{stem}_foreground.npy"; bp=mdir/f"{stem}_background.npy"
      np.save(pp,pred); np.save(gp,gt); np.save(fp,fg.astype(np.uint8)); np.save(bp,bg.astype(np.uint8))
      rows.append({"dataset":A.export_dataset_name or A.export_dataset_type,"split":cfgs.test_mode,
        "scene_id":scene_id,"anno_id":anno,"dataset_idx":int(index),
        "pred_depth_path":str(pp.resolve()),"gt_depth_path":str(gp.resolve()),
        "foreground_mask_path":str(fp.resolve()),"background_mask_path":str(bp.resolve()),
        "K_json":json.dumps(K.tolist()),"mask_source":mask_source,
        "checkpoint_sha256":cksha})
      if n%20==0: print(f"[EXPORT] {n+1}/{len(indices)}",flush=True)
    fields=sorted({k for r in rows for k in r})
    with open(root/"depth_manifest.csv","w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(root/"run_config.json","w",encoding="utf-8") as f:
      json.dump({"dataset_type":A.export_dataset_type,"dataset_module":A.export_dataset_module,
        "dataset_class":A.export_dataset_class,"dataset_kwargs":A.export_dataset_kwargs_json,
        "checkpoint_sha256":cksha,"mask_fallback":"valid GT depth if no full-resolution segmentation map is exposed"},f,indent=2)
    print(f"[DONE] exported {len(rows)} samples to {root}")

if __name__=="__main__": main()
