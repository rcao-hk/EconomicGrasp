#!/usr/bin/env python3
"""V4: run grasp inference with depth maps produced by the scale/shape analyzer.

GraspNet is the built-in dataset adapter. GN-Trans/GC6D can be supplied through
--v4_dataset_module/--v4_dataset_class and constructor kwargs JSON. For these
datasets, include dataset_idx in the source depth manifest whenever possible.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib,inspect,json,sys
from pathlib import Path
import numpy as np
import torch
from graspnetAPI import GraspGroup

def _parse_flags():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--v4_depth_manifest",required=True)
    p.add_argument("--v4_output_root",required=True)
    p.add_argument("--v4_dataset_type",choices=("graspnet","gntrans","gc6d"),default="graspnet")
    p.add_argument("--v4_dataset_module",default="")
    p.add_argument("--v4_dataset_class",default="")
    p.add_argument("--v4_dataset_kwargs_json",default="{}")
    p.add_argument("--v4_dataset_name",default="")
    p.add_argument("--v4_variants",nargs="*",default=None)
    p.add_argument("--v4_max_samples",type=int,default=0)
    p.add_argument("--v4_seed",type=int,default=20261123)
    a,remaining=p.parse_known_args(); sys.argv=[sys.argv[0],*remaining]
    return a
V=_parse_flags()
from utils.arguments import cfgs
from models.grasp_designs import design_options,validate_checkpoint_design
from models.economicgrasp_bip3d import economicgrasp_dpt,pred_decode_center_view_angle
from dataset.graspnet_dataset import collate_fn


def _load_model(device):
    if not cfgs.multi_modal or not bool(getattr(cfgs,"use_cdf",False)):
        raise RuntimeError("V4 requires --multi_modal --use_cdf")
    model=economicgrasp_dpt(
        **design_options(cfgs),min_depth=cfgs.min_depth,max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,is_training=False,use_obs_depth=False,
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
    model.load_state_dict(state,strict=True); model.eval()
    digest=hashlib.sha256(Path(cfgs.checkpoint_path).read_bytes()).hexdigest()
    return model,digest


def _dataset_spec():
    if V.v4_dataset_module:
        module_name=V.v4_dataset_module
    elif V.v4_dataset_type=="gc6d":
        module_name="dataset.gc6d_dataset"
    else:
        module_name="dataset.graspnet_dataset"
    if V.v4_dataset_class:
        class_name=V.v4_dataset_class
    elif V.v4_dataset_type=="graspnet":
        class_name="GraspNetMultiDataset"
    elif V.v4_dataset_type=="gntrans":
        class_name="GraspNetTransDataset"
    else:
        class_name="GraspClutter6DDataset"
    module=importlib.import_module(module_name)
    if not hasattr(module,class_name):
        candidates=[n for n,v in inspect.getmembers(module,inspect.isclass)
                    if "dataset" in n.lower() and (
                       (("gc6d" in n.lower()) or ("clutter" in n.lower())) if V.v4_dataset_type=="gc6d"
                       else ("trans" in n.lower()) if V.v4_dataset_type=="gntrans"
                       else ("graspnet" in n.lower()))]
        raise AttributeError(f"{module_name}.{class_name} not found. Candidate classes: {candidates}. "
                             "Pass --v4_dataset_module and --v4_dataset_class explicitly.")
    return module,getattr(module,class_name)


def _instantiate_dataset():
    _,cls=_dataset_spec()
    extra=json.loads(V.v4_dataset_kwargs_json)
    if not isinstance(extra,dict): raise ValueError("--v4_dataset_kwargs_json must be a JSON object")
    values={
      "root":cfgs.dataset_root,"root_dir":cfgs.dataset_root,"dataset_root":cfgs.dataset_root,
      "data_root":cfgs.dataset_root,"path":cfgs.dataset_root,
      "camera":cfgs.camera,"camera_name":cfgs.camera,
      "split":cfgs.test_mode,"mode":cfgs.test_mode,
      "num_points":cfgs.num_point,"num_point":cfgs.num_point,
      "remove_outlier":True,"augment":False,"load_label":False,
      "use_gt_depth":False,"train_split":getattr(cfgs,"train_split",None),
      "min_depth":cfgs.min_depth,"max_depth":cfgs.max_depth,
    }
    values.update(extra)
    sig=inspect.signature(cls.__init__); params=list(sig.parameters.values())[1:]
    accepts_kwargs=any(x.kind==inspect.Parameter.VAR_KEYWORD for x in params)
    kwargs={}; missing=[]
    for param in params:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,inspect.Parameter.VAR_KEYWORD): continue
        if param.name in values and values[param.name] is not None: kwargs[param.name]=values[param.name]
        elif param.default is inspect.Parameter.empty: missing.append(param.name)
    if missing:
        raise TypeError(f"Cannot instantiate {cls.__module__}.{cls.__name__}; missing {missing}. "
                        f"Pass them in --v4_dataset_kwargs_json. Signature: {sig}")
    if accepts_kwargs:
        for k,v in extra.items(): kwargs[k]=v
    return cls(**kwargs)


def _dataset_indices(ds):
    frac=float(getattr(cfgs,"sample_interval",1.0))
    if frac<=0: raise ValueError("sample_interval must be positive")
    stride=1 if frac>=1 else max(1,int(round(1./frac)))
    indices=[]
    for start in range(0,len(ds),256): indices.extend(range(start,min(start+256,len(ds)),stride))
    if V.v4_max_samples>0: indices=indices[:V.v4_max_samples]
    return indices


def _sample_meta(ds,index,sample):
    scene=None; anno=None
    if hasattr(ds,"scenename"): scene=str(ds.scenename[index])
    if hasattr(ds,"frameid"): anno=int(ds.frameid[index])
    for key in ("scene_name","scene","scenename"):
        if scene is None and key in sample: scene=str(sample[key])
    for key in ("anno_id","anno_idx","frame_id","frameid"):
        if anno is None and key in sample: anno=int(sample[key])
    if scene is None:
        for key in ("scene_idx","scene_id"):
            if key in sample:
                scene=f"scene_{int(sample[key]):04d}"; break
    if anno is None: anno=int(index%256)
    if scene is None: raise KeyError("Dataset adapter must expose scenename, scene_name, or scene_idx")
    digits="".join(ch for ch in scene if ch.isdigit())
    if not digits: raise ValueError(f"Cannot parse numeric scene id from {scene!r}")
    return scene,int(digits[-4:]),int(anno),int(index)


def _move(batch,device):
    for k,v in batch.items():
        if isinstance(v,(list,tuple)): raise TypeError(f"Unexpected list-valued key {k}")
        if torch.is_tensor(v): batch[k]=v.to(device)
    return batch


def _load_manifest():
    with open(V.v4_depth_manifest,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("Depth override manifest is empty")
    selected=V.v4_variants or sorted({r["variant"] for r in rows})
    mapping={}
    for r in rows:
        if r["variant"] not in selected: continue
        if V.v4_dataset_name and r.get("dataset","")!=V.v4_dataset_name: continue
        dsidx=r.get("dataset_idx","").strip()
        key=("idx",int(dsidx),r["variant"]) if dsidx else (
             "scene",r.get("split",""),int(r["scene_id"]),int(r["anno_id"]),r["variant"])
        mapping[key]=r
    return selected,mapping


def _find_row(mapping,variant,split,scene_id,anno_id,index):
    row=mapping.get(("idx",index,variant))
    if row is not None: return row
    return mapping.get(("scene",split,scene_id,anno_id,variant))


def _load_depth(row,hw):
    path=Path(row["depth_path"])
    if not path.exists(): raise FileNotFoundError(path)
    depth=np.asarray(np.load(path),dtype=np.float32)
    if depth.ndim==3: depth=depth.squeeze()
    if depth.ndim!=2: raise ValueError(f"Expected HxW depth map, got {depth.shape} from {path}")
    if depth.shape!=tuple(hw):
        import cv2
        depth=cv2.resize(depth,(int(hw[1]),int(hw[0])),interpolation=cv2.INTER_NEAREST)
    if not np.isfinite(depth).all(): raise FloatingPointError(f"Non-finite depth values in {path}")
    return depth


def _common_seeds(base_ep,depth_maps,model):
    idx=base_ep.get("kview_base_token_sel_idx",base_ep.get("token_sel_idx"))
    if idx is None: raise KeyError("Model did not export base seed indices")
    idx=idx[0].long(); pred=base_ep["depth_map_pred"][0,0].reshape(-1)
    valid=torch.isfinite(pred)&(pred>model.min_depth)&(pred<model.max_depth)
    for depth in depth_maps:
        d=torch.as_tensor(depth,device=pred.device,dtype=pred.dtype).reshape(-1)
        valid &= torch.isfinite(d)&(d>model.min_depth)&(d<model.max_depth)
    fixed=idx[valid.index_select(0,idx)]
    if fixed.numel()<16: raise RuntimeError(f"Only {fixed.numel()} common-valid action seeds")
    return fixed


def _save(root,variant,scene,camera,anno,pred,meta):
    out=Path(root)/variant/scene/camera; out.mkdir(parents=True,exist_ok=True)
    arr=pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    GraspGroup(arr).save_npy(str(out/f"{anno:04d}.npy"))
    bank=out/f"{anno:04d}_candidate_bank.npz"
    np.savez_compressed(bank,pred=arr,**meta)
    return str(out/f"{anno:04d}.npy"),str(bank)


def main():
    if cfgs.collision_thresh>0: raise RuntimeError("V4 saves unfiltered candidates; set collision_thresh=0")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model,ckpt_sha=_load_model(device); ds=_instantiate_dataset(); indices=_dataset_indices(ds)
    variants,mapping=_load_manifest()
    root=Path(V.v4_output_root); root.mkdir(parents=True,exist_ok=True)
    records=[]
    for n,index in enumerate(indices):
        np.random.seed(V.v4_seed+int(index)); torch.manual_seed(V.v4_seed+int(index))
        sample=ds[int(index)]; scene,scene_id,anno,dataset_idx=_sample_meta(ds,index,sample)
        batch=_move(collate_fn([sample]),device)
        rows={}
        for variant in variants:
            row=_find_row(mapping,variant,cfgs.test_mode,scene_id,anno,dataset_idx)
            if row is None:
                raise KeyError(f"No depth override for {V.v4_dataset_type} idx={index}, scene={scene_id}, "
                               f"anno={anno}, split={cfgs.test_mode}, variant={variant}")
            rows[variant]=row
        h,w=batch["img"].shape[-2:]
        maps={v:_load_depth(rows[v],(h,w)) for v in variants}
        with torch.inference_mode(): base_ep=model(dict(batch))
        fixed=_common_seeds(base_ep,list(maps.values()),model)
        for variant in variants:
            depth=torch.as_tensor(maps[variant],device=device,dtype=batch["img"].dtype)[None]
            run=dict(batch)
            run["cva_fixed_token_sel_idx"]=fixed.unsqueeze(0)
            run["cva_depth_override"]=depth
            with torch.inference_mode():
                ep=model(run)
                decoded=pred_decode_center_view_angle(ep,use_cdf=bool(getattr(cfgs,"use_cdf",True)))
            if len(decoded)!=1: raise RuntimeError("V4 requires batch size 1")
            pred=decoded[0]
            if pred.shape[0]!=fixed.numel():
                raise RuntimeError(f"{variant}: decoded candidate count differs from fixed seed bank")
            row=rows[variant]
            oracle=str(row.get("oracle","False")).lower() in ("true","1","yes")
            pred_path,bank_path=_save(root,variant,scene,cfgs.camera,anno,pred,{
              "candidate_id":np.arange(pred.shape[0],dtype=np.int64),
              "token_sel_idx":ep["token_sel_idx"][0].detach().cpu().numpy(),
              "fixed_seed_count":np.asarray(fixed.numel()),
              "scene_id":np.asarray(scene_id),"anno_id":np.asarray(anno),
              "dataset_idx":np.asarray(dataset_idx),"variant":np.asarray(variant),
              "oracle":np.asarray(oracle),"alignment_method":np.asarray(row.get("alignment_method","")),
              "scale_alpha":np.asarray(float(row.get("scale_alpha",1.0))),
            })
            records.append({"dataset_type":V.v4_dataset_type,"split":cfgs.test_mode,
              "scene_id":scene_id,"anno_id":anno,"dataset_idx":dataset_idx,
              "variant":variant,"oracle":int(oracle),"fixed_seed_count":int(fixed.numel()),
              "candidate_count":int(pred.shape[0]),"prediction_path":pred_path,
              "candidate_bank_path":bank_path,"checkpoint_sha256":ckpt_sha,
              "alignment_method":row.get("alignment_method",""),
              "scale_alpha":row.get("scale_alpha","")})
        if n%20==0: print(f"[V4] samples={n+1}/{len(indices)} dataset={V.v4_dataset_type}",flush=True)
    if not records: raise RuntimeError("No V4 samples processed")
    fields=sorted({k for r in records for k in r})
    with open(root/"sample_manifest.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)
    with open(root/"run_config.json","w",encoding="utf-8") as f:
        json.dump({"dataset_type":V.v4_dataset_type,"dataset_module":V.v4_dataset_module,
          "dataset_class":V.v4_dataset_class,"dataset_kwargs":V.v4_dataset_kwargs_json,
          "variants":variants,"checkpoint_sha256":ckpt_sha,"sample_count":len(indices)},f,indent=2)
    print(f"[V4 DONE] samples={len(indices)} output={root}")

if __name__=="__main__": main()
