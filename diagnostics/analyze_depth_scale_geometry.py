#!/usr/bin/env python3
"""V4: separate global metric-scale bias from local geometry degradation.

CSV columns: dataset,split,scene_id,anno_id,pred_depth_path,gt_depth_path,
foreground_mask_path,background_mask_path,K_json.
Optional calibrated-plane columns: support_plane_json or support_plane_path.
GT median-ratio and GT-fitted support-plane alignment are ORACLE diagnostics.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from diagnostics.rgb_only_validation_geometry import (
    align_global_median_ratio,align_support_plane,depth_geometry_metrics,
    fit_plane_svd,plane_depth_map,
)

def _load_mask(path,shape,fallback=None):
    if path:
      m=np.load(path)
      if m.shape!=shape: raise ValueError(f"Mask shape {m.shape} != depth shape {shape}: {path}")
      return m.astype(bool)
    if fallback is None: raise FileNotFoundError("Foreground/background mask path is required")
    return np.asarray(fallback,dtype=bool)

def _load_plane(row):
    value=row.get("support_plane_json","").strip()
    if value: return np.asarray(json.loads(value),dtype=np.float64)
    path=row.get("support_plane_path","").strip()
    if path: return np.asarray(np.load(path),dtype=np.float64).reshape(4)
    raise ValueError("calibrated mode requires support_plane_json or support_plane_path")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--plane-source",choices=("none","gt_fit","calibrated"),default="gt_fit")
    p.add_argument("--patch-size",type=int,default=16)
    p.add_argument("--min-depth",type=float,default=.15)
    p.add_argument("--max-depth",type=float,default=2.)
    p.add_argument("--seed",type=int,default=0)
    args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    aligned_root=out/"aligned_depth"; aligned_root.mkdir(parents=True,exist_ok=True)
    rows_out=[]; override_rows=[]
    with open(args.manifest,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("Input manifest is empty")
    for row in rows:
      pred=np.asarray(np.load(row["pred_depth_path"]),dtype=np.float32)
      gt=np.asarray(np.load(row["gt_depth_path"]),dtype=np.float32)
      if pred.shape!=gt.shape or pred.ndim!=2: raise ValueError("Depth shape mismatch")
      K=np.asarray(json.loads(row["K_json"]),dtype=np.float64).reshape(3,3)
      fg=_load_mask(row.get("foreground_mask_path",""),pred.shape,fallback=(gt>0))
      bg=_load_mask(row.get("background_mask_path",""),pred.shape,fallback=(~fg)&(gt>0))
      valid=np.isfinite(pred)&np.isfinite(gt)&(pred>=args.min_depth)&(pred<=args.max_depth)&(gt>=args.min_depth)&(gt<=args.max_depth)
      fg &= valid; bg &= valid
      ds=row.get("dataset","dataset"); split=row.get("split","split")
      scene=int(row["scene_id"]); anno=int(row["anno_id"])
      variants=[("raw",pred,1.,False,"none")]
      g_aligned,alpha=align_global_median_ratio(pred,gt,fg)
      variants.append(("global_median_oracle",g_aligned,alpha,True,"gt_foreground_median_ratio"))
      plane_stats={}
      if args.plane_source!="none":
        if args.plane_source=="gt_fit":
          plane,plane_stats=fit_plane_svd(gt,K,bg,seed=args.seed); oracle=True
        else:
          plane=_load_plane(row); oracle=False
        pz,pvalid=plane_depth_map(K,*pred.shape,plane)
        paligned,palpha=align_support_plane(pred,pz,bg&pvalid)
        variants.append(("support_plane_aligned",paligned,palpha,oracle,
                         "gt_fitted_plane" if oracle else "calibrated_plane"))
      for name,depth,scale,oracle,method in variants:
        metrics=depth_geometry_metrics(depth,gt,K,mask=fg,patch_size=args.patch_size)
        target=aligned_root/name/ds/split; target.mkdir(parents=True,exist_ok=True)
        path=target/f"scene{scene:04d}_anno{anno:04d}.npy"; np.save(path,depth.astype(np.float32))
        record={"dataset":ds,"split":split,"scene_id":scene,"anno_id":anno,
          "variant":name,"alignment_method":method,"scale_alpha":float(scale),
          "oracle":bool(oracle),
          "oracle_note":"uses GT depth or GT-fitted support plane" if oracle else "uses externally calibrated support plane",
          **metrics,**{f"plane_{k}":v for k,v in plane_stats.items()}}
        rows_out.append(record)
        override_rows.append({"dataset":ds,"split":split,"scene_id":scene,"anno_id":anno,
          "variant":name,"depth_path":str(path.resolve()),"oracle":bool(oracle),
          "alignment_method":method,"scale_alpha":float(scale)})
    def write_csv(path,records):
      fields=sorted({k for r in records for k in r})
      with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)
    write_csv(out/"depth_scale_geometry_summary.csv",rows_out)
    write_csv(out/"depth_override_manifest.csv",override_rows)
    with open(out/"manifest.json","w",encoding="utf-8") as f:
      json.dump({"input_manifest":str(Path(args.manifest).resolve()),
        "plane_source":args.plane_source,"global_median_alignment_is_oracle":True,
        "gt_fit_plane_alignment_is_oracle":args.plane_source=="gt_fit",
        "calibrated_plane_alignment_is_oracle":False,
        "warning":"Depth metrics are diagnostic. Run grasp inference with each override manifest and the official evaluator for AP/collision/success."},f,indent=2)
    print(f"[DONE] samples={len(rows)} output={out}")

if __name__=="__main__": main()
