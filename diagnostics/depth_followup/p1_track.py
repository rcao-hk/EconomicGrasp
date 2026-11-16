"""P1 track the SAME measured 3D surface across GC6D's 13 camera views.

Frozen prediction maps are reused. Source anchor points are defined ONLY by
raw sensor depth/seg and verified GC6D C2W; target depth visibility and 3D
roundtrip are explicit gates. No re-training, no CAD labels and no grasp AP.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from diagnostics.frozen_depth_response.geometry import crop_K_from_raw
from .common import sha256,digest,write_json
from .multiview import (sample_anchors,match_pair,
                        verify_c2w_scene_poses)

SCHEMA="gc6d-same-surface-track-v1"
ANCHOR_ORDINALS=(0,6,12)


def load_cache(path,*,complete=True):
    p=Path(path)
    m=json.loads(p.read_text(encoding="utf-8"))
    signature=m.pop("manifest_sha256",None)
    if (signature!=digest(m) or
            m.get("schema")!="gc6d-surface-depth-cache-v1"):
        raise ValueError("Frozen P1 cache signature/schema mismatch")
    m["manifest_sha256"]=signature
    if complete and (not m["complete"] or m.get("frames")!=1014 or
                      m.get("scenes")!=78):
        raise ValueError("Formal P1 requires full 78-scene 1014-view frozen cache")
    if len(m["records"])!=m["frames"]:
        raise ValueError("Truncated P1 cache record collection")
    keyed=set()
    for r in m["records"]:
        key=(int(r["scene"]),int(r["img_id"]))
        if key in keyed:raise ValueError("Duplicate source cache frame")
        keyed.add(key)
    return m


def load_frame(rec):
    """Read and independently verify signed depth/raster/C2W source."""
    required=("rgb","depth","seg","camera","scene_gt")
    paths=rec["source_paths"]
    src=rec["source_sha256"]
    for key in required:
        path=Path(paths[key])
        if not path.is_file() or sha256(path)!=src[key]:
            raise RuntimeError(f"Frozen view source modified or missing: {path}")
    pred_path=Path(rec["prediction_file"])
    if not pred_path.is_file() or sha256(pred_path)!=rec["prediction_sha256"]:
        raise RuntimeError(f"Frozen 448 predicted depth changed: {pred_path}")
    with np.load(pred_path,allow_pickle=False) as z:
        pred=np.asarray(z["pred_depth"],dtype=np.float32)
    if pred.shape!=(448,448) or not np.isfinite(pred).all() or (pred<=0).any():
        raise ValueError("Invalid frozen predicted depth [448,448]")
    sensor=np.asarray(Image.open(paths["depth"]))
    seg=np.asarray(Image.open(paths["seg"]))
    if seg.ndim==3:seg=seg[:,:,0]
    if sensor.ndim!=2 or sensor.shape!=seg.shape:
        raise ValueError("Raw GC6D sensor and segmentation are not aligned")
    depth_factor=float(rec["depth_factor_divisor"])
    if not np.isfinite(depth_factor) or depth_factor<=0:
        raise ValueError("Invalid GC6D depth factor")
    z=sensor.astype(np.float32)/depth_factor
    K=np.asarray(rec["K"],dtype=np.float32).reshape(3,3)
    Kraw=np.asarray(rec["raw_K"],dtype=np.float32).reshape(3,3)
    box=tuple(map(int,rec["crop_box"]))
    matched,_=crop_K_from_raw(Kraw,box,(448,448))
    if np.max(np.abs(K-matched))>2e-3:
        raise RuntimeError("Cache K not the native K'=A K")
    T=np.asarray(rec["T_c2w"],dtype=np.float64).reshape(4,4)
    camera=json.loads(Path(paths["camera"]).read_text(encoding="utf-8"))
    key=str(int(rec["img_id"]))
    info=camera[key]
    R=np.asarray(info["cam_R_w2c"],dtype=np.float64).reshape(3,3)
    t=np.asarray(info["cam_t_w2c"],dtype=np.float64).reshape(3)/1000.
    if not np.allclose(T[:3,:3],R,atol=1e-5) or \
            not np.allclose(T[:3,3],t,atol=1e-5):
        raise RuntimeError("Native GC6D JSON C2W and prediction-cache pose diverge")
    kr=np.asarray(info["cam_K"],dtype=np.float32).reshape(3,3)
    if not np.allclose(Kraw,kr,atol=1e-5,rtol=0):
        raise RuntimeError("Native GC6D raw intrinsics changed")
    if tuple(rec["raw_hw"])!=z.shape:
        raise RuntimeError("Native GC6D image size changed")
    return dict(scene=int(rec["scene"]),img_id=int(rec["img_id"]),
                anno=int(rec["anno"]),pred_depth=pred,depth_raw_m=z,
                seg_raw=seg.astype(np.int32),raw_K=Kraw,K=K,
                crop_box=np.asarray(box,dtype=np.int32),T_c2w=T,
                source_prediction_sha256=rec["prediction_sha256"],
                source_camera_sha256=src["camera"])


def _write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        raise ValueError("Empty rows cannot be silently used as accepted tracks")
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def run_one_scene(records,a,output,cache_sha):
    scene=int(records[0]["scene"])
    ordered=sorted(records,key=lambda r:int(r["anno"]))
    if len(ordered)!=13 or len({int(r["anno"]) for r in ordered})!=13:
        raise RuntimeError(f"Missing/duplicate 13 GC6D scene views: {scene}")
    views=[load_frame(r) for r in ordered]
    scene_gt=json.loads(Path(ordered[0]["source_paths"]["scene_gt"]).read_text(encoding="utf-8"))
    consistency=verify_c2w_scene_poses(views,scene_gt,
                     translation_tol=a.c2w_translation_tol,
                     angle_tol_deg=a.c2w_rotation_tol)
    output=Path(output)
    csv_path=output/f"scene_{scene:06d}"/"matches.csv"
    stat_path=csv_path.with_name("audit.json")
    code_sha=sha256(__file__)
    contract=dict(scene=scene,cache_manifest_sha256=cache_sha,
                  source_predictions=[r["prediction_sha256"] for r in ordered],
                  tracker_code_sha256=code_sha,
                  multiview_geometry_code_sha256=sha256(Path(__file__).with_name("multiview.py")),
                  anchor_ordinal=list(ANCHOR_ORDINALS),
                  points_per_object=a.points_per_object,max_points=a.max_points,
                  visibility_abs_tol=a.visibility_abs_tol,
                  visibility_rel_tol=a.visibility_rel_tol,
                  xyz_roundtrip_tol=a.xyz_roundtrip_tol,
                  c2w_translation_tol=a.c2w_translation_tol,
                  c2w_rotation_tol_deg=a.c2w_rotation_tol)
    if csv_path.exists() or stat_path.exists():
        if not a.resume or not csv_path.is_file() or not stat_path.is_file():
            raise FileExistsError(f"Existing/incomplete scene matching: {scene}")
        audit=json.loads(stat_path.read_text(encoding="utf-8"))
        if (audit["contract_sha256"]!=digest(contract) or
             audit["matching_sha256"]!=sha256(csv_path)):
            raise RuntimeError("Cannot resume a different/edited scene match")
        return audit
    pairs=[]
    counters=defaultdict(int)
    for anchor_idx in ANCHOR_ORDINALS:
        src=views[anchor_idx]
        u,v,label=sample_anchors(src["depth_raw_m"],src["seg_raw"],
              scene=scene,img_id=src["img_id"],points_per_object=a.points_per_object,
              max_points=a.max_points)
        counters["sampled_anchor_points"]+=len(u)
        for dst in views:
            if int(dst["img_id"])==int(src["img_id"]):continue
            values,counts=match_pair(src,dst,u,v,label,
                abs_tol=a.visibility_abs_tol,rel_tol=a.visibility_rel_tol,
                xyz_tol=a.xyz_roundtrip_tol)
            for key,x in counts.items():
                counters[key]+=int(x)
            for item in values:
                item["scene"]=scene
                item["source_ordinal"]=anchor_idx
                pairs.append(item)
    if pairs:
        _write_csv(csv_path,pairs)
        have_data=True
        data_sha=sha256(csv_path)
    else:
        # Explicitly represent a geometry-overlap failure, never invent
        # matched pairs or a zero slope.
        csv_path.parent.mkdir(parents=True,exist_ok=True)
        csv_path.write_text("no_matched_points\n",encoding="utf-8")
        have_data=False
        data_sha=sha256(csv_path)
    audit=dict(schema=SCHEMA,scene=scene,views=13,
          anchor_ordinals=list(ANCHOR_ORDINALS),
          matched_pairs=len(pairs),has_matches=have_data,
          visibility_counts=dict(counters),c2w_scene_gt_parity=consistency,
          matching_file=str(csv_path.resolve()),matching_sha256=data_sha,
          contract=contract,contract_sha256=digest(contract),
          source="same measured GC6D 3D surface, occlusion/seg and roundtrip gated",
          limitations=[
            "Raw segmentation numeric equality is a conservative consistency check, not independently verified CAD instance identity",
            "GC6D C2W world-pose parity checked using available repeated object IDs",
            "All track anchors derived from sensor depth, not predicted depth or grasp labels",
            "No rendered GT; predictions are frozen RGB+pose metric depth",
            "The 78 GC6D scenes were already inspected; no independent validation claim",
          ])
    write_json(stat_path,audit)
    return audit


def run(a):
    if (a.points_per_object<1 or a.max_points<1 or
        a.visibility_abs_tol<=0 or a.visibility_rel_tol<=0 or
        a.xyz_roundtrip_tol<=0):
        raise ValueError("Invalid visible-point tracking protocol")
    mf=load_cache(a.cache_manifest,complete=not a.smoke)
    by_scene=defaultdict(list)
    for r in mf["records"]:by_scene[int(r["scene"])].append(r)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()) and not a.resume:
        raise FileExistsError("Use new output or explicit identical resume")
    root.mkdir(parents=True,exist_ok=True)
    scenes=sorted(by_scene)
    if a.limit_scenes:
        if not a.smoke or a.limit_scenes>=len(scenes):
            raise ValueError("Limited scenes only for separate smoke output")
        scenes=scenes[:a.limit_scenes]
    audits=[]
    for i,scene in enumerate(scenes,1):
        item=run_one_scene(by_scene[scene],a,root,mf["manifest_sha256"])
        audits.append(item)
        print(f"[P1 same-surface] scene {i}/{len(scenes)} id={scene} "
              f"matches={item['matched_pairs']}",flush=True)
    result=dict(schema=SCHEMA,
        complete=not a.smoke and not a.limit_scenes and len(audits)==78,
        source_cache_manifest_sha256=sha256(a.cache_manifest),
        cache_identity_sha256=mf["manifest_sha256"],
        matched_scenes=len(audits),
        scenes_with_matches=sum(x["has_matches"] for x in audits),
        total_pairs=sum(x["matched_pairs"] for x in audits),
        audits=audits,labels_used_for_sampling=False,
        model_retrained=False,official_grasp_AP_evaluated=False)
    result["manifest_sha256"]=digest(result)
    p=root/"manifest.json"
    if p.exists():
        if json.loads(p.read_text(encoding="utf-8"))!=result:
            raise RuntimeError("Resume match manifest contract differs")
    else:
        write_json(p,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--points-per-object",type=int,default=64)
    p.add_argument("--max-points",type=int,default=512)
    p.add_argument("--visibility-abs-tol",type=float,default=.008)
    p.add_argument("--visibility-rel-tol",type=float,default=.012)
    p.add_argument("--xyz-roundtrip-tol",type=float,default=.018)
    p.add_argument("--c2w-translation-tol",type=float,default=.025)
    p.add_argument("--c2w-rotation-tol",type=float,default=5.)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--limit-scenes",type=int,default=0)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    run(a)


if __name__=="__main__":
    main()
