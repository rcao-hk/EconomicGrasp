#!/usr/bin/env python3
"""P0-B: distinguish cache coverage from actual model-matched CVA-CDF supervision.

offline: CPU scan of GN and filtered GC6D scene caches and positive graspness.
forward: optional single-GPU frozen checkpoint probing of VALID MATCH MASKS and
losses. GN-Trans reuses GN scene-level geometric CDF cache; it is NOT a new
independent geometry source. No optimizer steps and no model modifications.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_p0():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--p0_gc6d_root", required=True)
    p.add_argument("--p0_gntrans_root", default=None)
    p.add_argument("--p0_gc6d_split_json", required=True)
    p.add_argument("--p0_output_dir", required=True)
    p.add_argument("--p0_mode", choices=("offline", "forward", "both"), default="offline")
    p.add_argument("--p0_max_cache_scenes", type=int, default=0)
    p.add_argument("--p0_max_forward_frames", type=int, default=24)
    p.add_argument("--p0_seed", type=int, default=0)
    p.add_argument("--p0_gc6d_graspness_mode", choices=("scene","instance"), default="instance")
    p.add_argument("--p0_graspnet_cdf_folder",
                   default="economic_grasp_label_300views_extend_angle_cdf_depth")
    p.add_argument("--p0_gc6d_cdf_folder",
                   default="economic_grasp_label_300views_extend_angle_cdf_depth")
    p.add_argument("--p0_forward_num_workers", type=int, default=0)
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    return args


P0 = parse_p0()
from utils.arguments import cfgs


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_scene_ids():
    file = Path(P0.p0_gc6d_split_json)
    ids = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or len(ids) != 232 or len(set(map(int, ids))) != 232:
        raise ValueError("GC6D filtered scene list must contain exactly 232 unique IDs.")
    return sorted(map(int, ids))


def scene_label(root, folder, stem):
    path = Path(root) / folder / (stem + "_labels.npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def scan_cdf_cache(path, dataset, scene, source):
    with np.load(path, allow_pickle=False) as x:
        for k in ("cdf_bins","width_valids_depth","pointid","points","obj_ids" if dataset == "gc6d" else "topview"):
            if k not in x:
                raise KeyError(f"{k} absent from {path}")
        points = x["points"]
        ids = x["pointid"]
        if len(ids) != len(points) or len(points) == 0:
            raise ValueError(f"Invalid points/pointid cache at {path}")
        bins = x["cdf_bins"]
        valid = x["width_valids_depth"]
        if bins.shape != valid.shape or bins.ndim != 4 or bins.shape[0] != len(points):
            raise ValueError(f"CDF and width mask shape mismatch in {path}")
        if bins.shape[-2:] != (12,4) or bins.max() > 6:
            raise ValueError(f"CDF schema mismatch in {path}")
        uniq = np.unique(ids)
        if dataset == "gc6d":
            n_obj = len(x["obj_ids"])
            if uniq.max() >= n_obj:
                raise ValueError(f"GC6D local pointid mismatch in {path}")
        else:
            n_obj = len(uniq)  # GN scene may include objects without retained grasp points.
        histogram = np.bincount(bins.ravel(), minlength=7)[:7]
        all_num = int(histogram.sum())
        pos_num = int(all_num - histogram[0])
        valid_num = int(np.count_nonzero(valid))
        return dict(source=source, scene=scene, cache=str(path),
            num_points=int(len(points)), active_objects=int(len(uniq)),
            num_objects_with_pose=int(n_obj), unmatched_objects_upper_bound=
            (n_obj-len(uniq)) if dataset == "gc6d" else None,
            cdf_total_candidates=all_num, cdf_positive_candidates=pos_num,
            cdf_positive_ratio=float(pos_num/max(all_num,1)),
            width_valid_candidates=valid_num, width_valid_ratio=float(valid_num/max(all_num,1)),
            **{f"cdf_bin_{i}":int(histogram[i]) for i in range(7)})


def summarize_numeric(rows, keys):
    out = {}
    for key in keys:
        vals = np.asarray([r[key] for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))],np.float64)
        out[key] = dict(mean=float(vals.mean()), p10=float(np.percentile(vals,10)),
                        p50=float(np.median(vals)), p90=float(np.percentile(vals,90))) if len(vals) else None
    return out


def offline(root, ids, outdir):
    rows = []
    gnroot = Path(cfgs.dataset_root)
    # Shared GraspNet/GNTrans grasp-geometry labels. Scan once, report under
    # both dataset names with identical source cache fingerprints.
    gn_scenes = list(range(100))
    if P0.p0_max_cache_scenes:
        gn_scenes = gn_scenes[:P0.p0_max_cache_scenes]
        ids = ids[:P0.p0_max_cache_scenes]
    for sceneid in gn_scenes:
        scene = f"scene_{sceneid:04d}"
        path = scene_label(gnroot, P0.p0_graspnet_cdf_folder, scene)
        g = scan_cdf_cache(path,"gn",scene,"graspnet")
        rows.append(g)
        rows.append(dict(g, source="gntrans"))
    for sceneid in ids:
        scene = f"{sceneid:06d}"
        path = scene_label(root, P0.p0_gc6d_cdf_folder,scene)
        rows.append(scan_cdf_cache(path,"gc6d",scene,"gc6d"))
    write_csv(outdir/"cache_scene_coverage.csv",rows)
    result = {}
    for source in ("graspnet","gntrans","gc6d"):
        items = [x for x in rows if x["source"]==source]
        s = sum(x["cdf_positive_candidates"] for x in items)
        total = sum(x["cdf_total_candidates"] for x in items)
        result[source] = dict(n_scenes=len(items),total_points=sum(x["num_points"] for x in items),
            sum_active_objects=sum(x["active_objects"] for x in items),
            cdf_micro_positive_ratio=s/max(total,1),
            width_micro_valid_ratio=sum(x["width_valid_candidates"] for x in items)/max(total,1),
            scene_stats=summarize_numeric(items,["cdf_positive_ratio","width_valid_ratio","num_points","active_objects"]))
    # Do not equate 'cache positives' with actual matched Q×A×D labels.
    return rows,result


def even_indices(dataset, count, seed):
    rng = np.random.default_rng(seed)
    group = defaultdict(list)
    for idx,(scene,frame) in enumerate(zip(dataset.scenename,dataset.frameid)):
        group[str(scene)].append(idx)
    scenes = sorted(group)
    rng.shuffle(scenes)
    # One frame from each scene before repeating scene families; reduces leakage
    # from 256 neighboring GraspNet camera views.
    selected = []
    for scene in scenes:
        ids = group[scene]
        selected.append(ids[int(rng.integers(len(ids)))])
        if len(selected)>=count: break
    if len(selected)<count:
        for scene in scenes:
            remaining = [x for x in group[scene] if x not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[:max(0,count-len(selected))])
            if len(selected)>=count:break
    return selected[:count]


def make_datasets(gcroot, ids):
    from dataset.graspnet_dataset import GraspNetMultiDataset, GraspNetTransDataset
    from dataset.gc6d_dataset import GraspClutter6DMultiDataset
    from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
    if not P0.p0_gntrans_root:
        raise ValueError("--p0_gntrans_root required for forward mode")
    common = dict(camera=cfgs.camera, split="train", num_points=cfgs.num_point,
                  voxel_size=cfgs.voxel_size, min_depth=cfgs.min_depth,
                  max_depth=cfgs.max_depth, bin_num=cfgs.bin_num, depth_strides=1)
    gn = GraspNetMultiDataset(cfgs.dataset_root,remove_outlier=True, augment=False,
        load_label=True, load_grasp_payload=False,extend_angle=True,
        use_gt_depth=False,use_fuse_depth=True,graspness_mode="scene",**common)
    trans = GraspNetTransDataset(cfgs.dataset_root,P0.p0_gntrans_root,
        remove_outlier=True,augment=False,load_label=True,use_gt_depth=True,**common)
    trans.extend_angle=True
    trans.load_grasp_payload=False
    trans.use_fuse_depth=False   # Historical mixed-v2 GN-Trans used rendered depth.
    gc = GraspClutter6DMultiDataset(root=str(gcroot),camera="realsense-d435",
        split="train",scene_ids=ids,max_ann=13,num_points=cfgs.num_point,
        voxel_size=cfgs.voxel_size,load_label=True,load_grasp_payload=False,
        graspness_mode=P0.p0_gc6d_graspness_mode, use_gt_depth=False,
        use_fuse_depth=False,extend_angle=True,mask_mode="workspace_depth",
        remove_outlier=None,workspace_pose_mode="json",
        economic_label_root=str(gcroot/P0.p0_gc6d_cdf_folder),
        scene_name_format="plain6",min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,bin_num=cfgs.bin_num,depth_strides=1)
    return dict(
        graspnet=CVAExtendedLabelAdapter(gn,cfgs.dataset_root,use_cdf=True,
            label_folder=P0.p0_graspnet_cdf_folder),
        gntrans=CVAExtendedLabelAdapter(trans,cfgs.dataset_root,use_cdf=True,
            label_folder=P0.p0_graspnet_cdf_folder),
        gc6d=CVAExtendedLabelAdapter(gc,str(gcroot),use_cdf=True,
            label_folder=P0.p0_gc6d_cdf_folder))


def forward(root, ids, outdir):
    import torch
    from models.economicgrasp_bip3d import economicgrasp_dpt
    from models.grasp_designs import design_options, validate_checkpoint_design
    from models.loss_economicgrasp_depth_kview_transformer import get_loss
    from train_cva_ddp import validate_batch_label_contract,move_batch_to_device
    from dataset.graspnet_dataset import collate_fn
    if not cfgs.checkpoint_path:
        raise ValueError("--checkpoint_path required for forward diagnostics")
    if not torch.cuda.is_available():
        raise RuntimeError("P0-B forward mode expects one CUDA GPU. Use offline on CPU.")
    ck = torch.load(cfgs.checkpoint_path,map_location="cpu",weights_only=False)
    model=economicgrasp_dpt(**design_options(cfgs),min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,bin_num=cfgs.bin_num,is_training=True,
        use_obs_depth=False,pose_depth_mode=cfgs.pose_depth_mode,use_cdf=True)
    validate_checkpoint_design(model,ck)
    state = ck.get("model_state_dict",ck.get("state_dict",ck)) if isinstance(ck,dict) else ck
    model.load_state_dict(state,strict=True)
    model.cuda().eval()
    from torch.utils.data import DataLoader, Subset
    datasets=make_datasets(root,ids)
    rows=[]
    device=torch.device("cuda:0")
    for source,ds in datasets.items():
        indices=even_indices(ds,P0.p0_max_forward_frames,P0.p0_seed)
        loader=DataLoader(Subset(ds,indices),batch_size=1,shuffle=False,
            num_workers=P0.p0_forward_num_workers,collate_fn=collate_fn)
        for i,batch in enumerate(loader):
            validate_batch_label_contract(batch,use_cdf=True)
            batch=move_batch_to_device(batch,device,use_cdf=True)
            batch["cva_compute_diagnostics"]=False
            batch["geometry_compute_diagnostics"]=False
            batch["cva_export_angle_feature"]=False
            with torch.no_grad():
                ep=model(batch)
                _,ep=get_loss(ep,use_cdf=True)
            valid=ep["batch_grasp_cdf_valid_mask"].bool()
            bins=ep["batch_grasp_cdf_bins_angle_depth"]
            qvalid=valid.any(dim=(-1,-2))
            stats=dict(source=source, dataset_index=indices[i],
                scene=str(ds.scenename[indices[i]]),frame=int(ds.frameid[indices[i]]),
                num_queries=int(valid.shape[1]),
                matched_queries=int(qvalid.sum().item()),
                matched_query_ratio=float(qvalid.float().mean().item()),
                matched_action_ratio=float(valid.float().mean().item()),
                cdf_positive_valid_ratio=float(((bins>0)&valid).sum().item()/max(valid.sum().item(),1)),
                valid_action_count=int(valid.sum().item()))
            for key in ("B: Objectness Loss","B: Graspness Loss","B: View Loss",
                        "B: CDF Loss","B: Width Depth Loss","B: DepthReg Loss",
                        "A: Overall Loss","D: Depth MAE"):
                if key in ep and torch.is_tensor(ep[key]) and ep[key].numel()==1:
                    stats[key.replace(": ","_").replace(" ","_")]=float(ep[key].detach().item())
            rows.append(stats)
            print(f"[P0-B] {source} {i+1}/{len(indices)} matched={stats['matched_query_ratio']:.3f}",flush=True)
    write_csv(outdir/"forward_matched_frames.csv",rows)
    return {source:dict(samples=len([r for r in rows if r["source"]==source]),
       stats=summarize_numeric([r for r in rows if r["source"]==source],
           ["matched_query_ratio","matched_action_ratio","cdf_positive_valid_ratio",
            "valid_action_count"])) for source in datasets}


def main():
    outdir=Path(P0.p0_output_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    root=Path(P0.p0_gc6d_root).expanduser().resolve()
    ids=load_scene_ids()
    report=dict(mode=P0.p0_mode,graspnet_root=str(Path(cfgs.dataset_root).resolve()),
        gc6d_root=str(root),gc6d_filtered_scene_count=len(ids),
        gntrans_shares_graspnet_grasp_cache=True,
        offline_interpretation="Cache positive coverage is NOT a model-matched query rate.",
        actual_matching_interpretation="Forward-mode valid mask is generated after checkpoint-dependent center/query matching.",
        seed=P0.p0_seed)
    if P0.p0_mode in ("offline","both"):
        _,report["cache"]=offline(root,ids,outdir)
    if P0.p0_mode in ("forward","both"):
        report["forward"]=forward(root,ids,outdir)
    (outdir/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(status="complete",report=str(outdir/"summary.json")),indent=2))


if __name__=="__main__":
    main()
