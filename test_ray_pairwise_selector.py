#!/usr/bin/env python3
"""Test a trained relative-advantage ray selector on GraspNet splits.

The selector never sees test exact-action labels.  It chooses among frozen K
center hypotheses using cached training-derived weights and the native fallback
threshold stored in its checkpoint.  CAD/DexNet is called only after selection
to report native/raw/learned/oracle diagnostic outcomes.
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


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--checkpoint_path", required=True, help="Frozen Stage-1 checkpoint")
    p.add_argument("--selector_checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="test_seen", choices=("test_seen", "test_similar", "test_novel"))
    p.add_argument("--camera", default="realsense")
    p.add_argument("--sample_interval", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_points", type=int, default=20000)
    p.add_argument("--min_depth", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=1.0)
    p.add_argument("--bin_num", type=int, default=256)
    p.add_argument("--pose_depth_mode", default="global_film", choices=("none", "global_film", "ray_gravity_film"))
    p.add_argument("--offsets_mm", default="-40,-20,-10,0,10,20,40")
    p.add_argument("--query_eval_num", type=int, default=128)
    p.add_argument("--query_eval_mode", default="topk_uniform", choices=("all", "topk", "uniform", "topk_uniform"))
    p.add_argument("--selector_threshold", type=float, default=None, help="Override checkpoint threshold")
    p.add_argument("--fc_mode", default="reuse_contacts", choices=("reuse_contacts", "official"))
    p.add_argument("--verify_n", type=int, default=0)
    p.add_argument("--noop_check_samples", type=int, default=2)
    p.add_argument("--noop_atol", type=float, default=5e-5)
    p.add_argument("--profile_timing", action="store_true")
    p.add_argument("--save_candidate_rows", action="store_true")
    return p.parse_args()


ARGS = _parse_args()
sys.argv = [sys.argv[0]]

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
from utils.ray_pairwise_selector import (
    RayPairwiseSelector,
    compose_pairwise_features,
    extract_action_conditioned_features,
    select_with_native_fallback,
)


def _configure_cfg():
    cfgs.use_top4_view_infer = False
    cfgs.kview_mode = "A1"
    cfgs.kview_k = 1
    cfgs.use_cdf = True
    cfgs.use_obs_depth = False
    cfgs.pose_depth_mode = ARGS.pose_depth_mode


def _load_stage1(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    result = model.load_state_dict(state, strict=False)
    optional = ("rgb_geometry_diagnostics.",)
    missing = [k for k in result.missing_keys if not k.startswith(optional)]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(optional)]
    if missing or unexpected:
        raise RuntimeError(f"Stage-1 checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def _subset_indices(total, interval, max_samples):
    stride = max(1, int(round(1.0 / interval)))
    out = []
    for start in range(0, total, 256):
        out.extend(range(start, min(start + 256, total), stride))
    return out[:max_samples] if max_samples > 0 else out


def _move_batch(batch, device):
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=False)
        elif isinstance(v, (list, tuple)):
            raise TypeError(f"Unexpected list-valued key {k!r}.")
    return batch


def _uniform_pick(indices, count):
    if count <= 0 or count >= len(indices):
        return indices
    pos = torch.round(torch.linspace(0, len(indices) - 1, steps=count, device=indices.device)).long()
    pos = torch.unique(pos, sorted=True)
    if len(pos) < count:
        used = torch.zeros(len(indices), dtype=torch.bool, device=indices.device); used[pos] = True
        fill = torch.nonzero(~used, as_tuple=False).squeeze(1)[: count - len(pos)]
        pos = torch.sort(torch.cat((pos, fill))).values
    return indices.index_select(0, pos[:count])


def _select_queries(native, count, mode):
    n = native.shape[0]
    ids = torch.arange(n, device=native.device)
    if count <= 0 or count >= n or mode == "all":
        return ids
    score = native[:, 0]
    if mode == "topk":
        return torch.argsort(score, descending=True, stable=True)[:count]
    if mode == "uniform":
        return _uniform_pick(ids, count)
    n_top = count // 2
    ranked = torch.argsort(score, descending=True, stable=True)
    return torch.cat((ranked[:n_top], _uniform_pick(ranked[n_top:], count - n_top)))


def _success(f, threshold):
    f = np.asarray(f, dtype=np.float32)
    return np.isfinite(f) & (f > 0.0) & (f <= threshold + 1e-6)


def _evaluate_grid(evaluator, scene_id, anno_id, grasps, valid):
    K, N, _ = grasps.shape
    flat = grasps.reshape(K * N, 17)
    vf = valid.reshape(K * N)
    ids = np.flatnonzero(vf)
    t0 = time.perf_counter()
    r = evaluator.evaluate(scene_id, anno_id, flat[ids])
    elapsed = time.perf_counter() - t0
    friction = np.full(K * N, np.nan, np.float32)
    assigned = np.full(K * N, -1, np.int64)
    coll = np.full(K * N, -1, np.int8)
    pure = np.full(K * N, -1, np.int8)
    empty = np.full(K * N, -1, np.int8)
    friction[ids] = r.friction; assigned[ids] = r.assigned_obj
    coll[ids] = r.collision_or_empty.astype(np.int8)
    pure[ids] = r.pure_collision.astype(np.int8); empty[ids] = r.empty.astype(np.int8)
    return {
        "friction": friction.reshape(K, N),
        "assigned_obj": assigned.reshape(K, N),
        "collision_or_empty": coll.reshape(K, N),
        "pure_collision": pure.reshape(K, N),
        "empty": empty.reshape(K, N),
    }, {
        "eval_sec": elapsed,
        "collision_sec": float(r.stats.get("collision_sec", 0.0)),
        "force_closure_sec": float(r.stats.get("force_closure_sec", 0.0)),
    }


def _selected(mats, k):
    f = gather_kn(mats["friction"], k)
    return {
        "friction": f,
        "utility": friction_utility(f),
        "success04": _success(f, 0.4),
        "success08": _success(f, 0.8),
        "collision_or_empty": gather_kn(mats["collision_or_empty"], k),
        "pure_collision": gather_kn(mats["pure_collision"], k),
        "empty": gather_kn(mats["empty"], k),
        "assigned_obj": gather_kn(mats["assigned_obj"], k),
    }


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    _configure_cfg()
    offsets = parse_offsets_mm(ARGS.offsets_mm)
    zero = [i for i, x in enumerate(offsets) if abs(x) < 1e-9]
    if len(zero) != 1:
        raise ValueError("Exactly one zero offset required.")
    zero_k = zero[0]
    out = Path(ARGS.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    selector_ckpt = torch.load(ARGS.selector_checkpoint, map_location=device)
    selector = RayPairwiseSelector(
        int(selector_ckpt["feature_dim"]),
        int(selector_ckpt["hidden_dim"]),
        float(selector_ckpt["dropout"]),
    ).to(device)
    selector.load_state_dict(selector_ckpt["selector_state_dict"])
    selector.eval()
    feat_mean = selector_ckpt["feature_mean"].to(device).float()
    feat_std = selector_ckpt["feature_std"].to(device).float()
    threshold = float(selector_ckpt.get("selector_threshold", 0.0) if ARGS.selector_threshold is None else ARGS.selector_threshold)

    dataset = GraspNetMultiDataset(
        ARGS.dataset_root, split=ARGS.split, camera=ARGS.camera, num_points=ARGS.num_points,
        remove_outlier=True, augment=False, load_label=False, use_gt_depth=False,
        min_depth=ARGS.min_depth, max_depth=ARGS.max_depth, bin_num=ARGS.bin_num,
    )
    indices = _subset_indices(len(dataset), ARGS.sample_interval, ARGS.max_samples)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=ARGS.num_workers,
        collate_fn=collate_fn, pin_memory=False, persistent_workers=(ARGS.num_workers > 0),
    )
    model = economicgrasp_dpt_student(
        min_depth=ARGS.min_depth, max_depth=ARGS.max_depth, bin_num=ARGS.bin_num,
        is_training=False, use_obs_depth=False, pose_depth_mode=ARGS.pose_depth_mode,
        use_cdf=True, vis_dir=None,
    ).to(device)
    _load_stage1(model, ARGS.checkpoint_path); model.eval()
    evaluator = ExactGraspNetActionEvaluator(
        ARGS.dataset_root, ARGS.camera, split=ARGS.split,
        fc_mode=ARGS.fc_mode, verify_n=ARGS.verify_n, strict=True,
    )

    rows = []
    sample_rows = []
    timing = defaultdict(float)
    noop_max = defaultdict(float)
    learned_hist = Counter(); raw_hist = Counter(); oracle_hist = Counter()
    cand_file = None; cand_writer = None
    if ARGS.save_candidate_rows:
        cand_file = gzip.open(out / "per_candidate.csv.gz", "wt", newline="")
        fields = ["split","scene_id","anno_id","query_id","offset_mm","valid","raw_score","pred_delta","utility","friction","selected_learned","selected_raw","selected_oracle"]
        cand_writer = csv.DictWriter(cand_file, fieldnames=fields); cand_writer.writeheader()

    try:
        for local_i, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            batch["cva_export_angle_feature"] = False; batch["cva_compute_diagnostics"] = False; batch["geometry_compute_diagnostics"] = False
            t0 = time.perf_counter()
            with torch.inference_mode():
                ep = model(batch)
                native_pred = pred_decode_center_view_angle(ep, use_cdf=True)[0]
            native_sec = time.perf_counter() - t0
            native_xyz = ep["kview_base_xyz_graspable"].float(); token_idx = ep["kview_base_token_sel_idx"].long()
            H, W = ep["depth_map_used_for_geometry"].shape[-2:]
            centers, valid_center = build_ray_center_hypotheses(native_xyz, token_idx, ep["K"], (H,W), offsets, ARGS.min_depth, ARGS.max_depth)
            qidx = _select_queries(native_pred, ARGS.query_eval_num, ARGS.query_eval_mode)

            grasps_k=[]; sel_k=[]; mean_k=[]
            t1=time.perf_counter()
            with torch.inference_mode():
                for k in range(len(offsets)):
                    epk, grouped = rerun_cdf_with_read_center(model, ep, read_center=centers[k], output_center=centers[k])
                    if local_i < ARGS.noop_check_samples and k == zero_k:
                        m = assert_native_reread_equivalent(ep, epk, atol=ARGS.noop_atol)
                        replay = pred_decode_center_view_angle(epk, use_cdf=True)[0]
                        m["noop_decoded_max_abs"] = float((replay-native_pred).abs().max().item())
                        if m["noop_decoded_max_abs"] > ARGS.noop_atol:
                            raise RuntimeError("No-op decoded replay failed.")
                        for name,val in m.items(): noop_max[name]=max(noop_max[name],float(val))
                    pred = native_pred if k==zero_k else pred_decode_center_view_angle(epk,use_cdf=True)[0]
                    sel, mean, _ = extract_action_conditioned_features(grouped, epk)
                    grasps_k.append(pred.index_select(0,qidx)); sel_k.append(sel[0].index_select(0,qidx)); mean_k.append(mean[0].index_select(0,qidx))
            reread_sec=time.perf_counter()-t1

            grasps=torch.stack(grasps_k).detach().cpu().numpy().astype(np.float32)
            sel=torch.stack(sel_k); mean=torch.stack(mean_k)
            raw=torch.from_numpy(grasps[:,:,0]).to(device)
            offs=torch.as_tensor(offsets,device=device,dtype=sel.dtype)
            pair=compose_pairwise_features(sel,mean,raw,offs,zero_k)
            if pair.shape[-1] != selector.feature_dim:
                raise RuntimeError(f"Selector feature mismatch {pair.shape[-1]} vs {selector.feature_dim}")
            with torch.inference_mode():
                pred_delta=selector(((pair-feat_mean)/feat_std.clamp_min(1e-5)).reshape(-1,pair.shape[-1])).reshape(len(offsets),-1)
                pred_delta[zero_k]=0.0
            pred_delta_np=pred_delta.detach().cpu().numpy()
            valid=valid_center[:,0].index_select(1,qidx).detach().cpu().numpy().astype(bool)
            learned_k=select_with_native_fallback(pred_delta_np,valid,zero_k,threshold)

            scene_id=int(batch["scene_idx"].reshape(-1)[0]); anno_id=int(batch["anno_idx"].reshape(-1)[0])
            mats, et = _evaluate_grid(evaluator,scene_id,anno_id,grasps,valid)
            utility=friction_utility(mats["friction"]); raw_score=grasps[:,:,0]
            raw_k=select_raw_score(raw_score,valid); oracle_k=select_exact_oracle(utility,raw_score,valid)
            native_k=np.full(len(learned_k),zero_k,np.int64)
            mets={name:_selected(mats,k) for name,k in (("native",native_k),("raw",raw_k),("learned",learned_k),("oracle",oracle_k))}
            qids=qidx.detach().cpu().numpy().astype(np.int64)
            for j,qid in enumerate(qids):
                row={"split":ARGS.split,"scene_id":scene_id,"anno_id":anno_id,"query_id":int(qid),"selector_threshold":threshold,"learned_offset_mm":float(offsets[int(learned_k[j])]),"raw_offset_mm":float(offsets[int(raw_k[j])]),"oracle_offset_mm":float(offsets[int(oracle_k[j])]),"predicted_advantage":float(pred_delta_np[int(learned_k[j]),j]) if learned_k[j]!=zero_k else 0.0}
                for name,k in (("native",native_k),("raw",raw_k),("learned",learned_k),("oracle",oracle_k)):
                    m=mets[name]
                    for key in ("utility","success04","success08","collision_or_empty","pure_collision","empty","assigned_obj"):
                        row[f"{name}_{key}"]=float(m[key][j]) if key=="utility" else int(m[key][j])
                row["learned_rescue08"]=int((not row["native_success08"]) and row["learned_success08"])
                row["learned_harm08"]=int(row["native_success08"] and (not row["learned_success08"]))
                row["raw_rescue08"]=int((not row["native_success08"]) and row["raw_success08"])
                row["raw_harm08"]=int(row["native_success08"] and (not row["raw_success08"]))
                row["learned_matches_oracle"]=int(learned_k[j]==oracle_k[j]); row["raw_matches_oracle"]=int(raw_k[j]==oracle_k[j])
                rows.append(row)
                learned_hist[float(offsets[int(learned_k[j])])]+=1; raw_hist[float(offsets[int(raw_k[j])])]+=1; oracle_hist[float(offsets[int(oracle_k[j])])]+=1
                if cand_writer:
                    for k,o in enumerate(offsets):
                        cand_writer.writerow({"split":ARGS.split,"scene_id":scene_id,"anno_id":anno_id,"query_id":int(qid),"offset_mm":float(o),"valid":int(valid[k,j]),"raw_score":float(raw_score[k,j]),"pred_delta":float(pred_delta_np[k,j]),"utility":float(utility[k,j]),"friction":float(mats["friction"][k,j]),"selected_learned":int(k==learned_k[j]),"selected_raw":int(k==raw_k[j]),"selected_oracle":int(k==oracle_k[j])})

            sample={"split":ARGS.split,"scene_id":scene_id,"anno_id":anno_id,"num_queries":len(qids),"native_forward_sec":native_sec,"reread_sec":reread_sec,"exact_eval_sec":et["eval_sec"]}
            for name in mets:
                sample[f"{name}_utility"]=float(mets[name]["utility"].mean()); sample[f"{name}_success08"]=float(mets[name]["success08"].mean())
            sample_rows.append(sample)
            timing["native_forward_sec"]+=native_sec; timing["reread_sec"]+=reread_sec
            for k,v in et.items(): timing[k]+=float(v)
            if local_i%20==0:
                print(f"[PAIR-TEST] {local_i+1}/{len(indices)} scene={scene_id:04d} anno={anno_id:04d} native08={sample['native_success08']:.3f} learned08={sample['learned_success08']:.3f} oracle08={sample['oracle_success08']:.3f}",flush=True)
    finally:
        if cand_file: cand_file.close()

    _write_csv(out/"per_query.csv",rows); _write_csv(out/"per_sample_summary.csv",sample_rows)
    def agg(name):
        return {metric:float(np.mean([r[f"{name}_{metric}"] for r in rows])) for metric in ("utility","success04","success08","collision_or_empty","pure_collision","empty")}
    summary={
        "protocol":"pairwise exact-action relative-advantage selector v1",
        "split":ARGS.split,"num_samples":len(sample_rows),"num_queries":len(rows),"threshold":threshold,
        "selector_checkpoint":os.path.abspath(ARGS.selector_checkpoint),"stage1_checkpoint":os.path.abspath(ARGS.checkpoint_path),
        "noop_replay_max":dict(noop_max),
        "aggregate":{name:agg(name) for name in ("native","raw","learned","oracle")},
        "learned_rescue08":float(np.mean([r["learned_rescue08"] for r in rows])),"learned_harm08":float(np.mean([r["learned_harm08"] for r in rows])),
        "raw_rescue08":float(np.mean([r["raw_rescue08"] for r in rows])),"raw_harm08":float(np.mean([r["raw_harm08"] for r in rows])),
        "learned_matches_oracle":float(np.mean([r["learned_matches_oracle"] for r in rows])),"raw_matches_oracle":float(np.mean([r["raw_matches_oracle"] for r in rows])),
        "offset_hist":{"learned":{str(k):int(v) for k,v in sorted(learned_hist.items())},"raw":{str(k):int(v) for k,v in sorted(raw_hist.items())},"oracle":{str(k):int(v) for k,v in sorted(oracle_hist.items())}},
        "timing_total_sec":{k:float(v) for k,v in timing.items()},
    }
    summary["gaps"]={
        "learned_minus_native_utility":summary["aggregate"]["learned"]["utility"]-summary["aggregate"]["native"]["utility"],
        "learned_minus_native_success08":summary["aggregate"]["learned"]["success08"]-summary["aggregate"]["native"]["success08"],
        "oracle_minus_learned_utility":summary["aggregate"]["oracle"]["utility"]-summary["aggregate"]["learned"]["utility"],
        "oracle_minus_learned_success08":summary["aggregate"]["oracle"]["success08"]-summary["aggregate"]["learned"]["success08"],
    }
    with (out/"summary.json").open("w") as f: json.dump(summary,f,indent=2,sort_keys=True)
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
