#!/usr/bin/env python3
"""Select problematic *frames* in Novel scenes from archived, paired AP tensors.

No model inference or GT-grasp oracle. Saved 17-column GraspGroup dumps are
summarized descriptively; independently generated actions are NOT paired.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from analyze_gvar_scene_paired import load_all, scene_metric, AuditError, FRICTIONS, FRAMES

DEFAULT_SCENES = (165, 167, 174, 160, 189)


def frame_ap(tensor: np.ndarray, scene: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scene not in range(160, 190) or tensor.shape != (30, 26, 50, 6):
        raise AuditError(f"Invalid Novel scene {scene} or AP shape {tensor.shape}")
    frames = tensor[scene - 160]
    return (frames.mean(axis=(1, 2)).astype('float64') * 100,
            frames[:, :, 1].mean(axis=1).astype('float64') * 100,
            frames[:, :, 3].mean(axis=1).astype('float64') * 100)


def grasp_stats(path: Path, topk: int, require_grasps: bool) -> dict:
    if not path.is_file():
        if require_grasps:
            raise FileNotFoundError(f"Grasp dump missing: {path}")
        return {"available": False, "count": None, "topk": None, "score_topk_mean": None,
                "width_topk_mean_m": None, "insertion_topk_mean_m": None, "center_z_topk_mean_m": None,
                "file_sha256": None}
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] != 17 or not np.isfinite(arr).all():
        raise AuditError(f"Malformed/nonfinite GraspGroup [N,17]: {path} {arr.shape}")
    order = np.argsort(-arr[:, 0], kind="stable")[:topk]
    top = arr[order]
    def mean(idx): return float(np.mean(top[:, idx])) if len(top) else None
    return {"available": True, "count": len(arr), "topk": len(top),
            "score_topk_mean": mean(0), "width_topk_mean_m": mean(1),
            "insertion_topk_mean_m": mean(3), "center_z_topk_mean_m": mean(15),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_csv(path: Path, rows: list[dict]):
    if not rows: raise AuditError(f"Refusing to write empty table: {path}")
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def analyze(root: Path, out: Path, scenes: list[int], variants: list[str], reference: str,
            comparison: str, epoch: int = 19, worst_frames: int = 3, best_frames: int = 1,
            topk: int = 10, require_grasps: bool = False):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be a new, empty directory: {out}")
    if reference not in variants or comparison not in variants or reference == comparison:
        raise AuditError("Both comparison and reference must be distinct members of variants")
    if worst_frames < 0 or best_frames < 0 or worst_frames + best_frames < 1 or topk < 1:
        raise AuditError("Invalid selection / topk")
    if len(scenes) != len(set(scenes)) or not scenes or any(s not in range(160, 190) for s in scenes):
        raise AuditError(f"Scene IDs must be unique within 160..189: {scenes}")
    loaded, audits = load_all(root, variants, epoch)
    rows, chosen, scene_rows, stats = [], [], [], []
    for scene in scenes:
        b = loaded[reference]['data']['test_novel'].tensor
        c = loaded[comparison]['data']['test_novel'].tensor
        b_ap, b04, b08 = frame_ap(b, scene)
        c_ap, c04, c08 = frame_ap(c, scene)
        diff = c_ap - b_ap
        ranking = np.argsort(diff, kind='stable')
        worst = ranking[:worst_frames].tolist()
        best = [int(i) for i in ranking[::-1] if int(i) not in worst][:best_frames]
        selected = {i: 'worst' for i in worst}
        selected.update({i:'best' for i in best})
        all_metrics = {v:frame_ap(loaded[v]['data']['test_novel'].tensor, scene) for v in variants}
        for i, fid in enumerate(FRAMES):
            row={"scene_id":scene,"frame_id":fid,"reference":reference,"comparison":comparison,
                 "reference_AP":float(b_ap[i]),"comparison_AP":float(c_ap[i]),"delta_AP_pp":float(diff[i]),
                 "reference_mu04":float(b04[i]),"comparison_mu04":float(c04[i]),"delta_mu04_pp":float(c04[i]-b04[i]),
                 "reference_mu08":float(b08[i]),"comparison_mu08":float(c08[i]),"delta_mu08_pp":float(c08[i]-b08[i]),
                 "selection":selected.get(i,'')}
            for v in variants:
                row[f"AP_{v}"]=float(all_metrics[v][0][i]);row[f"mu04_{v}"]=float(all_metrics[v][1][i]);row[f"mu08_{v}"]=float(all_metrics[v][2][i])
            rows.append(row)
            if i in selected:
                chosen.append({"scene_id":scene,"frame_id":fid,"selection":selected[i],
                               "reference":reference,"comparison":comparison,"delta_AP_pp":float(diff[i]),
                               "delta_mu04_pp":float(c04[i]-b04[i]),"delta_mu08_pp":float(c08[i]-b08[i])})
                for v in variants:
                    path=root / 'eval' / f'{v}_e{epoch}' / 'test_novel' / f'scene_{scene:04d}' / 'realsense' / f'{fid:04d}.npy'
                    stat=grasp_stats(path, topk,require_grasps)
                    stats.append({"variant":v,"scene_id":scene,"frame_id":fid,"selection":selected[i],
                                  "AP":float(all_metrics[v][0][i]),"mu04":float(all_metrics[v][1][i]),
                                  "mu08":float(all_metrics[v][2][i]),"grasp_path":str(path),**stat})
        scene_rows.append({"scene_id":scene,"reference_AP":float(b_ap.mean()),
                           "comparison_AP":float(c_ap.mean()),"delta_AP_pp":float(diff.mean()),
                           "delta_mu04_pp":float((c04-b04).mean()),"delta_mu08_pp":float((c08-b08).mean()),
                           "positive_frames":int(np.count_nonzero(diff>0)),"negative_frames":int(np.count_nonzero(diff<0))})
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'novel_frames.csv', rows)
    save_csv(out/'novel_scenes.csv', scene_rows)
    save_csv(out/'grasp_stats.csv', stats)
    meta={"version":1,"root":str(root.resolve()),"epoch":epoch,"split":"test_novel",
          "scene_ids":scenes,"frame_ids":list(FRAMES),"variants":variants,"reference":reference,
          "comparison":comparison,"selected_frames":chosen,
          "grasp_dump_available":sum(bool(x['available']) for x in stats),"grasp_dump_total":len(stats),
          "train_eval_inputs": [{"path":x['path'],"sha256":x['sha256']} for x in audits]}
    (out/'selected_frames.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    lines=["# GVAR Novel failure-frame selection", "",f"Comparison: {comparison} - {reference}, epoch e{epoch}; Original GraspNet, same 26 frames/scene.",
           "", "**AP tensors compare different physical grasps, not fixed candidates or score-only effects.**",
           "", "| Scene | ΔAP (pp) | ΔAP μ0.4 | ΔAP μ0.8 | Positive/negative frames |",
           "|---|---:|---:|---:|---:|"]
    for s in scene_rows:
        lines.append(f"| {s['scene_id']:04d} | {s['delta_AP_pp']:+.3f} | {s['delta_mu04_pp']:+.3f} | {s['delta_mu08_pp']:+.3f} | {s['positive_frames']}/{s['negative_frames']} |")
    lines += ["", "## Selected frames", ""]
    for row in chosen:
        lines.append(f"- scene_{row['scene_id']:04d} frame {row['frame_id']:04d} ({row['selection']}): ΔAP {row['delta_AP_pp']:+.3f}, μ0.4 {row['delta_mu04_pp']:+.3f}, μ0.8 {row['delta_mu08_pp']:+.3f}")
    lines += ["", f"GraspGroup dumps available: {meta['grasp_dump_available']}/{meta['grasp_dump_total']} selected variant-frames.",
              "Missing dumps are reported as unavailable, never imputed. Per-grasp score/width/translation statistics are descriptive only.",
              "", "Use `selected_frames.json` as the *exact frame selector* for GPU depth-sidecar export. Do not select frames after inspecting uncertainty performance."]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return meta


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['baseline','volume_fixed','volume','volume_rel'])
    p.add_argument('--reference',default='baseline');p.add_argument('--comparison',default='volume_rel')
    p.add_argument('--scenes',nargs='+',type=int,default=DEFAULT_SCENES)
    p.add_argument('--epoch',type=int,default=19)
    p.add_argument('--worst-frames',type=int,default=3);p.add_argument('--best-frames',type=int,default=1)
    p.add_argument('--topk',type=int,default=10);p.add_argument('--require-grasps',action='store_true')
    a=p.parse_args(argv)
    try:
        data=analyze(a.root,a.output_dir,a.scenes,a.variants,a.reference,a.comparison,a.epoch,a.worst_frames,a.best_frames,a.topk,a.require_grasps)
    except (AuditError,ValueError,FileNotFoundError,OSError,KeyError) as e:
        p.exit(2,f'Novel failure analysis failed: {e}\n')
    print(json.dumps({"selected":len(data['selected_frames']),"grasp_dumps_available":data['grasp_dump_available'],"output":str(a.output_dir)},indent=2))

if __name__=='__main__': main()
