"""P0 full-split action-level failure attribution on frozen rank_executed dumps.

Runs must have already been officially evaluated. Each frame is verified
against BOTH official [50,6] AP arrays before attributing score, object/top-k,
and NMS-set changes. Only compiled NMS survivors' union is CAD-labeled,
avoiding an unbounded new full-candidate-label job.

Usage: python -m diagnostics.decision_supervision.action_attribution --help
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .core import BASE_SHA, SCHEMA, digest, sha, physical_ids, write_json, write_npz
from .attribution_core import (
    common_bank, compiled_keep, arrange_labels, replay,
)
from diagnostics.candidate_conflict.core import scene_cluster_bootstrap

SPLITS = {"test_seen":100, "test_similar":130, "test_novel":160}
DEFAULT_SEEDS = ("20261113","20261114","20261115")


def _load_run(root, *, seed, arm, split, policy, interval=10):
    root = Path(root)
    manifest_path, evaluation_path = root/"inference.json", root/"evaluation.json"
    if not manifest_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError(f"Inference/evaluation missing: {root}")
    mf = json.loads(manifest_path.read_text(encoding="utf-8"))
    ev = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if (mf.get("schema") != SCHEMA or mf.get("kind") != "inference"
            or not mf.get("complete") or mf.get("probe_arm") != arm
            or str(mf.get("probe_seed")) != str(seed)
            or mf.get("split") != split or mf.get("interval") != interval
            or mf.get("base_checkpoint_sha256") != BASE_SHA
            or policy not in mf.get("policies", [])
            or ev.get("inference_sha256") != sha(manifest_path)):
        raise ValueError(f"Run metadata/official eval mismatch: {root}")
    evidence = ev.get("results", {}).get(policy)
    if not evidence:
        raise ValueError(f"Official AP for {policy} missing: {root}")
    ap_path = Path(evidence["file"])
    if not ap_path.is_file() or sha(ap_path) != evidence["sha256"]:
        raise ValueError(f"Official AP file changed: {ap_path}")
    ap = np.load(ap_path, allow_pickle=False)
    if ap.shape != (30, 26, 50, 6) or not np.isfinite(ap).all():
        raise ValueError(f"Expected official [30,26,50,6], got {ap.shape}")
    recs = [r for r in mf["records"] if r["policy"] == policy]
    keyed = {(int(r["scene"]),int(r["anno"])):r for r in recs}
    expected={(scene, anno) for scene in range(SPLITS[split],SPLITS[split]+30)
              for anno in range(0, 256, interval)}
    if len(recs) != len(expected) or set(keyed) != expected:
        raise ValueError(f"Full 780-frame coverage missing/duplicated: {root}")
    return dict(root=root, manifest=mf, manifest_sha256=sha(manifest_path),
                official_ap_sha256=sha(ap_path), records=keyed, ap=ap)


def _dump(run, key):
    rec = run["records"][key]
    path = run["root"]/rec["file"]
    if not path.is_file() or sha(path) != rec["sha256"]:
        raise ValueError(f"Grasp dump changed: {path}")
    g = np.asarray(np.load(path,allow_pickle=False))
    if g.ndim != 2 or g.shape[1]!=17 or not np.isfinite(g).all():
        raise ValueError(f"Bad pre-NMS 17D dump: {path}")
    return g


def _paired_runs(mapping, *, split, seeds, left, right, policy):
    paired={}
    identity = None
    for seed in seeds:
        arms=mapping.get(str(seed),{})
        if left not in arms or right not in arms:
            raise ValueError(f"Missing {seed} {left}/{right} inference roots")
        paired[str(seed)]={}
        for arm in (left,right):
            root=arms[arm].get(split)
            if not root: raise ValueError(f"Missing {seed}/{arm}/{split}")
            info=_load_run(root,seed=seed,arm=arm,split=split,policy=policy)
            mf=info["manifest"]
            protocol={k:mf.get(k) for k in
                      ("interval","snapshot_seed","base_checkpoint_sha256",
                       "base_config_digest","collision_protocol","plan_sha256",
                       "runtime","implementation_sha256")}
            if identity is None:identity=protocol
            elif identity!=protocol:
                raise ValueError(f"Unpaired underlying model/input protocol: {seed}/{arm}")
            paired[str(seed)][arm]=info
    return paired,identity


def _frame_label(rt_eval, scene, anno, base, selected, output, contract, resume):
    path=output/"labels"/f"scene_{scene:04d}"/f"{anno:04d}.npz"
    side=path.with_suffix(".json")
    indices=np.asarray(selected,dtype=np.int64)
    if path.exists() or side.exists():
        if not resume or not path.is_file() or not side.is_file():
            raise FileExistsError(f"Preexisting/incomplete frame label: {path}")
        meta=json.loads(side.read_text(encoding="utf-8"))
        if meta.get("contract_sha256") != digest(contract) or sha(path) != meta.get("sha256"):
            raise RuntimeError(f"Frame label identity/hash mismatch: {path}")
        with np.load(path,allow_pickle=False) as z:
            values={k:z[k] for k in z.files}
        if (not np.array_equal(values["indices"],indices)
                or not np.array_equal(values["physical_id"].astype(str),
                                      physical_ids(base[indices]).astype(str))):
            raise RuntimeError(f"Cached label physical actions changed: {path}")
        return values,meta
    result=rt_eval.evaluate(scene,anno,np.asarray(base[indices],dtype=np.float64))
    values=dict(indices=indices,
                physical_id=physical_ids(base[indices]),
                assigned_obj=np.asarray(result.assigned_obj,dtype=np.int64),
                collision_or_empty=np.asarray(result.collision_or_empty,dtype=np.uint8),
                pure_collision=np.asarray(result.pure_collision,dtype=np.uint8),
                empty=np.asarray(result.empty,dtype=np.uint8),
                friction=np.asarray(result.friction,dtype=np.float64))
    for field in ("assigned_obj","collision_or_empty","friction"):
        if len(values[field]) != len(indices):
            raise ValueError(f"Evaluator returned incomplete {field} labels")
    if not np.isfinite(values["friction"]).all():
        raise ValueError("Exact evaluator returned nonfinite friction")
    write_npz(path,values)
    meta=dict(schema=SCHEMA,kind="P0_action_labels",scene=scene,anno=anno,
              path=str(path.resolve()),sha256=sha(path),
              contract=contract,contract_sha256=digest(contract),
              num_labeled=len(indices))
    write_json(side,meta)
    return values,meta


def _to_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(rows[0]) if rows else []
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        if keys:writer.writeheader();writer.writerows(rows)


def _numeric_summary(rows, seed=20261114):
    metrics=("total_pp","rank_fixed_final_pp","quota_reselection_pp",
             "nms_set_replacement_pp")
    scenes=sorted({int(r["scene"]) for r in rows})
    result={}
    for name in metrics:
        values={scene:float(np.mean([float(r[name]) for r in rows
                     if int(r["scene"])==scene])) for scene in scenes}
        result[name]=scene_cluster_bootstrap(values,n_boot=5000,seed=seed)
    result["frame_rows"]=len(rows)
    result["scene_count"]=len(scenes)
    for name in ("nms_lost","nms_gained","final_lost","final_gained",
                 "lost_positive_nms","gained_zero_nms",
                 "positive_lost_zero_gained_score_inversions"):
        result[name]=int(sum(int(r[name]) for r in rows))
    return result


def analyze(args):
    if args.policy != "rank_executed":
        raise ValueError("P0 causal-path diagnostics require identical executed depth/geometry: rank_executed")
    if args.left==args.right or args.limit < 0 or args.max_compiled_pairs < 0:
        raise ValueError("Invalid comparison")
    seeds=tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Expected distinct scorer seeds")
    mapping=json.loads(Path(args.runs).read_text(encoding="utf-8"))
    sources,identity=_paired_runs(mapping,split=args.split,seeds=seeds,
                                   left=args.left,right=args.right,policy=args.policy)
    source_sha={seed:{arm:dict(inference=info["manifest_sha256"],
                              official_ap=info["official_ap_sha256"])
                      for arm,info in arms.items()} for seed,arms in sources.items()}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    first=SPLITS[args.split]
    keys=[(scene,anno) for scene in range(first,first+30) for anno in range(0,256,10)]
    if args.limit:keys=keys[:args.limit]
    formal=len(keys)==780 and not args.limit
    if not formal and not args.allow_partial:
        raise ValueError("Partial frames require --allow-partial")
    from graspnetAPI import GraspGroup
    import grasp_nms
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
    import inspect
    def compiled(g):
        return GraspGroup(g).nms(.03,float(np.pi/6)).grasp_group_array
    evaluator=ExactGraspNetActionEvaluator(args.dataset_root,"realsense",
                                            split=args.split,fc_mode="official",strict=True)
    evaluator_source=inspect.getsourcefile(ExactGraspNetActionEvaluator)
    compiled_file=inspect.getfile(grasp_nms)
    contract=dict(schema=SCHEMA,kind="P0_action_failure_attribution",
                  policy=args.policy,split=args.split,left=args.left,right=args.right,
                  seeds=seeds,source_run_hashes=source_sha,
                  runtime_protocol=identity,dataset_root=str(Path(args.dataset_root).resolve()),
                  official_eval_code_sha256=sha(evaluator_source),
                  compiled_nms_sha256=sha(compiled_file),
                  script_sha256=sha(__file__),core_sha256=sha(Path(__file__).with_name("attribution_core.py")),
                  frame_count=len(keys),complete=formal,
                  geometry="post_sensor_pre_nms_fixed_native_depth",
                  exact_actions_labeled="union_of_compiled_NMS_survivors_only")
    frame_rows=[]
    for number,(scene,anno) in enumerate(keys,1):
        key=(scene,anno);frame_inputs={}
        for seed in seeds:
            for arm in (args.left,args.right):
                frame_inputs[(seed,arm)]=_dump(sources[seed][arm],key)
        base=common_bank(frame_inputs)
        if not len(base):
            raise RuntimeError(f"Unexpected empty candidate pool: scene {scene} frame {anno}")
        keep={}
        for label, g in frame_inputs.items():
            keep[label]=compiled_keep(g,compiled)
        union_ids=np.unique(np.concatenate(list(keep.values())))
        frame_contract=dict(contract_sha256=digest(contract),
                            scene=scene,anno=anno,
                            pre_nms_physical_sha256=digest(physical_ids(base).tolist()),
                            dump_sha256={f"{s}_{a}":sources[s][a]["records"][key]["sha256"]
                                         for s in seeds for a in (args.left,args.right)},
                            nms_survivor_ids_sha256=digest(union_ids.tolist()))
        if getattr(evaluator,"_p0_last_scene",None)!=scene:
            evaluator.scene_cache.clear()
            evaluator._p0_last_scene=scene
        values,meta=_frame_label(evaluator,scene,anno,base,union_ids,
                                  out,frame_contract,args.resume)
        label_obj=SimpleNamespace(
            assigned_obj=values["assigned_obj"],
            collision_or_empty=values["collision_or_empty"],
            friction=values["friction"])
        assigned,bad,friction=arrange_labels(len(base),union_ids,label_obj)
        for seed in seeds:
            left_arr=frame_inputs[(seed,args.left)]
            right_arr=frame_inputs[(seed,args.right)]
            ref_left=sources[seed][args.left]["ap"][scene-first,anno//10]
            ref_right=sources[seed][args.right]["ap"][scene-first,anno//10]
            result=replay(base,left_arr[:,0],right_arr[:,0],
                          keep[(seed,args.left)],keep[(seed,args.right)],
                          assigned,bad,friction,
                          official_left=ref_left,official_right=ref_right,
                          tolerance=args.ap_tolerance)
            # A changed conflict edge is not the original directed suppressor.
            # Validate a deterministic subset with the installed compiled NMS.
            pairs=result["conflicts"]
            indices=np.arange(len(pairs))
            if len(indices)>args.max_compiled_pairs:
                rng=np.random.default_rng(scene*1009+anno+int(seed))
                indices=np.sort(rng.choice(indices,args.max_compiled_pairs,replace=False))
            for ix in indices:
                a,b=map(int,pairs[int(ix)])
                if len(compiled(left_arr[[a,b]].astype(np.float64)))!=1:
                    raise RuntimeError("Geometric changed-pair conflict differs from compiled NMS")
            row=dict(scene=scene,anno=anno,seed=str(seed),arm_left=args.left,
                     arm_right=args.right,source_candidates=len(base),
                     new_exact_labels=len(union_ids),
                     checked_changed_edges=len(indices),
                     total_changed_edges=len(pairs),
                     **{k:v for k,v in result.items()
                        if k not in ("stages_pp","inversion_pairs","left_final_ids",
                                      "right_final_ids","left_nms_keep","right_nms_keep",
                                      "quality","conflicts")})
            frame_rows.append(row)
            path=out/"actions"/str(seed)/f"scene_{scene:04d}"/f"{anno:04d}.npz"
            write_npz(path,dict(
                source_physical_id=physical_ids(base),
                score_left=np.asarray(left_arr[:,0],dtype=np.float64),
                score_right=np.asarray(right_arr[:,0],dtype=np.float64),
                labeled_indices=union_ids,
                assigned_obj=values["assigned_obj"],
                exact_friction=values["friction"],
                collision_or_empty=values["collision_or_empty"],
                left_nms_keep=result["left_nms_keep"],
                right_nms_keep=result["right_nms_keep"],
                left_final_ids=result["left_final_ids"],
                right_final_ids=result["right_final_ids"],
                changed_conflict_pairs=result["conflicts"],
                verified_pair_indices=indices,
                inversion_pairs=np.asarray(result["inversion_pairs"],dtype=np.int64).reshape(-1,2),
                stages_pp=result["stages_pp"]))
        if number%26==0 or number==len(keys):
            print(f"[P0] {number}/{len(keys)} {args.split} scene={scene} frame={anno} "
                  f"new_exact_actions={len(union_ids)}",flush=True)
    _to_csv(out/"per_frame.csv",frame_rows)
    scenes=sorted({int(r["scene"]) for r in frame_rows})
    scene_rows=[]
    for scene in scenes:
        for seed in seeds:
            rows=[r for r in frame_rows if r["scene"]==scene and r["seed"]==seed]
            scene_rows.append(dict(scene=scene,seed=seed,
                **{k:float(np.mean([r[k] for r in rows]))
                   for k in ("total_pp","rank_fixed_final_pp","quota_reselection_pp",
                             "nms_set_replacement_pp")}))
    _to_csv(out/"per_scene_seed.csv",scene_rows)
    summary=dict(schema=SCHEMA,kind="P0_action_failure_attribution",
                 complete=formal,contract=contract,
                 source_contract_sha256=digest(contract),
                 conditional_scene_bootstrap=_numeric_summary(frame_rows),
                 notes=["Pathwise decomposition order is fixed-final rank, then per-object/global re-selection, then NMS survivor replacement.",
                        "No quality claims about candidates suppressed by BOTH methods; only compiled NMS survivor union received exact labels.",
                        "Changed-conflict edges are geometric undirected edges, not certified historical directed suppressors.",
                        "Scenes were already inspected; this is exploratory only, not unseen confirmation."])
    write_json(out/"summary.json",summary)
    lines=["# P0 动作级候选选择归因", "",
           f"Split: {args.split}; left={args.left}, right={args.right}; seeds={','.join(seeds)}.",
           "Only rank_executed; frozen geometry. Exact action CAD labels are restricted to the union of compiled NMS survivors.",
           "Every native-side AP result was checked against the previously saved official [50,6] matrix.",
           "", "| Stage | Conditional scene mean delta (pp) | 95% CI |",
           "|---|---:|---|"]
    for name,desc in (("rank_fixed_final_pp","固定原最终集合、只重排"),
                      ("quota_reselection_pp","固定原NMS、重新分配物体/全局名额"),
                      ("nms_set_replacement_pp","采用新NMS存活集合"),
                      ("total_pp","完整结果变化")):
        d=summary["conditional_scene_bootstrap"][name]
        lines.append(f"| {desc} | {d['mean']:+.5f} | [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}] |")
    lines.extend(["", "注：这是有序路径分解，不是唯一因果归因；置信区间以场景为单位，三个固定训练seed的变化先在场景内平均。"])
    (out/"RESULTS_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",required=True,help="Accepted report.py seed -> arm -> split -> inference-root JSON")
    p.add_argument("--split",required=True,choices=tuple(SPLITS))
    p.add_argument("--left",default="A",choices=("A","B","C","D"))
    p.add_argument("--right",default="C",choices=("A","B","C","D"))
    p.add_argument("--seeds",default=",".join(DEFAULT_SEEDS))
    p.add_argument("--policy",default="rank_executed")
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--ap-tolerance",type=float,default=1e-6)
    p.add_argument("--max-compiled-pairs",type=int,default=64)
    p.add_argument("--limit",type=int,default=0,help="First N frames; smoke only")
    p.add_argument("--allow-partial",action="store_true")
    p.add_argument("--resume",action="store_true")
    analyze(p.parse_args())


if __name__=="__main__":
    main()
