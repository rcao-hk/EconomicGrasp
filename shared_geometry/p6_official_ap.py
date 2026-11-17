"""P6-2B: one-way, pre-registered official GraspNet AP execution.

This invokes actual EconomicGrasp RGB inference_cva.py + graspnetAPI eval.py.
The P5 13k-parameter offline quality readers CANNOT be submitted as a
deployable grasp predictor. They only score a fixed physical-action bank.

No test-scene model selection, best-checkpoint sweep, sparse/partial evaluation,
or relabeling is permitted. Register exactly one full deployable checkpoint,
then execute each official split once at sample_interval=1.0.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .data import sha256,load_signed,signed,atomic_json
from .p4_common import new_directory

SCHEMA="shared-visible-geometry-p6-official-registered-v1"
SPLITS=("test_seen","test_similar","test_novel")
RANGES={"test_seen":range(100,130),
        "test_similar":range(130,160),"test_novel":range(160,190)}
BANNED=("--use_obs_depth","--use_gt_depth","--use_fuse_depth",
        "--save_nocollision","--save_dir","--test_mode",
        "--sample_interval","--dataset_root","--checkpoint_path",
        "--gntrans_rgb_root","--remove_dump","--enable_eval")
ARGUMENTS_WITH_VALUE=("--num_point","--m_point","--batch_size","--num_workers",
        "--num_view","--num_angle","--num_depth","--min_depth","--max_depth",
        "--bin_num","--collision_thresh","--collision_voxel_size",
        "--pose_depth_mode","--kview_mode","--graspness_mode","--fuse_type",
        "--rayrope_group_chunk","--gvar_action_chunk",
        "--rayrope_checkpoint_chunks","--gvar_activation_checkpoint",
        "--pose_hidden_dim","--ray_gravity_hidden_dim","--ray_gravity_mid_dim",
        "--camera_pose_key","--camera_gravity_key")
ARGUMENTS_FLAG=("--multi_modal","--use_cdf","--enable_point_rayrope",
                "--enable_gvar","--use_top4_view_infer")
ALLOWED=set(ARGUMENTS_WITH_VALUE+ARGUMENTS_FLAG)


def relevant_inference_sources(root):
    """Freeze all Python modules imported by the deployed inference/evaluator."""
    names=("inference_cva.py","eval.py")
    folders=("models","dataset","utils","rayrope_gvar","libs")
    files=[root/p for p in names]
    for folder in folders:
        directory=root/folder
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.py")))
    return sorted(set(files))


def validate_model_args(tokens):
    if not isinstance(tokens,list) or not tokens:
        raise ValueError("A locked list of model inference CLI arguments is required")
    args=[];i=0
    while i<len(tokens):
        flag=tokens[i]
        if not isinstance(flag,str) or flag in BANNED or flag not in ALLOWED:
            raise ValueError("Unsafe or unsupported official inference argument: "+str(flag))
        if flag in ARGUMENTS_WITH_VALUE:
            if i+1==len(tokens) or not isinstance(tokens[i+1],str) or tokens[i+1].startswith("--"):
                raise ValueError("Missing value for "+flag)
            args.extend([flag,tokens[i+1]]);i+=2
        else:
            args.append(flag);i+=1
    if "--multi_modal" not in args:
        raise ValueError("Official RGB-only EconomicGrasp requires --multi_modal")
    if len([x for x in args if x=="--multi_modal"])!=1:
        raise ValueError("Duplicate --multi_modal")
    return args


def register(registry,checkpoint,dataset_root,model_args,repo_root=".",
             independent_summary=None,attest=False):
    if not attest:
        raise ValueError("Attest that configuration was frozen without accessing official test outcomes")
    model_args=validate_model_args(model_args)
    root=Path(repo_root).resolve()
    ckpt=Path(checkpoint).resolve()
    dataset=Path(dataset_root).resolve()
    for f in ("inference_cva.py","eval.py","utils/arguments.py"):
        if not (root/f).is_file():raise FileNotFoundError(root/f)
    if not ckpt.is_file() or not dataset.is_dir():
        raise ValueError("Deployable checkpoint and GraspNet dataset directory required")
    # inference_cva.py uses observed sensor clouds for model-free collision
    # filtering when collision_thresh>0, even if the model itself is RGB-only.
    collision_threshold=0.
    if "--collision_thresh" in model_args:
        i=model_args.index("--collision_thresh")
        collision_threshold=float(model_args[i+1])
    if collision_threshold<0:
        raise ValueError("Invalid negative collision threshold")
    sensor_depth_postfilter=collision_threshold>0.
    independent=None
    if independent_summary:
        r=load_signed(independent_summary)
        if (r.get("schema")!="shared-visible-geometry-p6-independent-v1" or
            r.get("kind")!="independent_action_bank_readout" or
            not r.get("frozen_before_holdout") or not r.get("official_ap_not_run")):
            raise ValueError("Independent summary is not a completed frozen holdout")
        independent=dict(path=str(Path(independent_summary).resolve()),
                         sha256=sha256(independent_summary),
                         kind=r["kind"])
    manifest=dict(schema=SCHEMA,registered_before_official_evaluation=True,
        one_frozen_checkpoint_only=True,official_test_not_for_tuning=True,
        checkpoint=str(ckpt),checkpoint_sha256=sha256(ckpt),
        dataset_root=str(dataset),camera="realsense",sample_interval=1.0,
        split_order=list(SPLITS),repo_root=str(root),
        inference_args=model_args,
        code_hashes={str(p.relative_to(root)):sha256(p)
                     for p in relevant_inference_sources(root)},
        independent_summary=independent,
        independent_gate_completed=independent is not None,
        observed_sensor_depth_used_for_collision_postfilter=sensor_depth_postfilter,
        perception_input_policy=("RGB-only model with observed-depth collision postfilter" 
                                 if sensor_depth_postfilter else
                                 "RGB-only model; no observed-depth postfilter"),
        exact_metric="GraspNet-1B official graspnetAPI, full 256 views x 90 test scenes",
        provenance_note="External holdout attestation is not automatically proven by its metadata.")
    path=Path(registry)
    if path.exists():raise FileExistsError("Official registry cannot be overwritten")
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,signed(manifest))
    return manifest


def load_registration(registry):
    m=load_signed(registry)
    if (m.get("schema")!=SCHEMA or m.get("sample_interval")!=1. or
        m.get("split_order")!=list(SPLITS) or
        not m.get("registered_before_official_evaluation") or
        not m.get("one_frozen_checkpoint_only")):
        raise ValueError("Not an unmodified P6 registered official AP experiment")
    validate_model_args(m["inference_args"])
    if sha256(m["checkpoint"])!=m["checkpoint_sha256"]:
        raise ValueError("Official registered checkpoint changed")
    root=Path(m["repo_root"])
    for f,digest in m["code_hashes"].items():
        if sha256(root/f)!=digest:
            raise ValueError("Official inference/eval implementation changed: "+f)
    if m["independent_summary"]:
        r=m["independent_summary"]
        if sha256(r["path"])!=r["sha256"]:
            raise ValueError("Independent validation result changed after registration")
    return m


def commands(m,output_root,num_workers=4,eval_workers=10):
    if num_workers<0 or eval_workers<1:raise ValueError("Invalid worker budget")
    cmd=[]
    root=Path(m["repo_root"]);target=Path(output_root).resolve()
    for split in SPLITS:
        out=str(target/split)
        inf=[sys.executable,str(root/"inference_cva.py"),
             "--dataset_root",m["dataset_root"],"--checkpoint_path",m["checkpoint"],
             "--test_mode",split,"--save_dir",out,"--camera","realsense",
             "--sample_interval","1.0","--num_workers",str(num_workers)]
        inf+=m["inference_args"]
        eva=[sys.executable,str(root/"eval.py"),
             "--dataset_root",m["dataset_root"],"--dump_dir",out,
             "--camera","realsense","--split",split,
             "--num_workers",str(eval_workers),"--sample_interval","1"]
        cmd.append(dict(split=split,inference=inf,evaluation=eva,output=out))
    return cmd


def assert_complete_dump(directory,split):
    path=Path(directory)
    missing=[]
    count=0
    for sid in RANGES[split]:
        cam=path/("scene_%04d"%sid)/"realsense"
        for anno in range(256):
            if not (cam/("%04d.npy"%anno)).is_file():
                missing.append((sid,anno))
            else:
                count+=1
    if missing:raise ValueError("Refusing incomplete official AP: %s has %d missing frames (example %s)"
                                %(split,len(missing),missing[:3]))
    if count!=7680:raise AssertionError("Expected 30*256 full-frame grasp results")
    return count


def execute(registry,output_root,workers=4,eval_workers=10,run=False):
    m=load_registration(registry)
    steps=commands(m,output_root,workers,eval_workers)
    if not run:return steps
    root=new_directory(output_root)
    # This record is created BEFORE touching test data. Do not auto-resume after
    # partial results, as repeated test-dependent execution invites tuning.
    state=dict(registry_sha256=sha256(registry),
               checkpoint_sha256=m["checkpoint_sha256"],
               independent_gate_completed=m["independent_gate_completed"],
               status="started",full_official_test_only=True)
    atomic_json(root/"state.json",signed(state))
    files=[]
    for item in steps:
        split=item["split"];dest=root/split
        dest.mkdir()
        with (dest/"inference.log").open("w",encoding="utf-8") as log:
            subprocess.run(item["inference"],cwd=m["repo_root"],
                           check=True,stdout=log,stderr=subprocess.STDOUT)
        n=assert_complete_dump(dest,split)
        with (dest/"official_eval.log").open("w",encoding="utf-8") as log:
            subprocess.run(item["evaluation"],cwd=m["repo_root"],
                           check=True,stdout=log,stderr=subprocess.STDOUT)
        artifact=dest/("ap_"+split+"_realsense.npy")
        if not artifact.is_file():raise FileNotFoundError("Official evaluator failed to save "+str(artifact))
        files.append(dict(split=split,frames=n,path=str(artifact.resolve()),sha256=sha256(artifact)))
    state.update(status="complete",official_artifacts=files,
                 result_type="official_AP_not_P5_Q",
                 warning=("This official evaluator measures a deployable RGB grasp "
                          "checkpoint, NOT the frozen P5/P6 reader action-bank study."))
    atomic_json(root/"state.json",signed(state))
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="stage",required=True)
    r=sub.add_parser("register")
    r.add_argument("--registry",required=True);r.add_argument("--checkpoint",required=True)
    r.add_argument("--dataset-root",required=True);r.add_argument("--repo-root",default=".")
    r.add_argument("--model-args-json",required=True,help="JSON file containing a frozen list of CLI tokens")
    r.add_argument("--independent-summary")
    r.add_argument("--attest-no-official-test-tuning",action="store_true")
    e=sub.add_parser("evaluate")
    e.add_argument("--registry",required=True);e.add_argument("--output",required=True)
    e.add_argument("--num-workers",type=int,default=4)
    e.add_argument("--eval-workers",type=int,default=10)
    e.add_argument("--run",action="store_true",help="Explicitly execute full official test; default: dry-run")
    args=p.parse_args()
    if args.stage=="register":
        tokens=json.loads(Path(args.model_args_json).read_text(encoding="utf-8"))
        result=register(args.registry,args.checkpoint,args.dataset_root,tokens,
                        args.repo_root,args.independent_summary,args.attest_no_official_test_tuning)
    else:
        result=execute(args.registry,args.output,args.num_workers,args.eval_workers,args.run)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
