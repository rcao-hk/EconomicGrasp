"""P6-2A: read-only evaluation on an explicitly independent external action bank.

The manifest must be produced BEFORE evaluation on a separate dataset, with
fully frozen P4 Corrector and P5 predicted/corrected readers. This tool NEVER
constructs labels, trains a model, or selects weights using the holdout.
It does not infer geometry from RGB itself: input tokens must be generated
offline by a frozen pipeline, and their provenance must be audited separately.

The original GraspNet train/development scenes 0-99 and official test scenes
100-189 are explicitly forbidden as an 'independent' tuning set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from .data import sha256,load_signed,signed,atomic_json
from .model import ModelConfig,PredictedGeometryControl
from .p4_common import score_physics,family_rows,paired_summary,new_directory,write_records
from .p0_relative_decision import scene_bootstrap

SCHEMA="shared-visible-geometry-p6-independent-v1"
KEYS=("predicted_physics","corrected_physics","local_xyz","bank_dims",
      "physical_id","family_id","quality_count")
PROHIBITED=("clean_visible_physics","clean_depth","ground_truth_depth",
            "visible_depth","target_xyz","target_udf","target_state")


def validate_manifest(index, expected_depth_sha, smoke=False):
    if index.get("schema")!="shared-visible-geometry-p6-external-bank-v1":
        raise ValueError("Wrong independent holdout bank manifest schema")
    if index.get("partition")!="independent_holdout":
        raise ValueError("P6 independent test cannot use the P5 dev validation partition")
    if (not index.get("quality_independently_evaluated") or
        not index.get("clean_geometry_never_model_input") or
        not index.get("frozen_model_before_holdout") or
        index.get("depth_corrector_sha256")!=expected_depth_sha):
        raise ValueError("Holdout lacks required independent/frozen provenance")
    if not index.get("external_dataset_id") or not index.get("entries"):
        raise ValueError("External dataset/entries required")
    known=set()
    for entry in index["entries"]:
        row=entry["row"];scene=int(row["scene"])
        if scene<0 or row["domain"] not in ("original","gntrans"):
            raise ValueError("Invalid holdout row")
        k=(str(index["external_dataset_id"]),scene,int(row["anno"]),row["domain"])
        if k in known:raise ValueError("Repeated holdout frame")
        known.add(k)
        if index["external_dataset_id"].lower() in ("graspnet","graspnet-1billion","gn-trans"):
            raise ValueError("GraspNet training/development/official test cannot be rebranded as independent")
        if sha256(entry["path"])!=entry["sha256"]:
            raise ValueError("Independent bank file SHA mismatch")
    hold={(str(index["external_dataset_id"]),int(e["row"]["scene"])) for e in index["entries"]}
    for field in ("training_scene_keys","development_scene_keys","official_test_scene_keys"):
        if field not in index:raise ValueError("Manifest must document "+field)
        refs={tuple(x) for x in index[field]}
        if len(refs)!=len(index[field]) or refs&hold:
            raise ValueError("Holdout scenes overlap "+field)
    if not smoke and len({x[1] for x in hold})<2:
        raise ValueError("Formal holdout requires at least two independent scenes")
    return hold


def _load_reader(path,arm,depth_sha,device):
    obj=torch.load(path,map_location="cpu",weights_only=False)
    p=obj.get("protocol",{})
    if (obj.get("schema")!="shared-visible-geometry-p5-reader-v1" or
        p.get("kind")!="reader" or p.get("arm")!=arm or
        p.get("depth_weight_sha256")!=depth_sha or p.get("smoke",True)):
        raise ValueError("P6 independent eval requires formal P5 "+arm+" reader")
    selector=load_signed(Path(path).parent/"selection.json")
    if not selector["complete"] or selector["weight_sha256"]!=sha256(path):
        raise ValueError("Reader was not frozen at its selected validation epoch")
    model=PredictedGeometryControl(ModelConfig(**p["cfg"])).to(device)
    model.load_state_dict(obj["model"],strict=True);model.eval()
    return model,p


def _load_frame(entry):
    with np.load(entry["path"],allow_pickle=False) as f:
        if any(k in f.files for k in PROHIBITED) or set(f.files)!=set(KEYS):
            raise ValueError("Independent bank contains prohibited or missing tensor")
        data={key:np.asarray(f[key]) for key in KEYS}
    a=data["predicted_physics"];b=data["corrected_physics"]
    if a.ndim!=4 or a.shape[-2:]!=(75,6) or b.shape!=a or a.shape[1]!=7:
        raise ValueError("External action geometry must be [F,7,75,6]")
    n=a.shape[0]
    if (data["local_xyz"].shape!=(n,7,75,3) or
        data["bank_dims"].shape!=(n,7,3) or
        data["quality_count"].shape!=(n,7) or
        data["physical_id"].shape!=(n,7) or
        data["family_id"].shape!=(n,)):
        raise ValueError("External physical action bank misaligned")
    if any(not np.isfinite(data[k]).all() for k in
           ("predicted_physics","corrected_physics","local_xyz","bank_dims")):
        raise ValueError("Nonfinite external model input")
    if np.any(data["quality_count"]<0) or np.any(data["quality_count"]>6):
        raise ValueError("Invalid count/6 exact Q")
    if len(set(data["physical_id"].astype(str).ravel()))!=n*7:
        raise ValueError("Duplicated physical action IDs")
    return data


def evaluate(manifest,pred_weight,corr_weight,output,device="cpu",bootstrap=5000,seed=0,smoke=False):
    if bootstrap<1:raise ValueError("Invalid bootstrap repetitions")
    idx=load_signed(manifest)
    depth_sha=idx["depth_corrector_sha256"]
    validate_manifest(idx,depth_sha,smoke)
    pm,pp=_load_reader(pred_weight,"predicted",depth_sha,device)
    cm,cp=_load_reader(corr_weight,"corrected",depth_sha,device)
    if (pp["train_fingerprint"]!=cp["train_fingerprint"] or
        pp["val_fingerprint"]!=cp["val_fingerprint"] or
        pp["seed"]!=cp["seed"] or pp["epochs"]!=cp["epochs"]):
        raise ValueError("P6 readers have unmatched training budgets")
    offsets=idx["offsets_m"]
    if len(offsets)!=7 or abs(offsets[0])>1e-10:
        raise ValueError("External bank requires fixed original + six offsets")
    rows={"predicted":[],"corrected":[]}
    for entry in idx["entries"]:
        data=_load_frame(entry)
        phy={
            "predicted":data["predicted_physics"],
            "corrected":data["corrected_physics"],
        }
        loc=torch.as_tensor(data["local_xyz"],device=device,dtype=torch.float32)[None]
        dim=torch.as_tensor(data["bank_dims"],device=device,dtype=torch.float32)[None]
        for arm,model in (("predicted",pm),("corrected",cm)):
            with torch.inference_mode():
                logits=score_physics(model,torch.as_tensor(phy[arm],device=device,dtype=torch.float32)[None],
                                     loc,dim)[0].cpu().numpy()
            frame={"quality_count":data["quality_count"],"physical_id":data["physical_id"],
                   "family_id":data["family_id"]}
            rows[arm].extend(family_rows(logits,frame,entry["row"],offsets,phy[arm]))
    if not rows["predicted"] or len(rows["predicted"])!=len(rows["corrected"]):
        raise ValueError("Missing holdout decisions")
    summary={arm:paired_summary(r,rows["predicted"] if arm=="corrected" else None,
                                bootstrap,seed) for arm,r in rows.items()}
    paired=scene_bootstrap(rows["corrected"],rows["predicted"],metric="gain",
                           n_boot=bootstrap,seed=seed)
    dest=new_directory(output)
    report=dict(schema=SCHEMA,kind="independent_action_bank_readout",
        external_dataset_id=idx["external_dataset_id"],n_frames=len(idx["entries"]),
        n_scenes=len({r["scene"] for r in rows["predicted"]}),
        source_manifest_sha256=sha256(manifest),
        readers={"predicted":sha256(pred_weight),"corrected":sha256(corr_weight)},
        depth_corrector_sha256=depth_sha,results=summary,corrected_minus_predicted=paired,
        frozen_before_holdout=idx["frozen_model_before_holdout"],official_ap_not_run=True,
        source_observations="Manifest provenance is an assertion; inspect actual data generation before a generalization claim.",
        limitations=("External bank must be constructed with independent quality labels. "
                     "No Clean geometry enters model input. Fixed candidates are not official AP."))
    report["artifacts"]={k:write_records(dest/(k+"_families.jsonl.gz"),v) for k,v in rows.items()}
    atomic_json(dest/"summary.json",signed(report))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--predicted-weight",required=True);p.add_argument("--corrected-weight",required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cpu")
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    r=evaluate(a.manifest,a.predicted_weight,a.corrected_weight,a.output,
               a.device,a.bootstrap,a.seed,a.smoke)
    print("P6 external holdout Q (NOT AP): ",r["corrected_minus_predicted"])


if __name__=="__main__":
    main()
