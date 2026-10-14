"""Independent runtime for Mixed-v2 Point RayRoPE (+/- GVAR) feasibility.

Avoids modifying the archived MoGe/RayRoPE runtime or its code_fingerprint,
so completed U0/U3 checkpoints retain their original inference contract.
"""
from __future__ import annotations
import hashlib
from dataclasses import asdict
from pathlib import Path
import torch
from moge_rayrope.config import ModelConfig
from moge_rayrope.runtime import (ROOT, MAIN_FILES, verify_main_contract,
    configure_main, make_dataset, move_batch, seed_all, worker_init, rng_state,
    restore_rng, sha256_file, digest, atomic_json, atomic_torch, atomic_npy,
    ensure_manifest)
from .reader import GVARConfig

VERSION = "rayrope_point_gvar_mixed_v2_v1"
VARIANTS = ("rayrope_point","rayrope_point_gvar")


def point_config(*, group_chunk=32, seeds=1024):
    return ModelConfig(use_moge=False,use_rayrope=True,ray_encoding="point",
        uncertainty="fixed",uncertainty_loss="interval",pose_mode="global_film",
        group_chunk=group_chunk,seeds=seeds,ray_apply_vo=True)


def code_fingerprint():
    h=hashlib.sha256()
    files=[]
    files.extend((ROOT/"moge_rayrope").glob("*.py"))
    files.extend((ROOT/"rayrope_gvar").glob("*.py"))
    files.extend(ROOT/path for path in MAIN_FILES)
    files.extend(ROOT/path for path in (
        "train_rayrope_gvar.py","inference_rayrope_gvar.py",
        "eval_rayrope_gvar.py","compare_rayrope_gvar.py",
        "scripts/run_rayrope_point_gvar_mixed_v2.sh"))
    for path in sorted(set(files)):
        if not path.is_file(): raise FileNotFoundError(f"Missing source in fingerprint: {path}")
        h.update(str(path.relative_to(ROOT)).encode());h.update(path.read_bytes())
    return h.hexdigest()


def load_model(checkpoint,device="cuda",verify=True):
    from .model import EconomicGraspPointGVAR
    from moge_rayrope.mixed import DEPTH_CONTRACT
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if ck.get("version")!=VERSION: raise ValueError("Wrong experiment checkpoint version")
    p=ck["protocol"]; mc=ModelConfig(**p["model"])
    if p["variant"] not in VARIANTS: raise ValueError("Unknown Point/GVAR variant")
    if not mc.use_rayrope or mc.ray_encoding!="point" or mc.use_moge or mc.uncertainty!="fixed":
        raise ValueError("Checkpoint is not Point RayRoPE")
    if p.get("mixed_depth_supervision")!=DEPTH_CONTRACT or DEPTH_CONTRACT["contract_version"]!=2:
        raise RuntimeError("Checkpoint mixed v2 depth contract differs from loader")
    gc=GVARConfig(**p["gvar"])
    if gc.variant!="volume_rel": raise ValueError("Unexpected GVAR reader configuration")
    if verify: verify_main_contract()
    if p["code_sha256"]!=code_fingerprint():
        raise RuntimeError("Checkpoint source fingerprint changed; audit before inference")
    configure_main(mc,p["dataset_root"],p["label_folder"],p["use_fuse_depth"])
    dav=ROOT/"checkpoints"/f"depth_anything_v2_{mc.encoder}.pth"
    if sha256_file(dav)!=p["dav2_sha256"]: raise RuntimeError("Pretrained DAV2 weights changed")
    model=EconomicGraspPointGVAR(mc,enable_gvar=p["variant"]=="rayrope_point_gvar",
                                  gvar_config=gc).to(device)
    model.load_state_dict(ck["model"],strict=True)
    model.eval()
    return model,ck
