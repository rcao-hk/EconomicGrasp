"""Checkpoint protocols and DPT-only inference; never instantiate GVAR/CVA.

Only trusted local user checkpoints may be torch.load'ed (pickle). Strict key
matching avoids mixing historical Pose-FiLM and calibrated-height models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import torch


FAMILIES=("height_analytic","global_film")


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read_checkpoint_registry(path, *, expected_epochs=None):
    obj=json.loads(Path(path).read_text())
    rows=obj["checkpoints"]
    if not isinstance(rows,list) or not rows:
        raise ValueError("Nonempty 'checkpoints' list required.")
    seen=set(); result=[]
    for e in rows:
        family=str(e["family"]); epoch=int(e["epoch"])
        cp=Path(e["path"]).expanduser()
        if not cp.is_absolute(): cp=Path(path).resolve().parent/cp
        cp=cp.resolve()
        if family not in FAMILIES or epoch<1: raise ValueError("Invalid family/epoch")
        if (family,epoch) in seen:raise ValueError("Duplicate family+epoch.")
        if not cp.is_file():raise FileNotFoundError(str(cp))
        seen.add((family,epoch))
        result.append(dict(family=family,epoch=epoch,path=str(cp),sha256=sha256(cp)))
    if expected_epochs is None: expected_epochs=obj.get("expected_epochs",[])
    expected_epochs=sorted(set(map(int,expected_epochs)))
    if any(e<=0 for e in expected_epochs):raise ValueError("Expected epochs >0.")
    return result,expected_epochs


def inspect_checkpoint(ck, family, epoch):
    if not isinstance(ck,dict) or "model_state_dict" not in ck:
        raise ValueError("Require full checkpoint with model_state_dict; no raw/partial checkpoint.")
    saved_epoch=ck.get("epoch")
    if saved_epoch is None or int(saved_epoch)!=epoch:
        raise ValueError(f"Checkpoint epoch mismatch expected {epoch}, got {saved_epoch}.")
    if ck.get("pose_depth_mode")!=family:
        raise ValueError(f"Expected depth={family}, saved {ck.get('pose_depth_mode')!r}.")
    grasp=ck.get("grasp_design")
    if not isinstance(grasp,dict) or not (
        grasp.get("enable_point_rayrope") and grasp.get("enable_gvar")
    ):
        raise ValueError("Only Point RayRoPE+GVAR checkpoints are allowed.")
    mix=ck.get("gntrans_mix_protocol")
    if not isinstance(mix,dict):
        raise ValueError("Missing GN-Trans MIX50 sampling protocol.")
    if (abs(float(mix.get("train_fraction_each_domain",-1))-.5)>1e-8 or
        int(mix.get("original_train_count",-1))!=12800 or
        int(mix.get("gntrans_train_count",-1))!=12800):
        raise ValueError("Not the registered 12800+12800 mix50 protocol.")
    if family=="height_analytic":
        h=ck.get("height_analytic_protocol")
        if (not isinstance(h,dict) or h.get("mode")!="height_analytic" or
            h.get("no_pose_ray_plane_film") is not True):
            raise ValueError("Missing certified height/no-FiLM checkpoint protocol.")
        bounds=[float(h[k]) for k in ("height_min_m","height_max_m","height_min_downward_dot")]
        if not (-3<bounds[0]<bounds[1]<3) or not (0<bounds[2]<1):
            raise ValueError("Invalid trained height bounds.")
    else:
        if ck.get("height_analytic_protocol") not in (None,{}):
            raise ValueError("Pose baseline unexpectedly carries a height protocol.")
    state=ck["model_state_dict"]
    if not isinstance(state,dict) or not state:
        raise ValueError("Empty model_state_dict.")
    ours={k[len("depth_net."):]:v for k,v in state.items() if k.startswith("depth_net.")}
    if not ours:raise ValueError("No depth_net submodule found.")
    has_pose=any(k.startswith("pose_aware_adapter.") for k in ours)
    if has_pose != (family=="global_film"):
        raise ValueError(f"FiLM state-tree mismatch family={family} has_pose={has_pose}")
    has_height="height_analytic_signature" in ours
    if has_height!=(family=="height_analytic"):
        raise ValueError("Height signature mismatch.")
    if family=="height_analytic":
        if not torch.allclose(ours["height_analytic_signature"].float().cpu(),
               torch.tensor(bounds,dtype=torch.float32),atol=1e-6):
            raise ValueError("Checkpoint height decoder settings disagree with tensor signature.")
    return ours, {
        "family":family,"epoch":epoch,"train_per_domain":12800,
        "checkpoint_protocol_height":ck.get("height_analytic_protocol"),
        "num_depth_tensors":len(ours),
        "grasp_design":grasp,
    }


def load_depth_model(record, device):
    # Existing production imports utils.arguments at module import. Prevent the
    # experiment CLI flags from being consumed by that legacy global parser.
    argv=sys.argv
    try:
        sys.argv=[argv[0]]
        from models.economicgrasp_depth import DINOv2DepthRegressionNet
    finally:
        sys.argv=argv
    ck=torch.load(record["path"],map_location="cpu",weights_only=False)
    state,audit=inspect_checkpoint(ck,record["family"],record["epoch"])
    cfg=dict(encoder="vitb",stride=1,min_depth=.2,max_depth=1.,
             freeze_backbone=True,pose_depth_mode=record["family"])
    if record["family"]=="height_analytic":
        h=ck["height_analytic_protocol"]
        cfg.update(height_min_m=float(h["height_min_m"]),
                   height_max_m=float(h["height_max_m"]),
                   height_min_downward_dot=float(h["height_min_downward_dot"]))
    model=DINOv2DepthRegressionNet(**cfg)
    model.load_state_dict(state,strict=True)
    if bool(model.pose_aware_adapter is None)!=(record["family"]=="height_analytic"):
        raise ValueError("Executed depth architecture != checkpoint protocol.")
    model.to(device).eval()
    audit["model_args"]=cfg
    return model,audit


@torch.inference_mode()
def predict(model,family,batch,device,*,use_trans=False):
    x=batch["rgb_trans" if use_trans else "rgb"].to(device,non_blocking=True)
    K=batch["K"].to(device,non_blocking=True)
    if family=="height_analytic":
        z,_,_,aux=model(x,camera_K=K,camera_plane=batch["plane"].to(device),
                        return_pose_aux=True)
        h=aux["height_map_pred"]
        eligible=aux["height_ray_valid_mask"]
    else:
        z,_,_=model(x,camera_pose_vec=batch["pose"].to(device),camera_K=K)
        h=None;eligible=None
    if not torch.isfinite(z).all():
        raise FloatingPointError(f"{family}: nonfinite depth output.")
    return {"z":z[:,0].float().cpu().numpy(),
            "h":h[:,0].float().cpu().numpy() if h is not None else None,
            "eligible":eligible[:,0].cpu().numpy() if eligible is not None else None}
