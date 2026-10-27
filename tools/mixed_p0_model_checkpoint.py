"""Shared strict checkpoint compatibility for P0-B and P0-D.

The historical mixed-v2 Point/GVAR runs disable activation checkpointing and
use 2048 grouping/action chunks. These performance-only settings are read from
the saved grasp_design, never silently assumed from current-main defaults.
Model weights, module shapes and architecture flags must still load strictly.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch


def read_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError("Expected checkpoint dict with model weights and grasp_design metadata.")
    options = ("model_state_dict", "model", "state_dict", "model_state")
    state = next((obj[k] for k in options if k in obj and isinstance(obj[k], dict)), None)
    if state is None:
        raise KeyError(f"Expected one of {options} for model state in checkpoint: {path}")
    if "grasp_design" not in obj or not isinstance(obj["grasp_design"], dict):
        raise KeyError(f"Checkpoint missing grasp_design metadata; refuse guessing Point/GVAR flags: {path}")
    d = obj["grasp_design"]
    for key in ("enable_point_rayrope","enable_gvar"):
        if key not in d:
            raise KeyError(f"Checkpoint grasp_design lacks {key}")
    return obj, state


def design_init_options(checkpoint):
    d = checkpoint["grasp_design"]
    has_point = bool(d["enable_point_rayrope"])
    has_gvar = bool(d["enable_gvar"])
    if not (has_point or has_gvar):
        return dict(enable_point_rayrope=False, enable_gvar=False,
                    rayrope_group_chunk=32,gvar_action_chunk=256)
    if has_point and not isinstance(d.get("rayrope"), dict):
        raise KeyError("Point RayRoPE checkpoint has no detailed rayrope config")
    if has_gvar and not isinstance(d.get("gvar"), dict):
        raise KeyError("GVAR checkpoint has no detailed gvar config")
    return dict(enable_point_rayrope=has_point,enable_gvar=has_gvar,
        rayrope_group_chunk=int(d["rayrope"].get("group_chunk",32)) if has_point else 32,
        gvar_action_chunk=int(d["gvar"].get("action_chunk",256)) if has_gvar else 256)


def apply_runtime_checkpoint_flags(model, checkpoint):
    """Only change non-parameter checkpoint toggles; architecture remains strict."""
    d = checkpoint["grasp_design"]
    group_module = model.kview_grasp_module
    if bool(d["enable_point_rayrope"]):
        saved = bool(d["rayrope"].get("checkpoint_chunks",True))
        group = group_module.group
        if not hasattr(group,"cfg"):
            raise RuntimeError("RayRoPE group lacks config; checkpoint architecture mismatch.")
        group.cfg = replace(group.cfg,checkpoint_chunks=saved)
        model.grasp_design["rayrope"]["checkpoint_chunks"] = saved
    if bool(d["enable_gvar"]):
        saved = bool(d["gvar"].get("activation_checkpoint",True))
        decoder=group_module.decoder
        if not hasattr(decoder,"gvar_config") or decoder.reader is None:
            raise RuntimeError("Missing GVAR reader/config in constructed architecture.")
        cfg=replace(decoder.gvar_config, activation_checkpoint=saved)
        decoder.gvar_config=cfg
        decoder.reader.cfg=cfg
        model.grasp_design["gvar"]["activation_checkpoint"]=saved
    from models.grasp_designs import validate_checkpoint_design
    validate_checkpoint_design(model,checkpoint)


def load_strict_model(model, checkpoint, state):
    apply_runtime_checkpoint_flags(model,checkpoint)
    if state and all(str(k).startswith("module.") for k in state):
        state={k[len("module."):]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    return model
