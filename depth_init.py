"""Explicit canonical initialization and FP32 view-gradient policy.

The architecture reference supplies constructor metadata, never cold weights.
Imports stay lazy so CLI/help and argument checks need no CUDA environment.
"""
from pathlib import Path
import json


ARCHITECTURE_KEYS = (
    "distill_stage", "seed_selection_mode", "geometry_depth_source", "pose_depth_mode",
    "use_fuse_depth", "camera_pose_key", "camera_gravity_key", "pose_hidden_dim",
    "ray_gravity_hidden_dim", "ray_gravity_mid_dim",
)


def assert_resume_settings(state, args):
    for key, default in (("init_mode", "warm"), ("depth_gradient_policy", "normal"), ("probe_schedule", "legacy"),
                         ("deterministic_ops", False)):
        if state["arguments"].get(key, default) != getattr(args, key, default):
            raise ValueError(f"Resume diagnostic setting mismatch: {key}")


def read_source(args, sha256):
    import torch
    if args.canonical_init and args.mode != "initialize":
        path = Path(args.canonical_init)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("canonical_init_version") != 1 or state["init_mode"] != args.init_mode:
            raise ValueError("Canonical initialization version/mode mismatch")
        if state["seed"] != args.seed:
            raise ValueError("Canonical initialization seed mismatch")
        return state["architecture"], state, path
    path = Path(args.architecture_checkpoint if args.init_mode != "warm" else args.init_checkpoint)
    reference = torch.load(path, map_location="cpu", weights_only=False)
    architecture = {k: reference[k] for k in ARCHITECTURE_KEYS if k in reference}
    provenance = {"mode": args.init_mode, "architecture_reference": str(path.resolve()),
                  "architecture_reference_sha256": sha256(path), "historical_failure_reproduced": False,
                  "architecture_reference_weights_loaded": args.init_mode == "warm"}
    state = {"architecture": architecture, "provenance": provenance}
    if args.init_mode == "warm":
        state["model_state_dict"] = reference["model_state_dict"]
        provenance["source_has_optimizer"] = "optimizer_state_dict" in reference
        provenance["source_epoch"] = reference.get("epoch", 0)
    elif args.init_mode == "early":
        recipe = json.loads(Path(args.early_recipe).read_text())
        if not recipe.get("description") or not recipe.get("include_prefixes"):
            raise ValueError("Early recipe requires description and explicit include_prefixes")
        source = Path(args.init_checkpoint)
        if sha256(source) != recipe["source_sha256"]:
            raise ValueError("Early recipe checkpoint SHA256 mismatch")
        early = torch.load(source, map_location="cpu", weights_only=False)
        state["early_weights"] = early["model_state_dict"]
        state["early_recipe"] = recipe
        provenance.update(source_checkpoint=str(source.resolve()), source_sha256=recipe["source_sha256"],
                          recipe=recipe, source_has_optimizer="optimizer_state_dict" in early)
    return architecture, state, path


def apply_source(model, state):
    if "model_state_dict" in state:
        result = model.load_state_dict(state["model_state_dict"], strict=True)
        return {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
                "skipped_keys": [], "loaded_keys": list(state["model_state_dict"])}
    if "early_weights" in state:
        recipe, current = state["early_recipe"], model.state_dict()
        compatible, skipped = {}, []
        for source_name, value in state["early_weights"].items():
            name = source_name.removeprefix(recipe.get("strip_prefix", ""))
            if not any(name.startswith(p) for p in recipe["include_prefixes"]):
                skipped.append({"key": source_name, "reason": "not_selected"})
            elif name not in current or current[name].shape != value.shape:
                skipped.append({"key": source_name, "reason": "missing_or_shape_mismatch"})
            else:
                compatible[name] = value
        if not compatible:
            raise ValueError("Early recipe selected no compatible weights")
        result = model.load_state_dict(compatible, strict=False)
        return {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
                "skipped_keys": skipped, "loaded_keys": list(compatible)}
    return {"missing_keys": [], "unexpected_keys": [], "skipped_keys": [], "loaded_keys": [],
            "meaning": "current constructor only; no task-trained checkpoint loaded"}


def save_canonical(model, args, architecture, source, loading, sha256):
    import torch
    import depth_dynamics as dd
    path = Path(args.canonical_init).resolve()
    if path.exists():
        raise FileExistsError(f"Canonical state already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = dict(source["provenance"])
    weights = dd._cpu_clone(model.state_dict())
    parameter_sources = {name: "source_checkpoint" if name in loading["loaded_keys"] else
                         "constructor_seed" for name in weights}
    if args.init_mode != "warm":
        backbone = Path("checkpoints/depth_anything_v2_vitb.pth").resolve()
        pretrained = torch.load(backbone, map_location="cpu", weights_only=False)
        loaded, skipped = [], []
        for name, value in pretrained.items():
            target = "depth_net.depthnet." + name
            if "pretrained" in name:
                if target not in weights or not torch.equal(weights[target], value.cpu()):
                    raise ValueError(f"Backbone provenance check failed: {target}")
                loaded.append(name)
                parameter_sources[target] = "backbone_pretrained"
            else:
                skipped.append(name)
        provenance["backbone"] = {"path": str(backbone), "sha256": sha256(backbone),
                                  "loaded_keys": loaded, "skipped_keys": skipped,
                                  "decoder_loaded": False}
        if not loaded:
            raise ValueError("No pretrained backbone keys verified")
    manifest = {"canonical_init_version": 1, "init_mode": args.init_mode, "seed": args.seed,
                "architecture": architecture, "provenance": provenance, "loading": loading,
                "parameter_sources": parameter_sources, "optimizer": "fresh AdamW for every arm",
                "training_rng": "explicit restore of seed-0 constructor RNG after loading/probe-cache creation; never source checkpoint RNG"}
    state = dict(manifest, model_state_dict=weights, training_rng_state=source["training_rng_state"])
    with path.open("xb") as handle:
        torch.save(state, handle)
    manifest.update(path=str(path), sha256=sha256(path))
    dd.write_json(path.with_suffix(".manifest.json"), manifest)
    return manifest


def depth_scope(groups):
    return [(name, p) for group, pairs in groups.items()
            if group == "depth_film" or group.startswith("depth_dpt.")
            for name, p in pairs if p.requires_grad]


def backward_with_policy(loss, weighted_view, scope, policy, all_parameters=None):
    """One forward; remove only the current weighted view derivative before clip."""
    import torch
    if policy == "normal":
        loss.backward()
        return {}
    if policy not in ("remove_view_reclip", "view_noop") or not scope:
        raise ValueError("Invalid/nonempty depth gradient policy scope")
    if any(p.dtype != torch.float32 for _, p in scope):
        raise ValueError("View-gradient policy is FP32 only")
    gradients = torch.autograd.grad(weighted_view, [p for _, p in scope], retain_graph=True,
                                    allow_unused=True)
    loss.backward()
    global_norm = float(torch.nn.utils.clip_grad_norm_(list(all_parameters) if all_parameters is not None else
                        [p for _, p in scope], float("inf"), error_if_nonfinite=True))
    full = float(torch.nn.utils.clip_grad_norm_([p for _, p in scope], float("inf"), error_if_nonfinite=True))
    for (_, parameter), gradient in zip(scope, gradients):
        if gradient is not None:
            if parameter.grad is None:
                raise ValueError("View derivative connected but total derivative is None")
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Nonfinite view derivative")
            if policy == "remove_view_reclip":
                parameter.grad.sub_(gradient)
    return {"global_grad_before_removal": global_norm, "depth_scope_full_grad_norm": full,
            "view_connected_parameters": sum(g is not None for g in gradients),
            "view_grad_norm": sum(float(g.detach().square().sum()) for g in gradients if g is not None) ** .5}
