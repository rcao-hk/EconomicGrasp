"""Single-GPU, paired depth-gradient diagnostics on the existing CVA trainer.

This entry point intentionally does not import the repository argument parser
until its own options have been removed. ``--help`` needs no CUDA libraries.
Run one process per arm. Data order and per-example preprocessing randomness
are keyed by (seed, epoch, index), independently of diagnostic forwards.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import depth_init


LOSS_KEYS = {
    "depth": "B: DepthReg Loss", "objectness": "B: Objectness Loss",
    "graspness": "B: Graspness Loss", "view": "B: View Loss",
    "cdf": "B: CDF Loss", "width": "B: Width Loss",
}
WEIGHT_KEYS = {
    "depth": "depth_prob_loss_weight", "objectness": "objectness_loss_weight",
    "graspness": "graspness_loss_weight", "view": "view_loss_weight",
    "cdf": "score_loss_weight", "width": "width_loss_weight",
}
FORMAT_VERSION = 1
EVENT_STATE_VERSION = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognized options are forwarded to utils.arguments (e.g. "
               "--dataset_root, --batch_size, --learning_rate, loss weights).")
    parser.add_argument("--mode", choices=("initialize", "audit", "train"), default="audit")
    parser.add_argument("--init_checkpoint", default="", help="Warm weights or explicitly documented early source.")
    parser.add_argument("--init_mode", choices=("warm", "current_standard_cold", "early"), default="warm")
    parser.add_argument("--architecture_checkpoint", default="", help="Cold/early constructor metadata only; weights are NOT loaded.")
    parser.add_argument("--canonical_init", default="", help="Create once in initialize mode; immutable shared weights in audit/train.")
    parser.add_argument("--early_recipe", default="", help="JSON source SHA256, description and explicit include_prefixes.")
    parser.add_argument("--depth_gradient_policy", choices=("normal", "remove_view_reclip", "view_noop"), default="normal")
    parser.add_argument("--probe_schedule", choices=("legacy", "cold"), default="legacy")
    parser.add_argument("--output", required=True, help="One NEW arm directory, or existing directory with --resume_checkpoint.")
    parser.add_argument("--diagnostics_dir", help="Arm-specific diagnostic directory; default OUTPUT/diagnostics.")
    parser.add_argument("--arm", default="D0")
    parser.add_argument("--routes", default="none", help="none, all, gse, seed_xyz, support, or comma-separated routes")
    parser.add_argument("--max_steps", type=int, default=50, help="Total optimizer updates; extend 50 -> 500 -> 2000 with resume.")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit_batches", type=int, default=8)
    parser.add_argument("--audit_routes", default="none,all,gse,seed_xyz,support")
    parser.add_argument("--audit_interval", type=int, default=200, help="0 disables heavy replay audits during training.")
    parser.add_argument("--probe_interval", type=int, default=100)
    parser.add_argument("--train_probe_frames", type=int, default=16)
    parser.add_argument("--validation_probe_frames", "--heldout_test_probe_frames", dest="heldout_test_probe_frames", type=int, default=32,
                        help="User-designated test_seen validation frames (32 by default).")
    parser.add_argument("--directional_batches", type=int, default=1)
    parser.add_argument("--save_gradient_maps", action="store_true",
                        help="Save first fixed audit image's task/depth metric and raw gradients per route.")
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--keep_last", type=int, default=6)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--lr_schedule", choices=("constant", "source_cosine"), default="constant")
    parser.add_argument("--forward_atol", type=float, default=1e-6)
    parser.add_argument("--forward_rtol", type=float, default=1e-5)
    parser.add_argument("--replay_policy", choices=("strict", "calibrated"), default="strict",
                        help="Replay acceptance: strict original tolerance, or exact-state checks plus a locked CUDA update cap.")
    parser.add_argument("--skip_initial_audit", action="store_true",
                        help="Only for already-audited paired runs; provide --audit_contract.")
    parser.add_argument("--audit_contract", "--audit_gate", dest="audit_contract", default="",
                        help="Successful P0 gate to authorize skipping redundant P0.")
    parser.add_argument("--verify_diagnostic_step", action="store_true",
                        help="Verify one real optimizer update is identical with/without a diagnostic replay.")
    args, remaining = parser.parse_known_args(argv)
    if args.mode == "initialize":
        if not args.canonical_init or args.resume_checkpoint:
            parser.error("initialize requires a new --canonical_init and forbids resume")
        if Path(args.canonical_init).exists():
            parser.error("Canonical initialization already exists")
    if not args.canonical_init or args.mode == "initialize":
        if args.init_mode == "warm" and not args.init_checkpoint:
            parser.error("warm requires --init_checkpoint or an existing --canonical_init")
        if args.init_mode != "warm" and not args.architecture_checkpoint:
            parser.error("cold/early creation requires --architecture_checkpoint (metadata only)")
        if args.init_mode == "early" and (not args.init_checkpoint or not args.early_recipe):
            parser.error("early requires --init_checkpoint and --early_recipe")
    if args.init_mode != "warm" and args.mode != "initialize" and not args.canonical_init:
        parser.error("cold/early arms must reuse --canonical_init")
    if args.depth_gradient_policy != "normal" and args.routes != "all":
        parser.error("Selective view policy requires all E/Q/C routes open")
    for name in ("max_steps", "audit_batches", "probe_interval", "checkpoint_interval", "train_probe_frames"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.keep_last < 6:
        parser.error("--keep_last must be at least 6 to retain pre-event states")
    if args.heldout_test_probe_frames < 0 or args.audit_interval < 0:
        parser.error("probe/audit counts must be nonnegative")
    if args.skip_initial_audit and not args.audit_contract:
        parser.error("--skip_initial_audit requires --audit_contract")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one process and one visible GPU per arm; DDP is not supported here")
    forbidden = ("--resume", "--checkpoint_path", "--distill_stage", "--log_dir",
                 "--use_obs_depth", "--use_gt_depth", "--use_depth_comp", "--use_top4_view_infer")
    for value in remaining:
        if value.split("=", 1)[0] in forbidden:
            parser.error(f"{value} conflicts with the diagnostic contract")
    return args, remaining


def probe_due(step, args):
    if args.probe_schedule == "cold":
        return step in (0, 10, 25, 50, 100) or (100 < step <= 1000 and step % 50 == 0) or (step > 1000 and step % 100 == 0)
    return step % args.probe_interval == 0


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "numel") and value.numel() == 1:
        return json_safe(value.detach().cpu().item())
    return str(value)


def git_info():
    def run(*parts):
        return subprocess.check_output(["git", *parts], cwd=Path(__file__).parent).decode("utf-8", "replace").strip()
    try:
        diff = run("diff", "HEAD", "--")
        return {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
                "status": run("status", "--short"), "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "tracked_diff": diff}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:4], "little")


def advance_flat_event_state(previous, *, step, fraction_flat):
    """Advance fixed-probe events without calling initial flatness a collapse.

    Initially nonflat runs are immediately eligible. Initially flat runs need
    three consecutive nonflat probes before a subsequent flat event is eligible.
    "Nonflat" describes the existing std-ratio criterion, not depth accuracy.
    Legacy warm checkpoints lack a recorded reference fraction: their first
    eligible resumed probe establishes a prospective baseline. Mild changes
    before that observation cannot be dated; existing event timestamps survive.
    """
    if isinstance(step, bool) or int(step) != step or step < 0:
        raise ValueError("Event step must be a nonnegative integer")
    step = int(step)
    if fraction_flat is not None:
        fraction_flat = float(fraction_flat)
        if not math.isfinite(fraction_flat) or not 0. <= fraction_flat <= 1.:
            raise ValueError("Flat-image fraction must be finite in [0,1], or None")
    state = dict(previous or {})
    legacy = state.get("event_state_version") is None and state.get("initial_flat") is not None
    defaults = {"initial_flat": None, "first_anomaly": None, "first_threshold": None,
                "confirmed_event": None, "consecutive_thresholds": 0,
                "initial_observation_step": None, "nonflat_established": False,
                "first_nonflat_step": None, "nonflat_confirmed_step": None,
                "nonflat_candidate_step": None, "consecutive_nonflat": 0,
                "reference_flat_fraction": None, "event_origin": None,
                "migration_reference_unknown": False,
                "migration_observation_baseline_step": None,
                "migration_observation_baseline_fraction": None,
                "last_observation_step": None, "last_fraction_flat": None}
    for key, value in defaults.items():
        state.setdefault(key, value)
    if legacy:
        # Old warm runs were eligible from initialization. Old flat runs did
        # not record whether they later gained structure; do not invent history.
        state["nonflat_established"] = state["initial_flat"] is False or state["first_threshold"] is not None
        state["migration_reference_unknown"] = state["nonflat_established"] and state["reference_flat_fraction"] is None
        state["migration_note"] = (
            "legacy event timestamps retained; unknown nonflat timestamps are not inferred; "
            "an unknown warm reference is observed at the first eligible resumed probe, "
            "so earlier mild changes cannot be dated; flat-start emergence is observed prospectively")
    state["event_state_version"] = EVENT_STATE_VERSION
    last_step = state["last_observation_step"]
    if last_step is not None and step <= last_step:
        if step == last_step and fraction_flat == state["last_fraction_flat"]:
            return state  # Repeated diagnostics at one update are not new evidence.
        raise ValueError("Event observations must advance optimizer steps; conflicting/reordered probe")
    state["last_observation_step"], state["last_fraction_flat"] = step, fraction_flat
    if fraction_flat is None:
        state["consecutive_thresholds"] = 0
        state["consecutive_nonflat"] = 0
        state["nonflat_candidate_step"] = None
        state["observation_status"] = "no_eligible_images"
        return state
    flat = fraction_flat >= .8
    state["observation_status"] = "flat_threshold" if flat else "nonflat_threshold"
    if state["initial_flat"] is None:
        state["initial_flat"] = flat
        state["initial_observation_step"] = step
        if not flat:
            state.update(nonflat_established=True, first_nonflat_step=step,
                         nonflat_confirmed_step=step, reference_flat_fraction=fraction_flat)
        return state
    if not state["nonflat_established"]:
        if flat:
            state["consecutive_nonflat"] = 0
            state["nonflat_candidate_step"] = None
        else:
            if state["first_nonflat_step"] is None:
                state["first_nonflat_step"] = step
            if state["consecutive_nonflat"] == 0:
                state["nonflat_candidate_step"] = step
            state["consecutive_nonflat"] += 1
            if state["consecutive_nonflat"] >= 3:
                state.update(nonflat_established=True, nonflat_confirmed_step=step,
                             reference_flat_fraction=fraction_flat)
        state["consecutive_thresholds"] = 0
        return state
    if state["migration_reference_unknown"] and state["migration_observation_baseline_step"] is None:
        state["migration_observation_baseline_step"] = step
        state["migration_observation_baseline_fraction"] = fraction_flat
        state["reference_flat_fraction"] = fraction_flat
    reference = state["reference_flat_fraction"]
    if state["first_anomaly"] is None and (flat or (reference is not None and fraction_flat > reference)):
        state["first_anomaly"] = step
    if not flat and (reference is None or fraction_flat < reference):
        state["reference_flat_fraction"] = fraction_flat
    if flat and state["first_threshold"] is None:
        state["first_threshold"] = step
        state["event_origin"] = "lost_after_nonflat_emergence" if state["initial_flat"] else "loss_of_initial_nonflat_structure"
    state["consecutive_thresholds"] = state["consecutive_thresholds"] + 1 if flat else 0
    if state["consecutive_thresholds"] >= 3 and state["confirmed_event"] is None:
        state["confirmed_event"] = step
    return state


def assert_resume_outputs_not_newer(output, diagnostics, checkpoint_step):
    """Read-only guard against appending a branched history to an existing run."""
    try:
        integer_step = int(checkpoint_step)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Resume checkpoint step must be a nonnegative integer") from error
    if isinstance(checkpoint_step, bool) or integer_step != checkpoint_step or integer_step < 0:
        raise ValueError("Resume checkpoint step must be a nonnegative integer")
    checkpoint_step = integer_step
    folders = {Path(output).resolve(), Path(diagnostics).resolve()}
    checked = {}
    newer = []

    def observe(path, value, location):
        if isinstance(value, bool):
            raise ValueError(f"Malformed optimizer step in {path}:{location}")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Malformed optimizer step in {path}:{location}") from error
        if not math.isfinite(number) or number < 0 or int(number) != number:
            raise ValueError(f"Malformed optimizer step in {path}:{location}")
        number = int(number)
        checked[str(path)] = max(number, checked.get(str(path), -1))
        if number > checkpoint_step:
            newer.append({"path": str(path), "location": location, "step": number})

    paths = {path for folder in folders if folder.exists() for pattern in ("*.jsonl", "*.csv")
             for path in folder.glob(pattern)}
    for path in sorted(paths):
        try:
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(f"Non-object JSONL record in {path}:{line_number}")
                        if "step" in row:
                            observe(path, row["step"], f"line {line_number}")
            else:
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle, strict=True)
                    fields = reader.fieldnames
                    if fields is None:
                        continue
                    if any(not field for field in fields) or len(set(fields)) != len(fields):
                        raise csv.Error("Missing or duplicate CSV column name")
                    for row in reader:
                        if None in row or any(value is None for value in row.values()):
                            raise csv.Error(f"Missing or extra CSV columns at line {reader.line_num}")
                        if "step" in fields:
                            observe(path, row["step"], f"line {reader.line_num}")
        except (json.JSONDecodeError, csv.Error) as error:
            raise ValueError(f"Cannot safely resume: malformed log {path}; use a new run directory") from error
    for folder in folders:
        for filename in ("latest_checkpoint.json", "checkpoints_manifest.json", "contract.json", "failure.json"):
            path = folder / filename
            if not path.exists():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Cannot safely resume: malformed metadata {path}") from error
            if filename == "checkpoints_manifest.json":
                if not isinstance(value, list):
                    raise ValueError(f"Malformed checkpoint manifest {path}")
                for index, entry in enumerate(value):
                    if not isinstance(entry, dict):
                        raise ValueError(f"Malformed checkpoint manifest entry in {path}:{index}")
                    observe(path, entry.get("step"), f"entry {index}")
            elif filename == "contract.json":
                if not isinstance(value, dict):
                    raise ValueError(f"Malformed contract {path}")
                for field in ("completion", "resume", "audit"):
                    if isinstance(value.get(field), dict) and "step" in value[field]:
                        observe(path, value[field]["step"], field)
            else:
                if not isinstance(value, dict):
                    raise ValueError(f"Malformed metadata {path}")
                if "step" in value:
                    observe(path, value["step"], filename)
        if folder.exists():
            for path in folder.glob("fixed_frame_*.pt"):
                match = re.fullmatch(r"fixed_frame_(\d+)\.pt", path.name)
                if match:
                    observe(path, match.group(1), "filename")
    checkpoint_dir = Path(output) / "checkpoints"
    if checkpoint_dir.exists():
        for path in checkpoint_dir.glob("step_*.*"):
            match = re.fullmatch(r"step_(\d+)\.(pt|partial)", path.name)
            if match:
                observe(path, match.group(1), "filename")
    if newer:
        evidence = "; ".join(f"{entry['path']} ({entry['location']}, step {entry['step']})" for entry in newer[:5])
        raise ValueError(f"Cannot resume checkpoint step {checkpoint_step} into outputs containing newer history: {evidence}. "
                         "Use a new output and diagnostics directory; existing logs are never truncated automatically.")
    return {"checkpoint_step": checkpoint_step, "max_step_by_file": checked, "status": "no_newer_history"}


def compare_state(left, right, *, atol=0.0, rtol=0.0, path="state"):
    """Compare nested optimizer/RNG state without conflating None and zeros."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)) or left.shape != right.shape:
            return [path + ": tensor shape/type"]
        a, b = left.detach().cpu(), right.detach().cpu()
        equal = torch.allclose(a, b, atol=atol, rtol=rtol) if a.is_floating_point() else torch.equal(a, b)
        return [] if equal else [path + ": tensor values"]
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return [] if isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right) else [path + ": numpy values"]
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return [path + ": keys"]
        return [error for key in left for error in compare_state(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}")]
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [path + ": length"]
        return [error for i, (a, b) in enumerate(zip(left, right)) for error in compare_state(a, b, atol=atol, rtol=rtol, path=f"{path}.{i}")]
    return [] if left == right else [path + ": values"]


class DeterministicStream:
    """Real shuffled epochs, no worker prefetch and an exactly serializable cursor."""
    def __init__(self, dataset, batch_size, seed, collate, capture_rng, restore_rng):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.collate, self.capture_rng, self.restore_rng = collate, capture_rng, restore_rng
        self.epoch, self.offset = 0, 0

    def permutation(self):
        return np.random.default_rng(stable_seed(self.seed, "order", self.epoch)).permutation(len(self.dataset))

    def sample(self, index, epoch=0):
        state = self.capture_rng()
        seed = stable_seed(self.seed, "sample", epoch, index)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            return self.dataset[int(index)]
        finally:
            self.restore_rng(state)

    def next(self):
        if self.offset >= len(self.dataset):
            self.epoch += 1
            self.offset = 0
        indices = self.permutation()[self.offset:self.offset + self.batch_size].tolist()
        self.offset += len(indices)
        return self.collate([self.sample(index, self.epoch) for index in indices]), indices

    def state_dict(self):
        return {"epoch": self.epoch, "offset": self.offset, "seed": self.seed,
                "dataset_size": len(self.dataset), "batch_size": self.batch_size,
                "order_algorithm": "numpy.PCG64(SHA256(seed,order,epoch)); no prefetch"}

    def load_state_dict(self, state):
        for key in ("seed", "dataset_size", "batch_size", "order_algorithm"):
            if state[key] != self.state_dict()[key]:
                raise ValueError(f"Resume data stream mismatch: {key}")
        self.epoch, self.offset = int(state["epoch"]), int(state["offset"])


def dataset_manifest(dataset):
    base = dataset.base_dataset
    rows = [{"index": i, "scene": base.scenename[i], "frame": int(base.frameid[i]),
             "rgb": base.colorpath[i]} for i in range(len(base))]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return rows, digest


def select_probe_indices(dataset, count):
    """Evenly choose scene IDs first, then the central frame in each scene."""
    scenes = {}
    for i, scene in enumerate(dataset.base_dataset.scenename):
        scenes.setdefault(scene, []).append(i)
    scene_names = sorted(scenes)
    if count > len(dataset):
        raise ValueError("Probe count exceeds dataset size")
    result = []
    for cycle in range(math.ceil(count / len(scene_names)) if count else 0):
        take = min(count - len(result), len(scene_names))
        selected = np.linspace(0, len(scene_names) - 1, take, dtype=int)
        for i in selected:
            members = scenes[scene_names[int(i)]]
            # Middle frame, then quarter frame, gives 32 unique frames in the
            # 30-scene test_seen split without pretending there are 32 scenes.
            position = (len(members) // 2 + cycle * max(1, len(members) // 4)) % len(members)
            while members[position] in result:
                position = (position + 1) % len(members)
            result.append(members[position])
    return result


def assert_rescue_resume_explicit(state, args):
    if (state.get("arguments", {}).get("rescue_branch")
            and getattr(args, "mode", "train") == "train"
            and not getattr(args, "rescue_branch", None)):
        raise ValueError("Rescue snapshot requires rescue_cva_depth_counterfactual.py with an explicit --branch; "
                         "ordinary training would silently drop the gradient intervention protocol")


class Experiment:
    def __init__(self, args, remaining):
        self.args = args
        self.output = Path(args.output).resolve()
        self.diag = Path(args.diagnostics_dir).resolve() if args.diagnostics_dir else self.output / "diagnostics"
        if not args.resume_checkpoint and self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run: {self.output}")
        resume_state = None
        self.resume_output_check = None
        if args.resume_checkpoint:
            # Run this before Trainer opens text/TensorBoard logs or we overwrite
            # any manifests. A failed guard must leave the existing run intact.
            resume_state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
            if resume_state.get("format_version") != FORMAT_VERSION or "step" not in resume_state:
                raise ValueError("Resume requires a full dynamics checkpoint")
            assert_rescue_resume_explicit(resume_state, args)
            self.resume_output_check = assert_resume_outputs_not_newer(self.output, self.diag, resume_state["step"])
            depth_init.assert_resume_settings(resume_state, args)
        self.output.mkdir(parents=True, exist_ok=True)
        self.diag.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.source, initialization, source_path = depth_init.read_source(args, dd.sha256_file)
        required = {"distill_stage": 1, "seed_selection_mode": "image_fps", "geometry_depth_source": "pred"}
        for key, value in required.items():
            if self.source.get(key) != value:
                raise ValueError(f"Stage-1 initialization requires {key}={value!r}; got {self.source.get(key)!r}")
        if "pose_depth_mode" not in self.source or "use_fuse_depth" not in self.source:
            raise ValueError("Checkpoint must declare pose_depth_mode and use_fuse_depth")
        inherited = {key: self.source[key] for key in (
            "pose_depth_mode", "camera_pose_key", "camera_gravity_key", "pose_hidden_dim",
            "ray_gravity_hidden_dim", "ray_gravity_mid_dim") if key in self.source}
        for key, value in inherited.items():
            # The source checkpoint records constructor-only metadata that is
            # intentionally absent from utils.arguments. Validate those actual
            # model values after construction; never forward nonexistent flags.
            if key != "pose_depth_mode":
                continue
            token = "--" + key
            explicit = None
            for i, item in enumerate(remaining):
                if item == token:
                    explicit = remaining[i + 1]
                elif item.startswith(token + "="):
                    explicit = item.split("=", 1)[1]
            if explicit is not None and str(explicit) != str(value):
                raise ValueError(f"{token} must inherit checkpoint value {value!r}")
            if explicit is None:
                remaining += [token, str(value)]
        if self.source["use_fuse_depth"]:
            if "--use_fuse_depth" not in remaining:
                remaining += ["--use_fuse_depth"]
        elif "--use_fuse_depth" in remaining:
            raise ValueError("--use_fuse_depth conflicts with initialization")
        # Existing Trainer remains owner of the model, label adapter and optimizer.
        sys.argv = [sys.argv[0], *remaining, "--distill_stage", "1", "--use_cdf", "--multi_modal",
                    "--extend_angle", "--seed", str(args.seed), "--log_dir", str(self.output),
                    "--num_workers", "0", "--eval_num_workers", "0"]
        import train_cva_distill_ddp as original
        self.original, self.cfg = original, original.cfgs
        self.trainer = original.Trainer()
        self.model, self.optimizer = self.trainer.unwrap_model(), self.trainer.optimizer
        for key, expected in inherited.items():
            if getattr(self.model, key, None) != expected:
                raise ValueError(f"Checkpoint model contract mismatch: {key}={expected!r}, "
                                 f"actual={getattr(self.model, key, None)!r}")
        if not hasattr(self.model, "set_depth_grad_routes"):
            raise RuntimeError("Model lacks the E/Q/C gradient route controls")
        loading = depth_init.apply_source(self.model, initialization)
        canonical_manifest = None
        if args.mode == "initialize":
            initialization["training_rng_state"] = dd.capture_rng_state()
            canonical_manifest = depth_init.save_canonical(self.model, args, self.source, initialization,
                                                          loading, dd.sha256_file)
        self.model.set_depth_grad_routes(args.routes)
        self.groups = dd.parameter_groups(self.model)
        self.depth_gradient_scope = depth_init.depth_scope(self.groups)
        self.stream = DeterministicStream(self.trainer.TRAIN_DATASET, self.cfg.batch_size, args.seed,
                                          original.collate_fn, dd.capture_rng_state, dd.restore_rng_state)
        self.weights = {key: float(getattr(self.cfg, attr)) for key, attr in WEIGHT_KEYS.items()}
        self.step, self.seen_images = 0, 0
        self.initial_epoch = int(initialization.get("provenance", {}).get("source_epoch", 0))
        self.events = {"initial_flat": None, "first_anomaly": None, "first_threshold": None,
                       "confirmed_event": None, "consecutive_thresholds": 0,
                       "event_state_version": EVENT_STATE_VERSION}
        self.manifest = []
        self.locked_steps = set()
        self.probe_previous = {}
        self.train_rows, train_sha = dataset_manifest(self.trainer.TRAIN_DATASET)
        self.probe_specs = [("train", index) for index in select_probe_indices(
            self.trainer.TRAIN_DATASET, args.train_probe_frames)]
        self.probe_specs += [("validation_test_seen", index) for index in select_probe_indices(
            self.trainer.TEST_DATASET, args.heldout_test_probe_frames)]
        self.init_sha = dd.sha256_file(args.canonical_init if args.canonical_init else source_path)
        self.contract = {
            "format_version": FORMAT_VERSION, "mode": args.mode, "arm": args.arm,
            "arguments": vars(args), "resolved_config": vars(self.cfg), "git": git_info(),
            "command": ORIGINAL_COMMAND, "environment": {"python": sys.executable, "python_version": sys.version,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "amp": False,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "custom_cuda_backward_bitwise_determinism": "not guaranteed; measured replay tolerance recorded"},
            "model": {"class": type(self.model).__qualname__, "file": inspect.getfile(type(self.model)),
                      "pose_depth_mode": self.model.pose_depth_mode, "routes": self.model.get_depth_grad_routes()},
            "checkpoint": {"path": str(Path(args.canonical_init or source_path).resolve()), "sha256": self.init_sha,
                           "metadata": {k: v for k, v in self.source.items() if k not in ("model_state_dict", "optimizer_state_dict")},
                           "initialization": "weights-only restart; AdamW reset identically for both arms",
                           "source_has_optimizer": initialization.get("provenance", {}).get("source_has_optimizer", False),
                           "init_mode": args.init_mode, "provenance": initialization.get("provenance"),
                           "canonical_manifest": canonical_manifest,
                           "missing_keys": loading["missing_keys"], "unexpected_keys": loading["unexpected_keys"],
                           "skipped_keys": loading["skipped_keys"]},
            "parameters": dd.parameter_contract(self.model, self.optimizer),
            "inputs": {"model_geometry": "RGB predicted metric depth", "pose": inherited,
                       "K": "intrinsics cropped/resized by original dataset",
                       "captured_depth": "used by original dataset crop/workspace mask/token labels; not model geometry",
                       "GT_depth": "metric-depth supervision and diagnostic metrics",
                       "strict_end_to_end_RGB_only": False, "KD": "disabled", "augmentation": False},
            "data": {"train_manifest_sha256": train_sha, "train_frames": len(self.train_rows),
                     "batch_size": self.cfg.batch_size, "world_size": 1, "accumulation": 1,
                     "loader": self.stream.state_dict(), "legal_validation_split": "test_seen; explicitly user-designated",
                     "heldout_test_used": bool(args.heldout_test_probe_frames),
                     "validation_claim": "test_seen designated by user for this diagnostic task; train probes available to source Stage1",
                     "label_cache": str(self.trainer.TRAIN_DATASET.label_root)},
            "loss": {"weights": self.weights, "depth_valid_range": [0.2, 1.0],
                     "depth_denominator": "all image pixels including invalid pixels",
                     "native_reductions": "existing get_loss_cdf; CDF mean over thresholds, each over valid samples; width valid mean"},
            "optimization": {"algorithm": "AdamW", "clip_norm": args.clip_norm,
                             "depth_gradient_policy": args.depth_gradient_policy,
                             "depth_gradient_scope": [name for name, _ in self.depth_gradient_scope],
                             "schedule": args.lr_schedule, "scheduler_state": "derived source_epoch + stream.epoch",
                             "source_epoch": self.initial_epoch, "scaler": None},
            "events": {"std_ratio_threshold": 0.1, "eligible_gt_std_min_m": 0.005,
                       "fraction_images": 0.8, "confirmations": 3, "locked_before_training": True,
                       "state_version": EVENT_STATE_VERSION, "initially_flat_nonflat_confirmations": 3,
                       "nonflat_means": "fraction_flat < 0.8; this does not assert accurate depth",
                       "event_split": "fixed train frames, eval mode", "missing_eligible_images": "unclassified; streaks reset"},
            "probe_policy": {"frames": self.probe_specs, "modes": ["train", "eval"],
                             "depth_masks_pairs": "fixed GT and same-instance pixel pairs",
                             "task_loss": "native selection/assignment under fixed frame and RNG; not frozen physical-query loss",
                             "eval_selection": "module.eval plus all is_training=False plus forced labels"},
            "pending": ["frozen physical-query task-loss trajectory", "failure-conditioned P3/P4 interventions",
                        "seed/radius/residual boundary derivative instrumentation", "three-seed confirmation", "final AP"],
            "p0_passed": False,
        }
        dd.write_json(self.diag / "train_manifest.json", self.train_rows)
        validation_rows, validation_sha = dataset_manifest(self.trainer.TEST_DATASET)
        self.contract["data"]["validation_manifest_sha256"] = validation_sha
        dd.write_json(self.diag / "validation_manifest.json", validation_rows)
        train_scenes = {row["scene"] for row in self.train_rows}
        validation_scenes = {row["scene"] for row in validation_rows}
        self.contract["data"]["train_validation_scene_overlap"] = sorted(train_scenes & validation_scenes)
        if train_scenes & validation_scenes:
            raise ValueError("Current training/validation scene manifests overlap")
        parameter_check = self.contract["parameters"]
        if parameter_check["trainable_missing_optimizer"] or parameter_check["optimizer_unknown_parameter_count"] or parameter_check["optimizer_duplicate_parameter_count"]:
            raise ValueError("Optimizer parameter coverage failed")
        if any(p.requires_grad for _, p in self.groups.get("dino", [])):
            raise ValueError("Primary paired experiment requires frozen DINO")
        self.probe_cache = [] if args.mode == "initialize" else self.make_probe_cache()
        self.contract["probe_manifest"] = [{k: v for k, v in item.items() if k in ("split", "index", "scene", "frame")}
                                           for item in self.probe_cache]
        if args.canonical_init:
            dd.restore_rng_state(initialization["training_rng_state"])
        self.init_rng = dd.capture_rng_state()
        if args.resume_checkpoint:
            self.resume(args.resume_checkpoint, state=resume_state)
        self.write_contract()
        # Do not keep a second model-sized copy alive in CPU memory.
        self.source = {k: v for k, v in self.source.items() if k not in ("model_state_dict", "optimizer_state_dict")}

    def write_contract(self):
        dd.write_json(self.diag / "contract.json", json_safe(self.contract))

    def make_probe_cache(self):
        import cv2
        from PIL import Image
        rows = []
        for split, index in self.probe_specs:
            dataset = self.trainer.TRAIN_DATASET if split == "train" else self.trainer.TEST_DATASET
            stream = self.stream if split == "train" else DeterministicStream(
                dataset, 1, self.args.seed, self.original.collate_fn, dd.capture_rng_state, dd.restore_rng_state)
            sample = stream.sample(index, epoch=0)
            x0, y0, x1, y1 = map(int, sample["crop_box"])
            instance = np.asarray(Image.open(dataset.base_dataset.labelpath[index]))[y0:y1, x0:x1]
            h, w = sample["gt_depth_m"].shape[-2:]
            instance = torch.from_numpy(cv2.resize(instance.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64))
            gt = torch.as_tensor(sample["gt_depth_m"])
            # CPU sample cache is deterministic and bounded (16 frames by default).
            rows.append({"split": split, "index": index, "scene": dataset.base_dataset.scenename[index],
                         "frame": int(dataset.base_dataset.frameid[index]), "sample": sample,
                         "instances": instance, "gt": gt, "pairs": dd.make_fixed_pairs(
                             gt, instance=instance, seed=stable_seed(self.args.seed, "pairs", split, index))})
        return rows

    @contextlib.contextmanager
    def diagnostic_context(self, train_mode=True, seed=None):
        counters = [(module, {key: value for key, value in vars(module).items()
                              if key in ("_vis_iter", "_forward_count", "_step")}) for module in self.model.modules()]
        with dd.preserve_diagnostic_state(self.model, self.optimizer):
            self.model.train(train_mode)
            for module in self.model.modules():
                if hasattr(module, "is_training"):
                    module.is_training = bool(train_mode)
            if seed is not None:
                self.original.seed_everything(seed)
            try:
                yield
            finally:
                for module, values in counters:
                    for key, value in values.items():
                        setattr(module, key, value)

    def forward(self, batch, capture=False, diagnostics=True):
        inputs = dict(batch)
        self.original.drop_unused_point_inputs(inputs)
        self.original.validate_batch_label_contract(inputs, use_cdf=True)
        inputs = self.original.move_batch_to_device(inputs, self.trainer.device, use_cdf=True, non_blocking=False)
        inputs.update(depth_grad_capture_routes=capture, cva_compute_diagnostics=True,
                      geometry_compute_diagnostics=False, cva_export_angle_feature=False,
                      cva_force_process_grasp_labels=True, depth_dynamics_diagnostics=diagnostics)
        self.original.assert_cpu_resident_label_lists(inputs, use_cdf=True)
        for key in ("img", "K", "gt_depth_m"):
            if not bool(torch.isfinite(inputs[key]).all()):
                raise FloatingPointError(f"Nonfinite model input: {key}")
        endpoints = self.model(inputs)
        self.original.assert_geometry_depth_contract(endpoints, expected_source="pred", context="depth dynamics")
        loss, endpoints = self.original.get_loss_economicgrasp(endpoints, use_cdf=True)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite total objective")
        return loss, endpoints

    def losses(self, endpoints):
        return {key: endpoints[value] for key, value in LOSS_KEYS.items()}

    def scalar_metrics(self, endpoints):
        metrics = self.trainer.extract_scalar_metrics(endpoints)
        with torch.no_grad():
            gt = endpoints["gt_depth_m"].detach()
            valid_gt = torch.isfinite(gt) & (gt >= .2) & (gt <= 1.)
            metrics["D: Dynamics GT valid pixels"] = int(valid_gt.sum())
            metrics["D: Dynamics depth loss denominator"] = gt.numel()
            obj = endpoints["objectness_label_tok"].detach()
            metrics["D: Dynamics objectness denominator"] = int((obj != -1).sum())
            foreground = obj == 1
            if "token_valid_mask" in endpoints:
                foreground &= endpoints["token_valid_mask"].bool()
            metrics["D: Dynamics graspness denominator"] = int(foreground.sum())
            metrics["D: Dynamics view denominator"] = endpoints["view_score"].numel()
            for key, label in (("batch_grasp_cdf_valid_mask", "CDF valid count"),
                               ("batch_grasp_cdf_pos_mask", "CDF positive count"),
                               ("batch_grasp_width_valid_mask_angle_depth", "width valid count")):
                if key in endpoints:
                    metrics["D: Dynamics " + label] = int(endpoints[key].bool().sum())
            idx = endpoints.get("kview_base_token_sel_idx", endpoints["token_sel_idx"]).detach().long()
            raw_seed = endpoints["depth_net_pred"].detach().flatten(1).gather(1, idx)
            metrics["D: Dynamics seed clamp fraction"] = float(((raw_seed <= self.model.min_depth) |
                (raw_seed >= self.model.max_depth) | ~torch.isfinite(raw_seed)).float().mean())
            metrics["D: Dynamics repeated seed fraction"] = sum(1. - len(torch.unique(row)) / row.numel()
                                                               for row in idx) / len(idx)
            if "dbg_mask_pred" in endpoints:
                primary = endpoints["dbg_mask_pred"].gather(1, idx)
                metrics["D: Dynamics seed fallback fraction"] = float((~primary).float().mean())
            query_idx = endpoints["token_sel_idx"].detach().long()
            gt_seed = gt.reshape(gt.shape[0], -1).gather(1, query_idx)
            gt_seed_valid = (gt_seed >= .2) & (gt_seed <= 1.) & torch.isfinite(gt_seed)
            query_z = endpoints["xyz_graspable"][..., 2].detach()
            metrics["D: Dynamics center z error m"] = float((query_z - gt_seed)[gt_seed_valid].abs().mean()) if bool(gt_seed_valid.any()) else None
            traces = endpoints.get("depth_dynamics_label_trace", [])
            if traces:
                trace = traces[-1]
                distances = trace["point_distance_m"].detach()
                metrics["D: Dynamics NN distance mean m"] = float(distances.mean())
                metrics["D: Dynamics NN distance max m"] = float(distances.max())
                metrics["D: Dynamics point valid fraction"] = float(trace["point_valid"].float().mean())
                metrics["D: Dynamics view valid fraction"] = float(trace["view_valid"].float().mean())
        return {key: json_safe(value) for key, value in metrics.items()}

    def probe_identity_metrics(self, item, mode, endpoints):
        traces = endpoints.get("depth_dynamics_label_trace", [])
        if not traces:
            return {"status": "label_trace_not_exported"}
        latest = traces[-1]
        current = {key: latest[key].detach().cpu().clone() for key in (
            "nn_indices", "point_valid", "view_valid", "token_indices", "view_indices")}
        current["xyz"] = endpoints["xyz_graspable"].detach().cpu().clone()
        cache_key = f"{item['split']}/{item['index']}/{int(mode)}"
        prior = self.probe_previous.get(cache_key)
        self.probe_previous[cache_key] = current
        if prior is None:
            return {"status": "initial_reference"}
        same = (current["token_indices"] == prior["token_indices"]) & (current["view_indices"] == prior["view_indices"])
        changed_nn = current["nn_indices"] != prior["nn_indices"]
        return {"status": "compared_to_previous_probe_same_frame_mode",
                "query_identity_switch_rate_aligned_slots": float((~same).float().mean()),
                "nn_switch_rate_aligned_slots": float(changed_nn.float().mean()),
                "nn_switch_rate_same_identity": float(changed_nn[same].float().mean()) if bool(same.any()) else None,
                "same_identity_queries": int(same.sum()),
                "point_valid_switch_rate_aligned_slots": float((current["point_valid"] != prior["point_valid"]).float().mean()),
                "view_valid_switch_rate_aligned_slots": float((current["view_valid"] != prior["view_valid"]).float().mean()),
                "same_identity_query_displacement_m": float((current["xyz"] - prior["xyz"]).norm(dim=-1)[same].mean()) if bool(same.any()) else None}

    def equality_tensors(self, endpoints):
        # Include logits, seed/view identities, masks, matched targets and every loss.
        explicit = {"depth_net_pred", "depth_head_raw_pred", "objectness_score", "graspness_score",
                    "view_score", "xyz_graspable", "token_sel_idx", "grasp_top_view_inds",
                    "grasp_top_view_xyz", "grasp_cdf_pred_angle_depth", "grasp_width_pred_angle_depth"}
        return {key: value.detach().cpu().clone() for key, value in endpoints.items()
                if torch.is_tensor(value) and (key in explicit or key.startswith("batch_grasp_")
                   or "valid_mask" in key or key.startswith("dbg_mask") or key.startswith("B:"))}

    def compare_endpoints(self, reference, values):
        failures = []
        mask_keys = {"batch_grasp_cdf_bins_angle_depth": "batch_grasp_cdf_valid_mask",
                     "batch_grasp_width_angle_depth": "batch_grasp_width_valid_mask_angle_depth"}
        for key in reference.keys() | values.keys():
            if key not in reference or key not in values:
                failures.append({"key": key, "reason": "missing endpoint"})
                continue
            a, b = reference[key], values[key]
            if a.shape != b.shape:
                failures.append({"key": key, "reason": "shape mismatch", "shapes": [list(a.shape), list(b.shape)]})
                continue
            mismatch = ~torch.isclose(a, b, atol=self.args.forward_atol, rtol=self.args.forward_rtol,
                                     equal_nan=False) if a.is_floating_point() else a.ne(b)
            if not bool(mismatch.any()):
                continue
            coordinates = torch.nonzero(mismatch, as_tuple=False)[:12]
            flat_indices = torch.nonzero(mismatch.reshape(-1), as_tuple=False).reshape(-1)[:12]
            row = {"key": key, "reason": "value mismatch", "mismatch_count": int(mismatch.sum()),
                   "max_abs": float((a.float() - b.float()).abs().max()),
                   "first_coordinates": coordinates.tolist(),
                   "first_reference_values": a.reshape(-1)[flat_indices].tolist(),
                   "first_route_values": b.reshape(-1)[flat_indices].tolist()}
            mask_key = mask_keys.get(key)
            if mask_key is not None and mask_key in reference and mask_key in values:
                union_valid = reference[mask_key].bool() | values[mask_key].bool()
                row.update(effective_mask=mask_key, valid_mismatch_count=int((mismatch & union_valid).sum()),
                           invalid_mismatch_count=int((mismatch & ~union_valid).sum()),
                           first_valid_coordinates=torch.nonzero(mismatch & union_valid, as_tuple=False)[:12].tolist())
            failures.append(row)
        return failures

    def save_gradient_maps(self, endpoints, route, batch_id):
        if not self.args.save_gradient_maps or batch_id != 0 or self.step != 0:
            return
        terms = self.losses(endpoints)
        objectives = {"depth": terms["depth"] * self.weights["depth"],
                      "task": sum(value * self.weights[key] for key, value in terms.items() if key != "depth")}
        result = {"step": self.step, "route": route, "batch_id": batch_id,
                  "rgb": endpoints["img"][0].detach().cpu(),
                  "gt_m": endpoints["gt_depth_m"][0].detach().cpu(),
                  "pred_m": endpoints["depth_net_pred"][0].detach().cpu(),
                  "raw": endpoints["depth_head_raw_pred"][0].detach().cpu(),
                  "depth_color_range_m": [0.0, 1.0], "gradients": {}}
        for term, loss in objectives.items():
            for key in ("depth_net_pred", "depth_head_raw_pred"):
                target = endpoints[key]
                grad = torch.autograd.grad(loss, target, retain_graph=True, allow_unused=True,
                                           materialize_grads=False)[0] if loss.requires_grad and target.requires_grad else None
                result["gradients"][f"{term}_wrt_{key}"] = None if grad is None else grad[0].detach().cpu()
        torch.save(result, self.diag / f"gradient_maps_{route.replace(',', '-')}_initial.pt")

    def audit(self, count=None, routes=None):
        count = self.args.audit_batches if count is None else count
        routes = self.args.audit_routes.split(",") if routes is None else routes
        prior_routes = self.model.get_depth_grad_routes()
        all_pass = True
        connectivity_pass = True
        route_nonzero = {name: False for name in ("all", "gse", "seed_xyz", "support")}
        audit_indices = np.linspace(0, len(self.stream.dataset) - 1, count * self.cfg.batch_size, dtype=int)
        for batch_id in range(count):
            split = "train" if batch_id % 2 == 0 else "validation_test_seen"
            stream = self.stream if split == "train" else DeterministicStream(
                self.trainer.TEST_DATASET, self.cfg.batch_size, self.args.seed, self.original.collate_fn,
                dd.capture_rng_state, dd.restore_rng_state)
            selected = np.linspace(0, len(stream.dataset) - 1, count * self.cfg.batch_size, dtype=int)
            indices = selected[batch_id * self.cfg.batch_size:(batch_id + 1) * self.cfg.batch_size].tolist()
            batch = self.original.collate_fn([stream.sample(index, epoch=0) for index in indices])
            # First forward establishes a reference without any backward. The
            # first route below repeats this exact route/seed before switching
            # detach flags, so CUDA label nondeterminism cannot masquerade as a
            # route-dependent change. Mismatches still fail the strict gate.
            self.model.set_depth_grad_routes(routes[0])
            with self.diagnostic_context(train_mode=True, seed=stable_seed(self.args.seed, "audit", batch_id)):
                reference_loss, reference_endpoints = self.forward(batch, capture=True)
                reference = self.equality_tensors(reference_endpoints)
                del reference_loss, reference_endpoints
            with self.diagnostic_context(train_mode=True, seed=stable_seed(self.args.seed, "audit", batch_id)):
                plain_loss, plain_endpoints = self.forward(batch, capture=False, diagnostics=False)
                telemetry_failures = self.compare_endpoints(reference, self.equality_tensors(plain_endpoints))
                all_pass &= not telemetry_failures
                dd.append_jsonl(self.diag / "telemetry_equality.jsonl", {
                    "step": self.step, "batch": batch_id, "route": routes[0],
                    "passed": not telemetry_failures, "failures": telemetry_failures})
                del plain_loss, plain_endpoints
            for route in routes:
                self.model.set_depth_grad_routes(route)
                with self.diagnostic_context(train_mode=True, seed=stable_seed(self.args.seed, "audit", batch_id)):
                    loss, endpoints = self.forward(batch, capture=True)
                    values = self.equality_tensors(endpoints)
                    failures = self.compare_endpoints(reference, values)
                    all_pass &= not failures
                    dd.append_jsonl(self.diag / "forward_equality.jsonl", {"step": self.step, "batch": batch_id,
                        "indices": indices, "route": route, "reference_route": routes[0],
                        "comparison": "same_route_repeat" if route == routes[0] else "route_switch",
                        "compared": sorted(values), "passed": not failures,
                        "failures": failures, "atol": self.args.forward_atol, "rtol": self.args.forward_rtol})
                    observations = {"depth_net_pred": endpoints["depth_net_pred"],
                                    "depth_head_raw_pred": endpoints["depth_head_raw_pred"]}
                    for key in ("depth_grad_gse_input", "depth_grad_seed_xyz_input", "depth_grad_support_input"):
                        value = endpoints.get(key)
                        if isinstance(value, (tuple, list)):
                            observations.update({f"{key}.{i}": tensor for i, tensor in enumerate(value)})
                        elif torch.is_tensor(value):
                            observations[key] = value
                    rows = dd.audit_gradients(self.losses(endpoints), self.groups,
                                              weights=self.weights, observations=observations)
                    for row in rows:
                        if row["state"] == "nonfinite":
                            connectivity_pass = False
                        row.update(step=self.step, arm=self.args.arm, batch=batch_id, route=route,
                                   indices=indices, split=split)
                        dd.append_jsonl(self.diag / "gradient_audit.jsonl", json_safe(row))
                        dd.append_csv(self.diag / "route_connectivity.csv", json_safe(row))
                    # Direct output connectivity is independent of parameter group naming.
                    for term, term_loss in self.losses(endpoints).items():
                        target = endpoints["depth_net_pred"]
                        grad = torch.autograd.grad(term_loss, target, retain_graph=True, allow_unused=True,
                                                   materialize_grads=False)[0] if term_loss.requires_grad and target.requires_grad else None
                        status = "unused" if grad is None else ("nonfinite" if not bool(torch.isfinite(grad).all())
                            else ("connected_nonzero" if bool(grad.ne(0).any()) else "connected_zero"))
                        if status == "nonfinite":
                            connectivity_pass = False
                        dd.append_jsonl(self.diag / "output_connectivity.jsonl", {"step": self.step, "batch": batch_id,
                            "route": route, "loss": term, "status": status,
                            "norm": float(grad.norm()) if grad is not None else None})
                        if route == "none" and term != "depth" and grad is not None:
                            connectivity_pass = False
                        if term == "depth" and grad is None:
                            connectivity_pass = False
                        if route in route_nonzero and term != "depth" and status == "connected_nonzero":
                            route_nonzero[route] = True
                    self.directional_probe(endpoints, route, batch_id)
                    self.save_gradient_maps(endpoints, route, batch_id)
                    # Route aliases and the final scalar loss also retain the
                    # graph. Release them before constructing the next route's
                    # forward, otherwise a one-batch audit can peak at 2 graphs.
                    del endpoints, loss, rows, observations, term_loss, target, grad, value
            print(f"[audit] batch={batch_id + 1}/{count} forward_equal={all_pass} baseline_contract={connectivity_pass}", flush=True)
            if batch_id < self.args.directional_batches and len(routes) > 1:
                from depth_dynamics_directional import run_directional_probe
                prepared = dict(batch)
                self.original.drop_unused_point_inputs(prepared)
                prepared = self.original.move_batch_to_device(prepared, self.trainer.device, use_cdf=True)
                prepared.update(cva_force_process_grasp_labels=True, cva_compute_diagnostics=True,
                                geometry_compute_diagnostics=False, depth_grad_capture_routes=True,
                                depth_dynamics_diagnostics=True)
                directional = run_directional_probe(self.model, prepared,
                    lambda ep: self.original.get_loss_economicgrasp(ep, use_cdf=True),
                    loss_weights=self.weights, batch_id=batch_id,
                    output_csv=self.diag / "local_directional_probe.csv")
                self.contract["directional_probe"] = {"rows": len(directional), "batch": batch_id,
                                                       "status": "executed; see per-row flags and convergence"}
        enabled = [key for key, value in prior_routes.items() if value]
        self.model.set_depth_grad_routes(",".join(enabled) if enabled else "none")
        full_gate = {"none", "all", "gse", "seed_xyz", "support"}.issubset(routes)
        audit_record = {"step": self.step, "batches": count, "routes": routes,
                        "forward_equality": all_pass, "baseline_output_connectivity": connectivity_pass,
                        "opened_route_nonzero": route_nonzero if full_gate else "partial replay; see CSV"}
        if full_gate:
            self.contract["audit"] = audit_record
            self.contract["p0_passed"] = bool(all_pass and connectivity_pass and all(route_nonzero.values()))
            config = {key: value for key, value in vars(self.cfg).items() if key not in ("log_dir",)}
            gate = {"passed": False, "route_checks_passed": self.contract["p0_passed"],
                    "replay_policy": self.args.replay_policy,
                    "diagnostic_noninterference_passed": None, "init_sha256": self.init_sha,
                    "resolved_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    "git_head": self.contract["git"]["head"],
                    "tracked_diff_sha256": self.contract["git"]["tracked_diff_sha256"],
                    "forward_equality": all_pass, "baseline_disconnected_depth_connected": connectivity_pass,
                    "route_connected_nonzero": route_nonzero,
                    "P1": "raw+weighted audits on fixed train/user-designated validation batches; directional CSV records scope",
                    "P1_complete_mechanism": False}
            dd.write_json(self.diag / "p0_gate.json", gate)
        else:
            self.contract.setdefault("replay_audits", []).append(audit_record)
        self.write_contract()
        if not all_pass or not connectivity_pass or (full_gate and not all(route_nonzero.values())):
            raise RuntimeError("P0 failed; inspect forward_equality.jsonl and output_connectivity.jsonl")

    def directional_probe(self, endpoints, route, batch_id):
        pred = endpoints["depth_net_pred"]
        gt = endpoints["gt_depth_m"]
        if gt.ndim == 3:
            gt = gt[:, None]
        valid = torch.isfinite(gt) & (gt >= 0.2) & (gt <= 1.0)
        terms = self.losses(endpoints)
        terms["grasp_total"] = sum(value * self.weights[key] for key, value in terms.items() if key != "depth")
        for key, value in terms.items():
            grad = torch.autograd.grad(value, pred, retain_graph=True, allow_unused=True,
                                       materialize_grads=False)[0] if value.requires_grad and pred.requires_grad else None
            for image in range(pred.shape[0]):
                mask = valid[image]
                center = pred[image][mask].detach().mean() if bool(mask.any()) else pred.new_tensor(0.)
                dd.append_csv(self.diag / "native_autograd_directional.csv", {"step": self.step, "batch": batch_id,
                    "image": image, "route": route, "loss": key, "region": "GT_valid",
                    "dL_dmu": float(grad[image][mask].sum()) if grad is not None else None,
                    "dL_dalpha": float((grad[image][mask] * (pred[image][mask].detach() - center)).sum()) if grad is not None else None,
                    "status": "native_autograd_only; discrete assignments fixed by graph; finite_difference_pending",
                    "actual_network_update_claim": False})

    def probe(self):
        train_eval_metrics = []
        for mode in (True, False):
            with self.diagnostic_context(train_mode=mode, seed=stable_seed(self.args.seed, "probe", int(mode))):
                with torch.no_grad():
                    for item in self.probe_cache:
                        batch = self.original.collate_fn([item["sample"]])
                        loss, endpoints = self.forward(batch)
                        metrics = dd.depth_metrics(endpoints["depth_net_pred"].detach().cpu(), item["gt"],
                            foreground=item["instances"] > 0, raw=endpoints["depth_head_raw_pred"].detach().cpu(), pairs=item["pairs"])
                        row = {"step": self.step, "arm": self.args.arm, "split": item["split"],
                               "scene": item["scene"], "frame": item["frame"], "index": item["index"],
                               "module_mode": "train" if mode else "eval", "is_training": mode,
                               "selection_policy": "native_stochastic_fixed_rng" if mode else "native_argmax",
                               "task_targets": "native matching, not fixed physical queries", "metrics": metrics,
                               "identity_changes": self.probe_identity_metrics(item, mode, endpoints),
                               "native_loss": float(loss), "coverage": self.scalar_metrics(endpoints)}
                        dd.append_jsonl(self.diag / "fixed_probe.jsonl", json_safe(row))
                        if not mode and item["split"] == "train":
                            train_eval_metrics.extend(metrics)
                        if not mode and item is self.probe_cache[0]:
                            torch.save({"step": self.step, "rgb": batch["img"].cpu(), "gt": item["gt"],
                                        "pred": endpoints["depth_net_pred"].detach().cpu(),
                                        "raw": endpoints["depth_head_raw_pred"].detach().cpu(),
                                        "metric_color_range_m": [0.0, 1.0]}, self.diag / f"fixed_frame_{self.step:06d}.pt")
        self.record_event(train_eval_metrics)
        self.contract["healthy_probe_completed"] = True
        gate_path = self.diag / "p0_gate.json"
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())
            gate["healthy_probe_completed"] = True
            gate["passed"] = bool(gate["route_checks_passed"] and gate.get("diagnostic_noninterference_passed"))
            dd.write_json(gate_path, gate)
        self.write_contract()

    def record_event(self, metrics):
        data = [entry["regions"]["foreground"]["std_ratio"] for entry in metrics
                if entry["regions"]["foreground"]["count"] >= 2
                and entry["regions"]["foreground"]["gt_std"] is not None
                and entry["regions"]["foreground"]["gt_std"] > 0.005
                and entry["regions"]["foreground"]["std_ratio"] is not None]
        fraction = sum(value < .1 for value in data) / len(data) if data else None
        previous_anomaly = self.events.get("first_anomaly")
        self.events = advance_flat_event_state(self.events, step=self.step, fraction_flat=fraction)
        if previous_anomaly is None and self.events["first_anomaly"] is not None:
            self.locked_steps.update(row["step"] for row in self.manifest[-6:])
        dd.append_jsonl(self.diag / "events.jsonl", {"step": self.step, "eligible_frames": len(data),
            "fraction_flat": fraction, "per_image_std_ratios": data, **self.events})

    def checkpoint(self, tag="rolling"):
        destination = self.checkpoint_dir / f"step_{self.step:06d}.pt"
        if destination.exists():
            return destination
        if tag in ("initial", "first_threshold", "confirmed_event", "final"):
            self.locked_steps.add(self.step)
        free_bytes = __import__("shutil").disk_usage(self.checkpoint_dir).free
        estimated = sum(p.numel() * p.element_size() for p in self.model.state_dict().values()) * 3
        if free_bytes < estimated * 1.15:
            raise OSError(f"Insufficient disk for a full snapshot ({free_bytes} bytes free)")
        state = {"format_version": FORMAT_VERSION, "model_state_dict": self.model.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state": {"policy": self.args.lr_schedule,
                     "source_epoch": self.initial_epoch, "stream_epoch": self.stream.epoch}, "scaler_state": None,
                 "rng": dd.capture_rng_state(), "loader": self.stream.state_dict(), "step": self.step,
                 "seen_images": self.seen_images, "events": self.events, "locked_steps": sorted(self.locked_steps),
                 "probe_previous": self.probe_previous,
                 "init_sha256": self.init_sha, "routes": self.model.get_depth_grad_routes(),
                 "cfg": vars(self.cfg), "arguments": vars(self.args),
                 "train_manifest_sha256": self.contract["data"]["train_manifest_sha256"],
                 "module_training": {name: module.training for name, module in self.model.named_modules()},
                 "is_training": {name: module.is_training for name, module in self.model.named_modules() if hasattr(module, "is_training")}}
        state["runtime_counters"] = {name: {key: getattr(module, key) for key in ("_vis_iter", "_debug_iter")
                                          if hasattr(module, key)} for name, module in self.model.named_modules()}
        temporary = destination.with_suffix(".partial")
        torch.save(state, temporary)
        temporary.replace(destination)
        row = {"step": self.step, "tag": tag, "path": str(destination), "sha256": dd.sha256_file(destination),
               "bytes": destination.stat().st_size, "seen_images": self.seen_images, "loader": self.stream.state_dict()}
        self.manifest.append(row)
        rolling = [entry for entry in self.manifest if entry["step"] not in self.locked_steps and Path(entry["path"]).exists()]
        for old in rolling[:-self.args.keep_last]:
            Path(old["path"]).unlink()
            old["deleted_after_retention"] = True
        dd.write_json(self.diag / "checkpoints_manifest.json", self.manifest)
        dd.write_json(self.output / "latest_checkpoint.json", row)
        print(f"[checkpoint] step={self.step} path={destination}", flush=True)
        return destination

    def resume(self, path, state=None):
        state = torch.load(path, map_location="cpu", weights_only=False) if state is None else state
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError("Resume requires a full dynamics checkpoint")
        assert_rescue_resume_explicit(state, self.args)
        self.resume_output_check = assert_resume_outputs_not_newer(self.output, self.diag, state["step"])
        for key, current in (("init_sha256", self.init_sha), ("routes", self.model.get_depth_grad_routes()),
                             ("train_manifest_sha256", self.contract["data"]["train_manifest_sha256"])):
            if state[key] != current:
                raise ValueError(f"Resume mismatch: {key}")
        ignored = {"log_dir", "num_workers", "eval_num_workers"}
        differences = {key: [value, vars(self.cfg).get(key)] for key, value in state["cfg"].items()
                       if key not in ignored and value != vars(self.cfg).get(key)}
        if differences:
            raise ValueError(f"Resume resolved-config mismatch: {differences}")
        for key in ("seed", "arm", "clip_norm", "lr_schedule", "train_probe_frames", "heldout_test_probe_frames"):
            if state["arguments"][key] != vars(self.args)[key]:
                raise ValueError(f"Resume diagnostic setting mismatch: {key}")
        depth_init.assert_resume_settings(state, self.args)
        if state["arguments"].get("replay_policy", "strict") != self.args.replay_policy:
            raise ValueError("Resume diagnostic setting mismatch: replay_policy")
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.stream.load_state_dict(state["loader"])
        self.step, self.seen_images = state["step"], state["seen_images"]
        self.events, self.locked_steps = state["events"], set(state["locked_steps"])
        self.probe_previous = state.get("probe_previous", {})
        for name, module in self.model.named_modules():
            module.training = state["module_training"][name]
            if name in state["is_training"]:
                module.is_training = state["is_training"][name]
            for key, value in state.get("runtime_counters", {}).get(name, {}).items():
                setattr(module, key, value)
        dd.restore_rng_state(state["rng"])
        manifest_path = self.diag / "checkpoints_manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
        self.contract["resume"] = {"path": str(Path(path).resolve()), "sha256": dd.sha256_file(path), "step": self.step,
                                   "output_history_check": self.resume_output_check}

    def verify_diagnostic_step(self):
        """Two no-audit controls plus audit replay, retaining strict results.

        The extra control distinguishes backend variability from an audit effect.
        The optional calibrated policy has a fixed absolute parameter cap and
        exact-state requirements; its result never replaces the strict report.
        """
        path = self.checkpoint("initial" if self.step == 0 else "replay_reference")
        initial = torch.load(path, map_location="cpu", weights_only=False)

        def digest(value):
            result = hashlib.sha256()

            def visit(item):
                if torch.is_tensor(item):
                    tensor = item.detach().cpu().contiguous()
                    result.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}".encode())
                    result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
                elif isinstance(item, np.ndarray):
                    result.update(f"numpy:{item.dtype}:{item.shape}".encode())
                    result.update(item.tobytes())
                elif isinstance(item, dict):
                    for key in sorted(item, key=str):
                        result.update(repr(key).encode())
                        visit(item[key])
                elif isinstance(item, (tuple, list)):
                    result.update(type(item).__name__.encode())
                    for entry in item:
                        visit(entry)
                else:
                    result.update(repr(item).encode())
            visit(value)
            return result.hexdigest()

        def tensor_difference(left, right, *, gradients=False):
            if left is None or right is None:
                return {"reference_unused": left is None, "comparison_unused": right is None,
                        "exact_equal": left is None and right is None}
            left, right = left.detach().cpu(), right.detach().cpu()
            if left.shape != right.shape or left.dtype != right.dtype:
                return {"exact_equal": False, "reference_shape": list(left.shape),
                        "comparison_shape": list(right.shape), "reference_dtype": str(left.dtype),
                        "comparison_dtype": str(right.dtype)}
            a, b = left.double().reshape(-1), right.double().reshape(-1)
            finite = torch.isfinite(a) & torch.isfinite(b)
            difference = b - a
            changed = a.ne(b)
            above_tolerance = ~torch.isclose(a, b, atol=self.args.forward_atol, rtol=self.args.forward_rtol)
            valid_delta = difference[finite]
            row = {"shape": list(left.shape), "numel": left.numel(), "dtype": str(left.dtype),
                   "exact_equal": bool(torch.equal(left, right)), "changed_count": int(changed.sum()),
                   "exceeds_tolerance_count": int(above_tolerance.sum()), "nonfinite_count": int((~finite).sum()),
                   "max_abs_difference": float(valid_delta.abs().max()) if valid_delta.numel() else None,
                   "rms_difference": float(valid_delta.square().mean().sqrt()) if valid_delta.numel() else None,
                   "difference_l2": float(valid_delta.norm()) if valid_delta.numel() else None,
                   "reference_l2": float(a[finite].norm()), "comparison_l2": float(b[finite].norm()),
                   "reference_max_abs": float(a[finite].abs().max()) if bool(finite.any()) else None,
                   "comparison_max_abs": float(b[finite].abs().max()) if bool(finite.any()) else None,
                   "sign_flip_count": int(((a * b < 0) & finite).sum()),
                   "zero_vs_nonzero_count": int(((a == 0) != (b == 0)).sum())}
            if a.numel():
                ranked = torch.nan_to_num(difference.abs(), nan=float("inf")).topk(min(8, a.numel())).indices
                row["largest_difference_elements"] = [{"flat_index": int(index), "reference": float(a[index]),
                    "comparison": float(b[index]), "difference": float(difference[index])} for index in ranked]
            if gradients:
                magnitude = torch.maximum(a.abs(), b.abs())
                row["near_zero"] = {str(threshold): {
                    "both_abs_at_most_count": int((finite & (magnitude <= threshold)).sum()),
                    "changed_within_count": int((finite & changed & (magnitude <= threshold)).sum()),
                    "sign_flips_within_count": int((finite & (a * b < 0) & (magnitude <= threshold)).sum())}
                    for threshold in (1e-12, 1e-10, 1e-8, 1e-6)}
            return row

        def restore_initial():
            self.model.load_state_dict(initial["model_state_dict"])
            self.optimizer.load_state_dict(copy.deepcopy(initial["optimizer_state_dict"]))
            self.stream.load_state_dict(initial["loader"])
            self.optimizer.zero_grad(set_to_none=True)
            for name, module in self.model.named_modules():
                module.training = initial["module_training"][name]
                if name in initial["is_training"]:
                    module.is_training = initial["is_training"][name]
                for key, value in initial.get("runtime_counters", {}).get(name, {}).items():
                    setattr(module, key, value)
            dd.restore_rng_state(initial["rng"])

        def gradient_copy():
            return {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                    for name, parameter in self.model.named_parameters() if parameter.requires_grad}

        outcomes = []
        try:
            branches = [("no_audit_A", False), ("no_audit_B", False), ("with_audit", True)]
            if self.args.probe_schedule == "cold" and self.args.routes == "all":
                branches.append(("view_noop", False))
            for branch, with_diagnostics in branches:
                restore_initial()
                self.model.train()
                for module in self.model.modules():
                    if hasattr(module, "is_training"):
                        module.is_training = True
                audit_state_check = None
                if with_diagnostics:
                    audit_before = dd.snapshot_state(self.model, self.optimizer, loader=self.stream.state_dict(),
                                                     routes=self.model.get_depth_grad_routes(), existing_grads=gradient_copy())
                    audit_before_sha = {key: digest(value) for key, value in audit_before.items()}
                    del audit_before
                    self.audit(count=1, routes=[self.args.routes])
                    audit_after = dd.snapshot_state(self.model, self.optimizer, loader=self.stream.state_dict(),
                                                    routes=self.model.get_depth_grad_routes(), existing_grads=gradient_copy())
                    audit_after_sha = {key: digest(value) for key, value in audit_after.items()}
                    del audit_after
                    audit_state_check = {"before_sha256": audit_before_sha, "after_sha256": audit_after_sha,
                                         "mismatch_keys": [key for key in audit_before_sha
                                             if audit_before_sha[key] != audit_after_sha[key]]}
                    audit_state_check["exact_equal"] = not audit_state_check["mismatch_keys"]
                batch, indices = self.stream.next()
                self.optimizer.zero_grad(set_to_none=True)
                before = dd.snapshot_state(self.model, self.optimizer, loader=self.stream.state_dict(),
                                           routes=self.model.get_depth_grad_routes())
                state_sha = {key: digest(value) for key, value in before.items()}
                del before
                batch_sha = digest(batch)
                loss, endpoints = self.forward(batch)
                forward_values = self.equality_tensors(endpoints)
                forward_sha = {key: digest(value) for key, value in forward_values.items()}
                forward_losses = {key: float(value.detach()) for key, value in self.losses(endpoints).items()}
                forward_losses["total"] = float(loss.detach())
                del forward_values
                depth_init.backward_with_policy(loss, self.losses(endpoints)["view"] * self.weights["view"],
                    self.depth_gradient_scope, "view_noop" if branch == "view_noop" else "normal", self.model.parameters())
                preclip = gradient_copy()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm,
                                                                 error_if_nonfinite=True))
                postclip = gradient_copy()
                self.optimizer.step()
                state = dd.snapshot_state(self.model, self.optimizer, loader=self.stream.state_dict())
                outcomes.append({"branch": branch, "state": state, "preclip": preclip, "postclip": postclip,
                                 "pre_forward_state_sha256": state_sha, "batch_sha256": batch_sha,
                                 "audit_state_check": audit_state_check,
                                 "batch_indices": indices, "forward_tensor_sha256": forward_sha,
                                 "forward_losses": forward_losses, "gradient_norm": grad_norm,
                                 "clip_coefficient": min(1., self.args.clip_norm / (grad_norm + 1e-6))})
                del loss, endpoints, batch
                print(f"[noninterference] captured {branch}", flush=True)
        finally:
            restore_initial()

        def compare_branches(reference, other):
            left, right = reference["state"], other["state"]
            errors = []
            for key in ("model_state_dict", "optimizer_state_dict"):
                errors += compare_state(left[key], right[key], atol=self.args.forward_atol,
                                        rtol=self.args.forward_rtol, path=key)
            for key in ("rng_state", "module_modes", "module_flags", "metadata"):
                errors += compare_state(left[key], right[key], path=key)
            before_mismatch = [key for key in reference["pre_forward_state_sha256"]
                               if reference["pre_forward_state_sha256"][key] != other["pre_forward_state_sha256"][key]]
            if before_mismatch:
                errors.append("pre_forward_state_not_exact: " + ",".join(before_mismatch))
            if reference["batch_sha256"] != other["batch_sha256"]:
                errors.append("input_batch_not_exact")
            for branch in (reference, other):
                if branch["audit_state_check"] is not None and not branch["audit_state_check"]["exact_equal"]:
                    errors.append("audit_changed_exact_state: " + ",".join(branch["audit_state_check"]["mismatch_keys"]))
            model_differences, gradient_differences, context = {}, {}, {}
            for name, a in left["model_state_dict"].items():
                b = right["model_state_dict"][name]
                if not torch.equal(a, b):
                    model_differences[name] = tensor_difference(a, b)
                    if name in reference["preclip"]:
                        context[name] = {"preclip_gradient": tensor_difference(reference["preclip"][name], other["preclip"][name], gradients=True),
                                         "postclip_gradient": tensor_difference(reference["postclip"][name], other["postclip"][name], gradients=True),
                                         "actual_update": tensor_difference(a - initial["model_state_dict"][name],
                                                                              b - initial["model_state_dict"][name])}
            for name, a in reference["preclip"].items():
                b = other["preclip"][name]
                if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                    gradient_differences[name] = tensor_difference(a, b, gradients=True)
            return {"reference": reference["branch"], "comparison": other["branch"], "passed": not errors,
                    "errors": errors, "pre_forward_state_mismatch_keys": before_mismatch,
                    "batch_exact_equal": reference["batch_sha256"] == other["batch_sha256"],
                    "forward_loss_exact_equal": reference["forward_losses"] == other["forward_losses"],
                    "forward_tensor_exact_mismatch_keys": [key for key in reference["forward_tensor_sha256"]
                        if reference["forward_tensor_sha256"][key] != other["forward_tensor_sha256"].get(key)],
                    "model_tensor_differences": model_differences, "preclip_gradient_differences": gradient_differences,
                    "changed_model_tensor_context": context}

        comparison_pairs = {"no_audit_repeat": (0, 1), "audit_vs_reference": (0, 2), "audit_vs_repeat": (1, 2)}
        if len(outcomes) == 4:
            comparison_pairs.update(view_noop_vs_reference=(0, 3), view_noop_vs_repeat=(1, 3))
        comparisons = {name: compare_branches(outcomes[a], outcomes[b]) for name, (a, b) in comparison_pairs.items()}
        errors = [f"{name}: {error}" for name, comparison in comparisons.items() for error in comparison["errors"]]

        # These constants were locked before this run, following the independent
        # strict-repeat measurement. The repeat envelope may further restrict
        # acceptance, but can never increase the fixed absolute parameter cap.
        limits = {"parameter_abs_cap": 1e-5, "plain_repeat_multiplier": 2.0,
                  "numerical_floor": 1e-6, "parameter_rtol": 0.0,
                  "gradient_atol": self.args.forward_atol, "gradient_rtol": self.args.forward_rtol,
                  "optimizer_atol": self.args.forward_atol, "optimizer_rtol": self.args.forward_rtol,
                  "locked_before_run": True}
        parameter_names = {name for name, _ in self.model.named_parameters()}
        buffer_names = {name for name, _ in self.model.named_buffers()}
        global_weights = {}
        for name, comparison in comparisons.items():
            rows = [row for key, row in comparison["model_tensor_differences"].items() if key in parameter_names]
            global_weights[name] = {
                "max_abs": max((row["max_abs_difference"] or 0. for row in rows), default=0.),
                "l2": math.sqrt(sum((row["difference_l2"] or 0.) ** 2 for row in rows)),
                "changed_count": sum(row.get("changed_count", 0) for row in rows),
                "nonfinite_count": sum(row.get("nonfinite_count", 0) for row in rows),
                "strict_tolerance_exceed_count": sum(row.get("exceeds_tolerance_count", 0) for row in rows)}
        baseline = global_weights["no_audit_repeat"]
        max_envelope = limits["plain_repeat_multiplier"] * baseline["max_abs"] + limits["numerical_floor"]
        l2_envelope = limits["plain_repeat_multiplier"] * baseline["l2"] + limits["numerical_floor"]
        calibrated_errors = []
        calibrated_checks = {}
        for name, (reference_index, other_index) in comparison_pairs.items():
            reference, other = outcomes[reference_index], outcomes[other_index]
            comparison = comparisons[name]
            guard_errors = []
            for field in ("rng_state", "module_modes", "module_flags", "metadata"):
                guard_errors += compare_state(reference["state"][field], other["state"][field], path=field)
            guard_errors += compare_state(reference["state"]["optimizer_state_dict"], other["state"]["optimizer_state_dict"],
                                          atol=self.args.forward_atol, rtol=self.args.forward_rtol, path="optimizer_original_tolerance")
            for field in ("preclip", "postclip"):
                guard_errors += compare_state(reference[field], other[field], atol=self.args.forward_atol,
                                              rtol=self.args.forward_rtol, path=field + "_original_tolerance")
            guard_errors += compare_state(
                {key: value for key, value in reference["state"]["model_state_dict"].items() if key in buffer_names},
                {key: value for key, value in other["state"]["model_state_dict"].items() if key in buffer_names},
                path="buffers_exact")
            if comparison["pre_forward_state_mismatch_keys"]:
                guard_errors.append("pre_forward_state_not_exact")
            if not comparison["batch_exact_equal"]:
                guard_errors.append("input_batch_not_exact")
            if not comparison["forward_loss_exact_equal"] or comparison["forward_tensor_exact_mismatch_keys"]:
                guard_errors.append("forward_outputs_not_exact")
            for branch in (reference, other):
                if branch["audit_state_check"] is not None and not branch["audit_state_check"]["exact_equal"]:
                    guard_errors.append("audit_changed_exact_state_including_existing_grads")
            weights = global_weights[name]
            if weights["nonfinite_count"]:
                guard_errors.append("nonfinite_parameters")
            if weights["max_abs"] > limits["parameter_abs_cap"]:
                guard_errors.append("parameter_difference_exceeds_fixed_absolute_cap")
            if name != "no_audit_repeat":
                if weights["max_abs"] > max_envelope:
                    guard_errors.append("parameter_max_exceeds_plain_repeat_envelope")
                if weights["l2"] > l2_envelope:
                    guard_errors.append("parameter_l2_exceeds_plain_repeat_envelope")
            calibrated_checks[name] = {"passed": not guard_errors, "errors": guard_errors,
                                       "global_parameter_difference": weights}
            calibrated_errors += [f"{name}: {error}" for error in guard_errors]
        calibrated = {"passed": not calibrated_errors, "errors": calibrated_errors, "limits": limits,
                      "plain_repeat_measured": baseline, "audit_max_envelope": max_envelope,
                      "audit_effective_max_cap": min(limits["parameter_abs_cap"], max_envelope),
                      "audit_l2_envelope": l2_envelope, "comparisons": calibrated_checks,
                      "claim": "exact audited state and forward; CUDA parameter updates within a locked absolute cap, not bitwise replay"}
        selected_errors = errors if self.args.replay_policy == "strict" else calibrated_errors
        selected_passed = not selected_errors
        summary = {"passed": selected_passed, "errors": selected_errors, "policy": self.args.replay_policy,
                   "strict_passed": not errors, "strict_errors": errors, "calibrated": calibrated,
                   "atol": self.args.forward_atol,
                   "rtol": self.args.forward_rtol, "comparisons": comparisons,
                   "branches": [{key: value for key, value in outcome.items() if key not in ("state", "preclip", "postclip")}
                                for outcome in outcomes],
                   "scope": "three real updates from identical complete state; strict original model/optimizer tolerance; exact RNG/loader/modes/flags",
                   "interpretation": "no_audit_repeat_also_exceeds_gate; audit causality not established"
                   if not comparisons["no_audit_repeat"]["passed"] else (
                       "only_audit_comparison_exceeds_gate; investigate diagnostic or backend execution-path effects" if errors else "passed"),
                   "strict_tolerances_changed": False,
                   "selected_acceptance_claim": "original strict tensor tolerances" if self.args.replay_policy == "strict" else calibrated["claim"]}
        dd.write_json(self.diag / "diagnostic_noninterference.json", summary)
        self.contract["diagnostic_noninterference"] = {"policy": self.args.replay_policy,
            "passed": selected_passed, "strict_passed": not errors, "calibrated_passed": calibrated["passed"],
            "calibrated_limits": limits, "claim": summary["selected_acceptance_claim"]}
        self.write_contract()
        gate_path = self.diag / "p0_gate.json"
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())
            gate["replay_policy"] = self.args.replay_policy
            gate["diagnostic_noninterference_passed"] = selected_passed
            gate["diagnostic_noninterference_strict_passed"] = not errors
            gate["diagnostic_noninterference_calibrated_passed"] = calibrated["passed"]
            gate["calibrated_limits"] = limits
            gate["passed"] = bool(selected_passed and gate["route_checks_passed"] and gate.get("healthy_probe_completed"))
            dd.write_json(gate_path, gate)
        if not selected_passed:
            raise RuntimeError(f"Diagnostic replay failed the explicit {self.args.replay_policy} policy")

    def train(self):
        from replay_cva_depth_counterfactual import digest
        self.model.train()
        for module in self.model.modules():
            if hasattr(module, "is_training"):
                module.is_training = True
        if self.step == 0:
            self.probe()
            self.checkpoint("initial")
        started = time.perf_counter()
        while self.step < self.args.max_steps:
            batch, indices = self.stream.next()
            epoch = self.initial_epoch + self.stream.epoch
            if self.args.lr_schedule == "source_cosine":
                self.trainer.adjust_learning_rate(min(epoch, self.cfg.max_epoch))
            self.optimizer.zero_grad(set_to_none=True)
            pairing = None
            if self.args.probe_schedule == "cold":
                pairing = {"step": self.step + 1, "batch_indices": indices,
                           "batch_sha256": digest(batch), "loader": self.stream.state_dict(),
                           "rng_before_sha256": digest(dd.capture_rng_state())}
            loss, endpoints = self.forward(batch)
            policy_metrics = depth_init.backward_with_policy(loss, self.losses(endpoints)["view"] * self.weights["view"],
                self.depth_gradient_scope, self.args.depth_gradient_policy, self.model.parameters())
            group_before = {name: math.sqrt(sum(float(parameter.grad.detach().float().square().sum())
                for _, parameter in group if parameter.grad is not None)) for name, group in self.groups.items()}
            norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm, error_if_nonfinite=True))
            clip_coef = min(1.0, self.args.clip_norm / (norm + 1e-6))
            should_log = (self.step + 1) % (10 if self.step < 100 else 50) == 0 or self.step == 0
            if self.args.probe_schedule == "cold":
                should_log = (self.step + 1) % 10 == 0 or self.step == 0
            before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()
                      if parameter.requires_grad} if should_log else None
            self.optimizer.step()
            self.step += 1
            self.seen_images += len(indices)
            if pairing is not None:
                pairing["rng_after_sha256"] = digest(dd.capture_rng_state())
                dd.append_jsonl(self.diag / "pairing_steps.jsonl", pairing)
                if self.step <= 100:
                    dd.append_jsonl(self.diag / "early_steps.jsonl", json_safe({"step": self.step,
                        "metrics_scope": "actual training batch, GT-valid; not the fixed foreground event",
                        "depth": dd.depth_metrics(endpoints["depth_net_pred"].detach().cpu(), batch["gt_depth_m"],
                                                  raw=endpoints["depth_head_raw_pred"].detach().cpu()),
                        "coverage": self.scalar_metrics(endpoints)}))
            if should_log:
                update_norms = {name: math.sqrt(sum(float((parameter.detach() - before[pname]).float().square().sum())
                    for pname, parameter in group if pname in before)) for name, group in self.groups.items()}
                raw_losses = {name: float(value.detach()) for name, value in self.losses(endpoints).items()}
                dd.append_jsonl(self.diag / "train_steps.jsonl", {"step": self.step, "arm": self.args.arm,
                    "seen_images": self.seen_images, "batch_indices": indices, "stream": self.stream.state_dict(),
                    "loss": float(loss.detach()), "raw_losses": raw_losses,
                    "weighted_losses": {key: value * self.weights[key] for key, value in raw_losses.items()},
                    "coverage": self.scalar_metrics(endpoints), "global_grad_preclip": norm,
                    "gradient_policy": self.args.depth_gradient_policy, "gradient_policy_metrics": policy_metrics,
                    "clip_coefficient": clip_coef, "group_grad_preclip": group_before,
                    "group_grad_postclip": {key: value * clip_coef for key, value in group_before.items()},
                    "actual_parameter_update_norm": update_norms, "lr": [group["lr"] for group in self.optimizer.param_groups],
                    "step_skipped": False, "elapsed_s": time.perf_counter() - started})
                print(f"[train] arm={self.args.arm} step={self.step}/{self.args.max_steps} loss={float(loss):.6g} grad={norm:.4g}", flush=True)
            del loss, endpoints, before
            if probe_due(self.step, self.args) or self.step == self.args.max_steps:
                self.probe()
            if self.args.audit_interval and self.step % self.args.audit_interval == 0:
                self.audit(count=1, routes=[self.args.routes])
            event_tag = "first_threshold" if self.events["first_threshold"] == self.step else (
                "confirmed_event" if self.events["confirmed_event"] == self.step else "rolling")
            if self.args.probe_schedule == "cold":
                for name in ("first_nonflat_step", "nonflat_confirmed_step", "first_anomaly", "first_threshold", "confirmed_event"):
                    if self.events.get(name) == self.step:
                        self.locked_steps.update([self.step, *[row["step"] for row in self.manifest[-6:]]])
                if self.step in (50, 200, 500, 2000, 5000, 10000):
                    self.locked_steps.add(self.step)
                save_due = probe_due(self.step, self.args)
            else:
                save_due = self.step % self.args.checkpoint_interval == 0
            if save_due or event_tag != "rolling" or self.step == self.args.max_steps:
                self.checkpoint("final" if self.step == self.args.max_steps else event_tag)
            # Keep paired requested budgets; a confirmed event is reported, not used
            # to stop one arm early. Launcher decides a matched extension <=200 steps.
        self.contract["completion"] = {"step": self.step, "seen_images": self.seen_images,
                                        "events": self.events, "status": "completed_requested_budget"}
        self.write_contract()


def main(argv=None):
    global torch, np, dd, ORIGINAL_COMMAND
    ORIGINAL_COMMAND = [sys.executable, *sys.argv]
    args, remaining = parse_args(argv)
    import torch
    import numpy as np
    import depth_dynamics as dd
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CVA diagnostics require the project CUDA/custom-op environment")
    experiment = None
    try:
        experiment = Experiment(args, remaining)
        if args.mode == "initialize":
            manifest = experiment.contract["checkpoint"]["canonical_manifest"]
            print(json.dumps({k: manifest[k] for k in ("path", "sha256", "init_mode", "seed")}), flush=True)
            return
        if args.skip_initial_audit:
            prior = json.loads(Path(args.audit_contract).read_text())
            if not prior.get("passed") or prior["init_sha256"] != experiment.init_sha:
                raise ValueError("Audit contract must have passed P0 on exactly this initialization")
            if prior.get("replay_policy", "strict") != args.replay_policy:
                raise ValueError("Audit and training replay acceptance policies differ")
            if prior["git_head"] != experiment.contract["git"]["head"]:
                raise ValueError("Audit and training code HEAD differ")
            if prior["tracked_diff_sha256"] != experiment.contract["git"]["tracked_diff_sha256"]:
                raise ValueError("Audit and training source diff differ")
            config = {key: value for key, value in vars(experiment.cfg).items() if key not in ("log_dir",)}
            config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            if prior["resolved_config_sha256"] != config_sha:
                raise ValueError("Audit and training resolved configurations differ")
            experiment.contract["p0_passed"] = True
            experiment.contract["audit_reference"] = str(Path(args.audit_contract).resolve())
            experiment.contract["audit_acceptance"] = {
                "replay_policy": prior.get("replay_policy", "strict"),
                "strict_noninterference_passed": prior.get("diagnostic_noninterference_strict_passed"),
                "calibrated_noninterference_passed": prior.get("diagnostic_noninterference_calibrated_passed"),
                "selected_noninterference_passed": prior.get("diagnostic_noninterference_passed"),
                "calibrated_limits": prior.get("calibrated_limits")}
            experiment.write_contract()
        elif not args.resume_checkpoint:
            experiment.audit()
        if args.verify_diagnostic_step or (args.mode == "train" and not args.skip_initial_audit and not args.resume_checkpoint):
            experiment.verify_diagnostic_step()
        if args.mode == "train":
            experiment.train()
        elif not args.skip_initial_audit and args.resume_checkpoint:
            experiment.audit()
        if args.mode == "audit":
            experiment.probe()
    except BaseException as error:
        if experiment is not None:
            if isinstance(error, FloatingPointError):
                torch.save(dd.snapshot_state(experiment.model, experiment.optimizer,
                    loader=experiment.stream.state_dict(), failed_step=experiment.step,
                    error=str(error), forensic_only=True), experiment.output / "failure_state.pt")
            dd.write_json(experiment.diag / "failure.json", {"type": type(error).__name__, "message": str(error),
                          "step": experiment.step, "time": time.time()})
        raise
    finally:
        if experiment is not None:
            experiment.trainer.close()


if __name__ == "__main__":
    main()
