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
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognized options are forwarded to utils.arguments (e.g. "
               "--dataset_root, --batch_size, --learning_rate, loss weights).")
    parser.add_argument("--mode", choices=("audit", "train"), default="audit")
    parser.add_argument("--init_checkpoint", required=True)
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
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--keep_last", type=int, default=6)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--lr_schedule", choices=("constant", "source_cosine"), default="constant")
    parser.add_argument("--forward_atol", type=float, default=1e-6)
    parser.add_argument("--forward_rtol", type=float, default=1e-5)
    parser.add_argument("--skip_initial_audit", action="store_true",
                        help="Only for already-audited paired runs; provide --audit_contract.")
    parser.add_argument("--audit_contract", "--audit_gate", dest="audit_contract", default="",
                        help="Successful P0 gate to authorize skipping redundant P0.")
    parser.add_argument("--verify_diagnostic_step", action="store_true",
                        help="Verify one real optimizer update is identical with/without a diagnostic replay.")
    args, remaining = parser.parse_known_args(argv)
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


class Experiment:
    def __init__(self, args, remaining):
        self.args = args
        self.output = Path(args.output).resolve()
        self.diag = Path(args.diagnostics_dir).resolve() if args.diagnostics_dir else self.output / "diagnostics"
        if not args.resume_checkpoint and self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.diag.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
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
        result = self.model.load_state_dict(self.source["model_state_dict"], strict=True)
        self.model.set_depth_grad_routes(args.routes)
        self.groups = dd.parameter_groups(self.model)
        self.stream = DeterministicStream(self.trainer.TRAIN_DATASET, self.cfg.batch_size, args.seed,
                                          original.collate_fn, dd.capture_rng_state, dd.restore_rng_state)
        self.weights = {key: float(getattr(self.cfg, attr)) for key, attr in WEIGHT_KEYS.items()}
        self.step, self.seen_images = 0, 0
        self.initial_epoch = int(self.source.get("epoch", 0))
        self.events = {"initial_flat": None, "first_anomaly": None, "first_threshold": None,
                       "confirmed_event": None, "consecutive_thresholds": 0}
        self.manifest = []
        self.locked_steps = set()
        self.train_rows, train_sha = dataset_manifest(self.trainer.TRAIN_DATASET)
        self.probe_specs = [("train", index) for index in select_probe_indices(
            self.trainer.TRAIN_DATASET, args.train_probe_frames)]
        self.probe_specs += [("validation_test_seen", index) for index in select_probe_indices(
            self.trainer.TEST_DATASET, args.heldout_test_probe_frames)]
        self.init_sha = dd.sha256_file(args.init_checkpoint)
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
            "checkpoint": {"path": str(Path(args.init_checkpoint).resolve()), "sha256": self.init_sha,
                           "metadata": {k: v for k, v in self.source.items() if k not in ("model_state_dict", "optimizer_state_dict")},
                           "initialization": "weights-only restart; AdamW reset identically for both arms",
                           "source_has_optimizer": "optimizer_state_dict" in self.source,
                           "missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys},
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
                             "schedule": args.lr_schedule, "scheduler_state": "derived source_epoch + stream.epoch",
                             "source_epoch": self.initial_epoch, "scaler": None},
            "events": {"std_ratio_threshold": 0.1, "eligible_gt_std_min_m": 0.005,
                       "fraction_images": 0.8, "confirmations": 3, "locked_before_training": True},
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
        self.probe_cache = self.make_probe_cache()
        self.contract["probe_manifest"] = [{k: v for k, v in item.items() if k in ("split", "index", "scene", "frame")}
                                           for item in self.probe_cache]
        self.init_rng = dd.capture_rng_state()
        if args.resume_checkpoint:
            self.resume(args.resume_checkpoint)
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

    def forward(self, batch, capture=False):
        inputs = dict(batch)
        self.original.drop_unused_point_inputs(inputs)
        self.original.validate_batch_label_contract(inputs, use_cdf=True)
        inputs = self.original.move_batch_to_device(inputs, self.trainer.device, use_cdf=True, non_blocking=False)
        inputs.update(depth_grad_capture_routes=capture, cva_compute_diagnostics=True,
                      geometry_compute_diagnostics=False, cva_export_angle_feature=False,
                      cva_force_process_grasp_labels=True)
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
        return {key: json_safe(value) for key, value in metrics.items()}

    def equality_tensors(self, endpoints):
        # Include logits, seed/view identities, masks, matched targets and every loss.
        explicit = {"depth_net_pred", "depth_head_raw_pred", "objectness_score", "graspness_score",
                    "view_score", "xyz_graspable", "token_sel_idx", "grasp_top_view_inds",
                    "grasp_top_view_xyz", "grasp_cdf_pred_angle_depth", "grasp_width_pred_angle_depth"}
        return {key: value.detach().cpu().clone() for key, value in endpoints.items()
                if torch.is_tensor(value) and (key in explicit or key.startswith("batch_grasp_")
                   or "valid_mask" in key or key.startswith("dbg_mask") or key.startswith("B:"))}

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
            reference = None
            for route in routes:
                self.model.set_depth_grad_routes(route)
                with self.diagnostic_context(train_mode=True, seed=stable_seed(self.args.seed, "audit", batch_id)):
                    loss, endpoints = self.forward(batch, capture=True)
                    values = self.equality_tensors(endpoints)
                    failures = []
                    if reference is None:
                        reference = values
                    else:
                        for key in reference.keys() | values.keys():
                            if key not in reference or key not in values:
                                failures.append({"key": key, "reason": "missing endpoint"})
                                continue
                            a, b = reference[key], values[key]
                            equal = a.shape == b.shape and (torch.allclose(a, b, atol=self.args.forward_atol,
                                      rtol=self.args.forward_rtol, equal_nan=False) if a.is_floating_point() else torch.equal(a, b))
                            if not equal:
                                failures.append({"key": key, "reason": "value mismatch",
                                                 "max_abs": float((a.float() - b.float()).abs().max()) if a.shape == b.shape else None})
                    all_pass &= not failures
                    dd.append_jsonl(self.diag / "forward_equality.jsonl", {"step": self.step, "batch": batch_id,
                        "indices": indices, "route": route, "compared": sorted(values), "passed": not failures,
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
                                geometry_compute_diagnostics=False, depth_grad_capture_routes=True)
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

    def record_event(self, metrics):
        data = [entry["regions"]["foreground"]["std_ratio"] for entry in metrics
                if entry["regions"]["foreground"]["count"] >= 2
                and entry["regions"]["foreground"]["gt_std"] is not None
                and entry["regions"]["foreground"]["gt_std"] > 0.005
                and entry["regions"]["foreground"]["std_ratio"] is not None]
        fraction = sum(value < .1 for value in data) / len(data) if data else None
        if self.events["initial_flat"] is None:
            self.events["initial_flat"] = bool(fraction is not None and fraction >= .8)
        threshold = bool(fraction is not None and fraction >= .8 and not self.events["initial_flat"])
        if fraction is not None and fraction > 0 and self.step > 0 and self.events["first_anomaly"] is None:
            self.events["first_anomaly"] = self.step
            self.locked_steps.update(row["step"] for row in self.manifest[-6:])
        if threshold and self.events["first_threshold"] is None:
            self.events["first_threshold"] = self.step
        self.events["consecutive_thresholds"] = self.events["consecutive_thresholds"] + 1 if threshold else 0
        if self.events["consecutive_thresholds"] >= 3 and self.events["confirmed_event"] is None:
            self.events["confirmed_event"] = self.step
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

    def resume(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError("Resume requires a full dynamics checkpoint")
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
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.stream.load_state_dict(state["loader"])
        self.step, self.seen_images = state["step"], state["seen_images"]
        self.events, self.locked_steps = state["events"], set(state["locked_steps"])
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
        if any(entry["step"] > self.step and not entry.get("deleted_after_retention") for entry in self.manifest):
            raise ValueError("Resume from an older snapshot into a new output directory to avoid forking an existing trajectory")
        self.contract["resume"] = {"path": str(Path(path).resolve()), "sha256": dd.sha256_file(path), "step": self.step}

    def verify_diagnostic_step(self):
        """Real one-update paired replay; restore original state after verification."""
        path = self.checkpoint("initial" if self.step == 0 else "replay_reference")
        initial = torch.load(path, map_location="cpu", weights_only=False)
        outcomes = []
        for with_diagnostics in (False, True):
            self.model.load_state_dict(initial["model_state_dict"])
            self.optimizer.load_state_dict(copy.deepcopy(initial["optimizer_state_dict"]))
            self.stream.load_state_dict(initial["loader"])
            dd.restore_rng_state(initial["rng"])
            for name, module in self.model.named_modules():
                for key, value in initial.get("runtime_counters", {}).get(name, {}).items():
                    setattr(module, key, value)
            self.model.train()
            for module in self.model.modules():
                if hasattr(module, "is_training"):
                    module.is_training = True
            if with_diagnostics:
                self.audit(count=1, routes=[self.args.routes])
            batch, _ = self.stream.next()
            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm, error_if_nonfinite=True)
            self.optimizer.step()
            outcomes.append(dd.snapshot_state(self.model, self.optimizer, loader=self.stream.state_dict()))
        errors = compare_state(outcomes[0]["model_state_dict"], outcomes[1]["model_state_dict"],
                               atol=self.args.forward_atol, rtol=self.args.forward_rtol, path="model_and_buffers")
        errors += compare_state(outcomes[0]["optimizer_state_dict"], outcomes[1]["optimizer_state_dict"],
                                atol=self.args.forward_atol, rtol=self.args.forward_rtol, path="optimizer")
        errors += compare_state(outcomes[0]["rng_state"], outcomes[1]["rng_state"], path="RNG")
        errors += compare_state(outcomes[0]["module_modes"], outcomes[1]["module_modes"], path="module_modes")
        errors += compare_state(outcomes[0]["module_flags"], outcomes[1]["module_flags"], path="module_flags")
        errors += compare_state(outcomes[0]["metadata"], outcomes[1]["metadata"], path="loader")
        self.model.load_state_dict(initial["model_state_dict"])
        self.optimizer.load_state_dict(copy.deepcopy(initial["optimizer_state_dict"]))
        self.stream.load_state_dict(initial["loader"])
        self.optimizer.zero_grad(set_to_none=True)
        dd.restore_rng_state(initial["rng"])
        for name, module in self.model.named_modules():
            module.training = initial["module_training"][name]
            if name in initial["is_training"]:
                module.is_training = initial["is_training"][name]
            for key, value in initial.get("runtime_counters", {}).get(name, {}).items():
                setattr(module, key, value)
        dd.write_json(self.diag / "diagnostic_noninterference.json", {"passed": not errors,
                      "errors": errors, "atol": self.args.forward_atol, "rtol": self.args.forward_rtol,
                      "scope": "one real update with/without actual audit; parameters/buffers/optimizer tolerance, exact RNG/loader/modes/flags"})
        gate_path = self.diag / "p0_gate.json"
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())
            gate["diagnostic_noninterference_passed"] = not errors
            gate["passed"] = bool(not errors and gate["route_checks_passed"])
            dd.write_json(gate_path, gate)
        if errors:
            raise RuntimeError("Diagnostic replay changed the next optimizer update")

    def train(self):
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
            loss, endpoints = self.forward(batch)
            loss.backward()
            group_before = {name: math.sqrt(sum(float(parameter.grad.detach().float().square().sum())
                for _, parameter in group if parameter.grad is not None)) for name, group in self.groups.items()}
            norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm, error_if_nonfinite=True))
            clip_coef = min(1.0, self.args.clip_norm / (norm + 1e-6))
            should_log = (self.step + 1) % (10 if self.step < 100 else 50) == 0 or self.step == 0
            before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()
                      if parameter.requires_grad} if should_log else None
            self.optimizer.step()
            self.step += 1
            self.seen_images += len(indices)
            if should_log:
                update_norms = {name: math.sqrt(sum(float((parameter.detach() - before[pname]).float().square().sum())
                    for pname, parameter in group if pname in before)) for name, group in self.groups.items()}
                raw_losses = {name: float(value.detach()) for name, value in self.losses(endpoints).items()}
                dd.append_jsonl(self.diag / "train_steps.jsonl", {"step": self.step, "arm": self.args.arm,
                    "seen_images": self.seen_images, "batch_indices": indices, "stream": self.stream.state_dict(),
                    "loss": float(loss.detach()), "raw_losses": raw_losses,
                    "weighted_losses": {key: value * self.weights[key] for key, value in raw_losses.items()},
                    "coverage": self.scalar_metrics(endpoints), "global_grad_preclip": norm,
                    "clip_coefficient": clip_coef, "group_grad_preclip": group_before,
                    "group_grad_postclip": {key: value * clip_coef for key, value in group_before.items()},
                    "actual_parameter_update_norm": update_norms, "lr": [group["lr"] for group in self.optimizer.param_groups],
                    "step_skipped": False, "elapsed_s": time.perf_counter() - started})
                print(f"[train] arm={self.args.arm} step={self.step}/{self.args.max_steps} loss={float(loss):.6g} grad={norm:.4g}", flush=True)
            del loss, endpoints, before
            if self.step % self.args.probe_interval == 0 or self.step == self.args.max_steps:
                self.probe()
            if self.args.audit_interval and self.step % self.args.audit_interval == 0:
                self.audit(count=1, routes=[self.args.routes])
            event_tag = "first_threshold" if self.events["first_threshold"] == self.step else (
                "confirmed_event" if self.events["confirmed_event"] == self.step else "rolling")
            if self.step % self.args.checkpoint_interval == 0 or event_tag != "rolling" or self.step == self.args.max_steps:
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
        if args.skip_initial_audit:
            prior = json.loads(Path(args.audit_contract).read_text())
            if not prior.get("passed") or prior["init_sha256"] != experiment.init_sha:
                raise ValueError("Audit contract must have passed P0 on exactly this initialization")
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
            experiment.write_contract()
        elif not args.resume_checkpoint:
            experiment.audit()
        if args.verify_diagnostic_step or (args.mode == "train" and not args.skip_initial_audit and not args.resume_checkpoint):
            experiment.verify_diagnostic_step()
        if args.mode == "train":
            experiment.train()
        elif not args.skip_initial_audit and args.resume_checkpoint:
            experiment.audit()
    except BaseException as error:
        if experiment is not None:
            dd.write_json(experiment.diag / "failure.json", {"type": type(error).__name__, "message": str(error),
                          "step": experiment.step, "time": time.time()})
        raise
    finally:
        if experiment is not None:
            experiment.trainer.close()


if __name__ == "__main__":
    main()
