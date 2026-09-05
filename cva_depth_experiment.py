"""Shared runtime for the independent CVA depth training/diagnostic entry points.

Repository model/dataset imports are lazy, so --help and tensor self-tests do
not load CUDA extensions. Existing training and inference entry points are not
modified. Diagnostic overrides are scoped to one model instance and restored.
"""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from cva_depth_geometry import PairConfig, depth_b1hw


CONTRACT_VERSION = 1
GRASP_WEIGHT_DEFAULTS = {"objectness": 1.0, "graspness": 10.0, "view": 100.0, "score": 1.0, "width": 10.0}
CPU_LABEL_KEYS = {
    "object_poses_list", "grasp_points_list", "view_graspness_list",
    "top_view_index_list", "grasp_cdf_bins_list", "grasp_widths_depth_list",
    "grasp_width_valids_depth_list",
}
INPUT_KEYS = CPU_LABEL_KEYS | {
    "img", "K", "gt_depth_m", "objectness_label_tok", "graspness_label_tok",
    "token_valid_mask", "cdf_thresholds", "camera_pose_vec", "camera_gravity_vec",
    "camera_tilt_deg", "scene_idx", "anno_idx", "dataset_idx",
}


def add_common_arguments(parser):
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint_path", required=True, help="Healthy full RGB student checkpoint.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera", choices=("realsense", "kinect"), default="realsense")
    parser.add_argument("--cdf_label_folder", default=os.environ.get(
        "CDF_LABEL_FOLDER", "economic_grasp_label_300views_extend_angle_cdf_depth"))
    parser.add_argument("--pose_depth_mode", choices=("none", "global_film", "ray_gravity_film"), default=None,
                        help="Default: checkpoint metadata; explicit values must agree with metadata.")
    parser.add_argument("--use_fuse_depth", type=int, choices=(0, 1), default=None,
                        help="GT depth construction, read from checkpoint by default. Never an RGB-D model input.")
    parser.add_argument("--min_depth", type=float, default=0.2)
    parser.add_argument("--max_depth", type=float, default=1.0)
    parser.add_argument("--bin_num", type=int, default=256)
    parser.add_argument("--m_point", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--scene_ids", default="", help="Optional comma-separated integer scene IDs.")
    parser.add_argument("--graspness_mode", choices=("scene", "instance"), default="scene")
    for name, default in GRASP_WEIGHT_DEFAULTS.items():
        parser.add_argument(f"--{name}_loss_weight", type=float, default=default)
    parser.add_argument("--anchors_per_image", type=int, default=128)
    parser.add_argument("--pairs_per_anchor", type=int, default=8)
    parser.add_argument("--pair_radius_min_m", type=float, default=0.005)
    parser.add_argument("--pair_radius_max_m", type=float, default=0.03)
    parser.add_argument("--visibility_tolerance_m", type=float, default=0.005)
    parser.add_argument("--control_depth_tolerance_m", type=float, default=0.01)
    parser.add_argument("--relative_huber_beta_m", type=float, default=0.005)


def pair_config(args):
    return PairConfig(anchors_per_image=args.anchors_per_image,
                      pairs_per_anchor=args.pairs_per_anchor,
                      radius_min_m=args.pair_radius_min_m, radius_max_m=args.pair_radius_max_m,
                      visibility_tolerance_m=args.visibility_tolerance_m,
                      control_depth_tolerance_m=args.control_depth_tolerance_m,
                      min_depth=args.min_depth, max_depth=args.max_depth,
                      huber_beta_m=args.relative_huber_beta_m)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def source_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path, data):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, allow_nan=False) + "\n")


def validate_output_dir(path, resume=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is nonempty: {path}. Use a fresh directory or --resume for training.")
    path.mkdir(parents=True, exist_ok=True)


def configure_repository(args, checkpoint):
    """Initialize the legacy import-time parser once, without consuming our CLI."""
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        from utils.arguments import cfgs
    finally:
        sys.argv = original_argv
    if checkpoint.get("geometry_depth_source") != "pred" or checkpoint.get("seed_selection_mode") != "image_fps":
        raise ValueError("Use a full predicted-depth/image-FPS Stage-1/2 student checkpoint, not a GT teacher or legacy point-FPS checkpoint.")
    for key in ("pose_depth_mode", "use_fuse_depth"):
        saved = checkpoint.get(key)
        requested = getattr(args, key)
        if saved is None and requested is None:
            raise ValueError(f"Checkpoint lacks {key}; provide it explicitly.")
        if requested is not None and saved is not None and requested != saved:
            raise ValueError(f"{key} disagrees with checkpoint: requested={requested}, saved={saved}.")
        setattr(args, key, saved if requested is None else requested)
    saved_args = checkpoint.get("depth_geometry_args", {})
    for key in ("min_depth", "max_depth", "bin_num", "camera"):
        if key in saved_args and saved_args[key] != getattr(args, key):
            raise ValueError(f"Checkpoint {key}={saved_args[key]} differs from requested {getattr(args, key)}.")
    for key, value in vars(args).items():
        if hasattr(cfgs, key):
            setattr(cfgs, key, value)
    for name in GRASP_WEIGHT_DEFAULTS:
        value = getattr(args, f"{name}_loss_weight")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name}_loss_weight must be finite and nonnegative.")
    cfgs.use_cdf = True
    cfgs.extend_angle = True
    cfgs.use_obs_depth = False
    cfgs.use_depth_comp = False
    cfgs.use_gt_depth = False
    cfgs.use_fuse_depth = bool(args.use_fuse_depth)
    cfgs.kview_mode = "A1"
    cfgs.use_top4_view_infer = False
    cfgs.vis_dir = None
    cfgs.num_view, cfgs.num_angle, cfgs.num_depth = 300, 12, 4
    for key in ("pose_hidden_dim", "ray_gravity_hidden_dim", "ray_gravity_mid_dim",
                "camera_pose_key", "camera_gravity_key"):
        if key in checkpoint:
            setattr(cfgs, key, checkpoint[key])
    for key, expected in (("camera_pose_key", "camera_pose_vec"), ("camera_gravity_key", "camera_gravity_vec")):
        if checkpoint.get(key, expected) != expected:
            raise ValueError(f"This dataset adapter produces {expected}; custom {key} is unsupported.")
    return cfgs


def load_model(args, device, training=False):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a full checkpoint with model_state_dict and input-contract metadata.")
    cfgs = configure_repository(args, checkpoint)
    from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
    model = economicgrasp_dpt_student(is_training=training, use_cdf=True, use_obs_depth=False,
                                    pose_depth_mode=args.pose_depth_mode, min_depth=args.min_depth,
                                    max_depth=args.max_depth, bin_num=args.bin_num,
                                    pose_hidden_dim=int(checkpoint.get("pose_hidden_dim", 64)),
                                    ray_gravity_hidden_dim=int(checkpoint.get("ray_gravity_hidden_dim", 64)),
                                    ray_gravity_mid_dim=int(checkpoint.get("ray_gravity_mid_dim", 32)),
                                    use_depth_comp=False, vis_dir=None)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    assert_detach_contract(model)
    return model, checkpoint, cfgs


def assert_detach_contract(model):
    if not model.spatial_enhancer.detach_depth_grad or not model.kview_grasp_module.group.config.detach_depth:
        raise RuntimeError("This experiment requires the existing geometry detach switches to remain enabled.")
    if any(p.requires_grad for p in model.depth_net.depthnet.pretrained.parameters()):
        raise RuntimeError("DINO must stay frozen for this experiment.")
    if model.use_obs_depth or model.geometry_depth_source != "pred":
        raise RuntimeError("Expected an RGB-only predicted-depth student.")


class CompactFrames(Dataset):
    """Drop unused dense depth-bin and point-cloud arrays before worker IPC."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {k: v for k, v in self.dataset[index].items() if k in INPUT_KEYS}


def select_frame_indices(base, scene_ids, frame_stride, max_frames, excluded_scenes=()):
    if frame_stride < 1 or max_frames < 0:
        raise ValueError("frame_stride must be positive; max_frames must be nonnegative.")
    scene_filter = {int(s) for s in scene_ids.split(",") if s.strip()}
    excluded_scenes = set(excluded_scenes)
    indices = [i for i in range(len(base))
               if int(base.frameid[i]) % frame_stride == 0
               and int(base.scenename[i].split("_")[-1]) not in excluded_scenes
               and (not scene_filter or int(base.scenename[i].split("_")[-1]) in scene_filter)]
    if max_frames and len(indices) > max_frames:
        indices = [indices[i] for i in np.linspace(0, len(indices) - 1, max_frames, dtype=int)]
    return indices


def make_dataset(args, split, max_frames=0, excluded_scenes=()):
    from dataset.graspnet_dataset import GraspNetMultiDataset
    from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
    if args.frame_stride < 1 or max_frames < 0:
        raise ValueError("frame_stride must be positive; max_frames must be nonnegative.")
    base = GraspNetMultiDataset(args.dataset_root, camera=args.camera, split=split,
                               num_points=20000, voxel_size=0.005, remove_outlier=True,
                               augment=False, use_gt_depth=False, use_fuse_depth=bool(args.use_fuse_depth),
                               graspness_mode=args.graspness_mode, min_depth=args.min_depth,
                               max_depth=args.max_depth, bin_num=args.bin_num, depth_strides=1,
                               extend_angle=True, load_grasp_payload=False)
    dataset = CVAExtendedLabelAdapter(base, dataset_root=args.dataset_root, use_cdf=True,
                                     label_folder=args.cdf_label_folder, num_angle=12, num_depth=4)
    indices = select_frame_indices(base, args.scene_ids, args.frame_stride, max_frames, excluded_scenes)
    if not indices:
        raise ValueError(f"No frames selected for split={split}.")
    return Subset(CompactFrames(dataset), indices), indices


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) and key not in CPU_LABEL_KEYS else value
            for key, value in batch.items()}


def assert_cpu_label_residency(batch):
    """Check the collated cache at the model boundary, after any DDP wrapping."""
    for key in sorted(CPU_LABEL_KEYS.intersection(batch)):
        samples = batch[key]
        if not isinstance(samples, (list, tuple)):
            raise TypeError(f"{key} must be a per-sample list of CPU object tensors.")
        for sample_i, objects in enumerate(samples):
            if not isinstance(objects, (list, tuple)):
                raise TypeError(f"{key}[{sample_i}] must be a list of CPU object tensors.")
            for object_i, tensor in enumerate(objects):
                if not torch.is_tensor(tensor) or tensor.device.type != "cpu":
                    location = tensor.device if torch.is_tensor(tensor) else type(tensor).__name__
                    raise RuntimeError(
                        f"{key}[{sample_i}][{object_i}] must remain CPU-resident; got {location}. "
                        "Use DDP device_ids=None and move_batch() for explicit dense-input transfer."
                    )


def grasp_objective(end_points, cfgs):
    """Use main's five supervised grasp terms; exclude metric depth entirely."""
    from models import loss_economicgrasp_depth_kview_transformer as losses
    functions = (("objectness", losses.compute_objectness_loss_tok, cfgs.objectness_loss_weight),
                 ("graspness", losses.compute_graspness_loss_tok, cfgs.graspness_loss_weight),
                 ("view", losses.compute_view_graspness_loss, cfgs.view_loss_weight),
                 ("cdf", lambda ep: losses.compute_cva_cdf_loss(ep, balanced=False), cfgs.score_loss_weight),
                 ("width", losses.compute_cva_width_depth_loss, cfgs.width_loss_weight))
    terms = {}
    total = end_points["depth_map_pred"].new_zeros(())
    for name, function, weight in functions:
        value, end_points = function(end_points)
        terms[name] = value
        total = total + float(weight) * value
    return total, terms


def tensor_copy(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(tensor_copy(v) for v in value)
    if isinstance(value, list):
        return [tensor_copy(v) for v in value]
    if isinstance(value, dict):
        return {k: tensor_copy(v) for k, v in value.items()}
    return value


@contextlib.contextmanager
def capture_depth_outputs(model):
    captured = []
    hook = model.depth_net.register_forward_hook(lambda module, inputs, output: captured.append(tensor_copy(output)))
    try:
        yield captured
    finally:
        hook.remove()


@contextlib.contextmanager
def replay_depth(model, cached_outputs, depth):
    """Reuse frozen DINO/DPT outputs; intervene only on the geometry-depth value."""
    original = model.depth_net.forward
    def forward(*args, **kwargs):
        output = list(cached_outputs)
        output[0] = depth
        return tuple(output)
    model.depth_net.forward = forward
    try:
        yield
    finally:
        model.depth_net.forward = original


@contextlib.contextmanager
def fixed_anchor_seeds(model, xyz, pixel):
    xyz, pixel = tensor_copy(xyz), tensor_copy(pixel)
    original = model._select_graspable_seed_queries
    def select(*, feat_grid, depth_map, camera_K, graspable_mask, valid_tok, grasp_score, end_points):
        batch_size, channels = feat_grid.shape[:2]
        if xyz.shape != (*pixel.shape, 3) or pixel.shape[0] != batch_size:
            raise ValueError("Fixed anchor query shape mismatch.")
        features = feat_grid.flatten(2).gather(2, pixel[:, None].expand(-1, channels, -1))
        return features, xyz.clone(), pixel.clone(), None, None, float(pixel.numel())
    model._select_graspable_seed_queries = select
    try:
        yield
    finally:
        model._select_graspable_seed_queries = original


class FixedSupportReplay:
    """Record then replay BOTH label passes and the actual sampling grids.

    Stop-gradient alone would not freeze forward values. Replayed tensors are
    literal copies from the GT reference. The dense view prediction is retained;
    only selected views are forced by the repository's exact-query selector.
    """
    def __init__(self, model, views):
        self.model, self.views = model, tensor_copy(views)
        self.grids, self.labels = [], []
        self.recording = True
        self.grid_cursor = self.label_cursor = 0

    def begin(self, recording=False):
        if recording and (self.grids or self.labels):
            raise RuntimeError("The GT reference can only be recorded once.")
        self.recording = recording
        self.grid_cursor = self.label_cursor = 0

    def finish(self):
        if self.label_cursor != 2:
            raise RuntimeError(f"Expected both base and query label passes, got {self.label_cursor}.")
        if not self.grid_cursor:
            raise RuntimeError("Fixed-support diagnosis did not record or replay any sampling grids.")
        if not self.recording and (self.grid_cursor != len(self.grids) or self.label_cursor != len(self.labels)):
            raise RuntimeError("Fixed-support replay did not consume its exact reference contract.")

    def __enter__(self):
        group = self.model.kview_grasp_module.group
        self.original_grid = group._make_view_conditioned_grid
        def make_grid(*args, **kwargs):
            if args:
                raise RuntimeError("Expected the repository's keyword-only grid call site.")
            index = self.grid_cursor
            self.grid_cursor += 1
            # depth_map is the intervention. All query/calibration inputs are
            # fixed, and the actual grid output is replayed even if a future
            # grid builder starts using depth_map values internally.
            fixed_inputs = {k: v for k, v in kwargs.items() if k != "depth_map"}
            if self.recording:
                output = self.original_grid(*args, **kwargs)
                self.grids.append((tensor_copy(fixed_inputs), tensor_copy(output)))
                return output
            if index >= len(self.grids):
                raise RuntimeError("More sampling-grid calls than in the GT reference.")
            expected, output = self.grids[index]
            if expected.keys() != fixed_inputs.keys():
                raise RuntimeError("Fixed grid call signature changed.")
            for key, value in expected.items():
                current = kwargs[key]
                same = torch.equal(value, current) if torch.is_tensor(value) else value == current
                if not same:
                    raise RuntimeError(f"Fixed query/grid input changed: {key}.")
            return tensor_copy(output)
        group._make_view_conditioned_grid = make_grid

        def prepare(module, positional, kwargs):
            original_label = kwargs.get("process_grasp_labels_fn")
            if original_label is None:
                raise RuntimeError("Fixed-support diagnosis requires label processing.")
            def label_process(ep, **extra):
                index = self.label_cursor
                self.label_cursor += 1
                if self.recording:
                    ep["grasp_top_view_inds"] = self.views.clone()
                    rotation, ep = original_label(ep, **extra)
                    labels = {k: tensor_copy(v) for k, v in ep.items() if k.startswith("batch_")}
                    self.labels.append((tensor_copy(rotation), labels))
                    return rotation, ep
                if index >= len(self.labels):
                    raise RuntimeError("More label passes than in the GT reference.")
                rotation, labels = self.labels[index]
                ep.update(tensor_copy(labels))
                return tensor_copy(rotation), ep
            kwargs["process_grasp_labels_fn"] = label_process
            return positional, kwargs
        self.hook = self.model.kview_grasp_module.register_forward_pre_hook(prepare, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.hook.remove()
        self.model.kview_grasp_module.group._make_view_conditioned_grid = self.original_grid

    def fingerprint(self):
        digest = hashlib.sha256()
        for _, output in self.grids:
            for tensor in output:
                if torch.is_tensor(tensor):
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        for _, labels in self.labels:
            for key, tensor in sorted(labels.items()):
                if torch.is_tensor(tensor):
                    digest.update(key.encode())
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def checkpoint_metadata(model, args):
    return {"distill_stage": 1, "distill_contract_version": 2,
            "seed_selection_mode": "image_fps", "geometry_depth_source": "pred",
            "depth_head_executed": True, "pose_depth_mode": args.pose_depth_mode,
            "use_fuse_depth": bool(args.use_fuse_depth), "legacy_dataset_use_gt_depth": False,
            "stage2_shared_teacher_image_fps": False,
            "camera_pose_key": model.camera_pose_key, "camera_gravity_key": model.camera_gravity_key,
            "pose_hidden_dim": model.pose_hidden_dim, "ray_gravity_hidden_dim": model.ray_gravity_hidden_dim,
            "ray_gravity_mid_dim": model.ray_gravity_mid_dim,
            "depth_geometry_contract_version": CONTRACT_VERSION,
            "depth_geometry_args": vars(args).copy(), "source_revision": source_revision()}
