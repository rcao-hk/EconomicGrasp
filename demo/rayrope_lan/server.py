"""GPU inference server for scene-level Point RayRoPE grasps over a private LAN.

Run from the EconomicGrasp repository root:
    python -m demo.rayrope_lan.server --host 0.0.0.0 --port 8765

The model uses RGB-predicted metric depth. The transmitted sensor depth is used
only to build the model-free collision cloud; it is not fused into the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socketserver
import sys
import threading
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .protocol import decode_npz, encode_npz, recv_frame, send_frame


LOGGER = logging.getLogger("economicgrasp.rayrope_lan")
IMAGE_SIZE = (448, 448)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: str):
    # PyTorch 2.6 changed the default of weights_only. These are trusted local
    # training checkpoints produced by this repository.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def camera_pose_vec_from_transform(T_table_from_camera: np.ndarray) -> np.ndarray:
    """Match GraspNetMultiDataset.build_camera_orientation_metadata exactly."""
    transform = np.asarray(T_table_from_camera, dtype=np.float32)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_table_from_camera must be a finite 4x4 matrix.")
    vector = -transform[:3, :3][:, 2]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("T_table_from_camera has an invalid optical axis.")
    return (vector / norm).astype(np.float32)


def crop_intrinsics(
    K_full: np.ndarray,
    crop_box_xyxy: tuple[int, int, int, int],
    out_hw: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return crop-resolution K and 448x448 K using the dataset convention."""
    K = np.asarray(K_full, dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("camera_intrinsics must be a finite 3x3 matrix.")
    x0, y0, x1, y1 = (int(v) for v in crop_box_xyxy)
    crop_w, crop_h = x1 - x0, y1 - y0
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop_box_xyxy: {crop_box_xyxy}")

    K_crop = K.copy()
    K_crop[0, 2] -= float(x0)
    K_crop[1, 2] -= float(y0)

    out_h, out_w = out_hw
    K_resized = K_crop.copy()
    K_resized[0, 0] *= out_w / float(crop_w)
    K_resized[1, 1] *= out_h / float(crop_h)
    K_resized[0, 2] *= out_w / float(crop_w)
    K_resized[1, 2] *= out_h / float(crop_h)
    return K_crop, K_resized


def _metadata_json(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    value = arrays.get("metadata_json")
    if value is None:
        raise ValueError("Request is missing the metadata_json array.")
    try:
        return json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid metadata_json: {exc}") from exc


def _validate_transform(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], np.array([0, 0, 0, 1], dtype=np.float32), atol=1e-3):
        raise ValueError(f"{name} must use homogeneous-transform convention.")
    return matrix


def _validate_request(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    metadata = _metadata_json(arrays)
    rgb = np.asarray(arrays.get("rgb"))
    depth = np.asarray(arrays.get("depth"))

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(
            f"rgb must be uint8 [Hcrop,Wcrop,3] in RGB order; got "
            f"shape={rgb.shape}, dtype={rgb.dtype}."
        )
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise ValueError(
            f"depth must be [Hcrop,Wcrop] and align with rgb; "
            f"got depth={depth.shape}, rgb={rgb.shape[:2]}."
        )
    if depth.dtype.kind not in {"u", "i", "f"}:
        raise ValueError(f"Unsupported depth dtype: {depth.dtype}.")

    if "crop_box_xyxy" in metadata:
        crop_box = tuple(int(v) for v in metadata["crop_box_xyxy"])
    elif "crop_box" in metadata:
        crop_box = tuple(int(v) for v in metadata["crop_box"])
    else:
        raise ValueError("metadata requires crop_box_xyxy=[x0,y0,x1,y1].")
    if len(crop_box) != 4:
        raise ValueError("crop_box_xyxy must contain four integers.")

    original_hw = metadata.get("original_hw")
    if not isinstance(original_hw, (list, tuple)) or len(original_hw) != 2:
        raise ValueError("metadata requires original_hw=[Hfull,Wfull].")
    full_h, full_w = (int(v) for v in original_hw)
    x0, y0, x1, y1 = crop_box
    if not (0 <= x0 < x1 <= full_w and 0 <= y0 < y1 <= full_h):
        raise ValueError(
            f"crop_box_xyxy={crop_box} is outside original_hw={original_hw}."
        )
    expected_hw = (y1 - y0, x1 - x0)
    if rgb.shape[:2] != expected_hw:
        raise ValueError(
            f"RGB/depth must be the raw-resolution crop: expected {expected_hw}, "
            f"received {rgb.shape[:2]}. Do not resize on the client."
        )

    K_value = metadata.get("camera_intrinsics", metadata.get("K"))
    if K_value is None:
        raise ValueError("metadata requires camera_intrinsics (full-image K).")
    K_full = np.asarray(K_value, dtype=np.float32)
    K_crop, K_resized = crop_intrinsics(K_full, crop_box)
    if K_crop[0, 0] <= 0 or K_crop[1, 1] <= 0:
        raise ValueError("Camera focal lengths fx and fy must be positive.")

    if "camera_pose_vec" in metadata:
        pose_vec = np.asarray(metadata["camera_pose_vec"], dtype=np.float32).reshape(-1)
        if pose_vec.shape != (3,) or not np.isfinite(pose_vec).all():
            raise ValueError("camera_pose_vec must be a finite 3-vector.")
        pose_norm = float(np.linalg.norm(pose_vec))
        if pose_norm < 1e-8:
            raise ValueError("camera_pose_vec must have non-zero norm.")
        pose_vec = pose_vec / pose_norm
        pose_source = "camera_pose_vec"
    elif "T_table_from_camera" in metadata:
        pose_vec = camera_pose_vec_from_transform(metadata["T_table_from_camera"])
        pose_source = "T_table_from_camera"
    else:
        raise ValueError(
            "Pose-aware global-FiLM DPT requires camera_pose_vec or "
            "T_table_from_camera. Camera intrinsics alone are insufficient."
        )

    depth_scale_to_m = float(metadata.get("depth_scale_to_m", 0.0))
    if not np.isfinite(depth_scale_to_m) or depth_scale_to_m <= 0:
        raise ValueError(
            "metadata.depth_scale_to_m is required and must be positive "
            "(uint16 millimeters: 0.001; float meters: 1.0)."
        )

    T_base_from_camera = None
    if metadata.get("T_base_from_camera") is not None:
        T_base_from_camera = _validate_transform(
            metadata["T_base_from_camera"], "T_base_from_camera"
        )

    color_order = str(metadata.get("rgb_color_order", "RGB")).upper()
    if color_order == "BGR":
        rgb = rgb[..., ::-1].copy()
    elif color_order != "RGB":
        raise ValueError("rgb_color_order must be RGB or BGR.")

    return {
        "metadata": metadata,
        "rgb": rgb,
        "depth": depth,
        "crop_box": crop_box,
        "original_hw": (full_h, full_w),
        "K_crop": K_crop,
        "K_resized": K_resized,
        "camera_pose_vec": pose_vec.astype(np.float32),
        "pose_source": pose_source,
        "depth_scale_to_m": depth_scale_to_m,
        "T_base_from_camera": T_base_from_camera,
    }


def _depth_crop_to_cloud(
    depth: np.ndarray,
    depth_scale_to_m: float,
    K_crop: np.ndarray,
    min_depth: float,
    max_depth: float,
    max_points: int,
) -> np.ndarray:
    depth_m = np.asarray(depth, dtype=np.float32) * float(depth_scale_to_m)
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= float(min_depth))
        & (depth_m <= float(max_depth))
    )
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - float(K_crop[0, 2])) / float(K_crop[0, 0]) * z
    y = (ys.astype(np.float32) - float(K_crop[1, 2])) / float(K_crop[1, 1]) * z
    cloud = np.stack((x, y, z), axis=1).astype(np.float32, copy=False)

    # Bound CPU/GPU collision-filter cost while preserving image-wide coverage.
    if max_points > 0 and cloud.shape[0] > max_points:
        keep = np.linspace(
            0, cloud.shape[0] - 1, num=int(max_points), dtype=np.int64
        )
        cloud = cloud[keep]
    return cloud


def _empty_grasp_arrays() -> dict[str, np.ndarray]:
    return {
        "grasps_camera": np.empty((0, 17), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "widths_m": np.empty((0,), dtype=np.float32),
        "rotation_matrices_camera": np.empty((0, 3, 3), dtype=np.float32),
        "translations_camera_m": np.empty((0, 3), dtype=np.float32),
    }


class RayRoPEInferenceService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {args.device}, but CUDA is unavailable.")
        if args.collision_thresh > 0 and (
            self.device.type != "cuda" or self.device.index not in (None, 0)
        ):
            raise RuntimeError(
                "ModelFreeCollisionDetectorTorch currently allocates on cuda:0. "
                "Use --device cuda:0 (with CUDA_VISIBLE_DEVICES selecting the "
                "physical GPU), or set --collision-thresh 0 to disable it."
            )

        torch.set_num_threads(max(1, int(args.cpu_threads)))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        if not args.checkpoint:
            raise ValueError(
                "Set --checkpoint or ECONOMICGRASP_RAYROPE_CKPT to the intended "
                "two-source 50% GraspNet + 50% GN-Trans Point RayRoPE checkpoint. "
                "The exact checkpoint path is not recorded in the available "
                "experiment notes, so no path is guessed here."
            )
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(
                f"RayRoPE checkpoint not found: {args.checkpoint}. "
                "Restore the 50/50 checkpoint or pass its current path."
            )

        checkpoint_sha = _sha256_file(args.checkpoint)
        if args.expected_checkpoint_sha256:
            expected = args.expected_checkpoint_sha256.strip().lower()
            if checkpoint_sha.lower() != expected:
                raise RuntimeError(
                    "Checkpoint SHA256 mismatch: "
                    f"expected={expected}, actual={checkpoint_sha}."
                )

        checkpoint = _torch_load(args.checkpoint)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                "Expected a full EconomicGrasp checkpoint containing "
                "model_state_dict and grasp_design metadata."
            )
        design = checkpoint.get("grasp_design")
        if not isinstance(design, dict):
            raise ValueError("RayRoPE checkpoint is missing grasp_design metadata.")
        if not bool(design.get("enable_point_rayrope", False)):
            raise ValueError(
                f"Checkpoint is not a Point RayRoPE model: grasp_design={design}."
            )
        if bool(design.get("enable_gvar", False)):
            raise ValueError(
                "This demo is configured for Point RayRoPE without GVAR, but the "
                "checkpoint enables GVAR."
            )
        rayrope_cfg = design.get("rayrope")
        if not isinstance(rayrope_cfg, dict):
            raise ValueError("Checkpoint grasp_design is missing the rayrope config.")

        # Parameter-free RayRoPE settings are read from the checkpoint metadata.
        # This keeps the inference grouping identical to the training contract.
        model_argv = [
            sys.argv[0],
            "--multi_modal",
            "--use_cdf",
            "--enable_point_rayrope",
            "--kview_mode", "A1",
            "--pose_depth_mode", args.pose_depth_mode,
            "--min_depth", str(args.min_depth),
            "--max_depth", str(args.max_depth),
            "--bin_num", str(args.bin_num),
            "--num_view", str(args.num_view),
            "--num_angle", str(args.num_angle),
            "--num_depth", str(args.num_depth),
            "--m_point", str(args.m_points),
            "--graspness_threshold", str(args.graspness_threshold),
            "--grasp_max_width", str(args.grasp_max_width),
            "--rayrope_group_chunk", str(int(rayrope_cfg.get("group_chunk", 32))),
            "--rayrope_checkpoint_chunks",
            str(int(bool(rayrope_cfg.get("checkpoint_chunks", True)))),
        ]
        original_argv = sys.argv
        sys.argv = model_argv
        try:
            # Import after replacing argv because utils.arguments parses CLI
            # options at import time in the training/evaluation codebase.
            from utils.arguments import cfgs
            from models.grasp_designs import design_options, validate_checkpoint_design
            from models.economicgrasp_bip3d import (
                economicgrasp_dpt,
                pred_decode_center_view_angle,
            )
            from utils.collision_detector import ModelFreeCollisionDetectorTorch
        finally:
            sys.argv = original_argv

        self._decode = pred_decode_center_view_angle
        self._collision_detector_cls = ModelFreeCollisionDetectorTorch
        self._cfgs = cfgs
        self._validate_checkpoint_design = validate_checkpoint_design

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model = economicgrasp_dpt(
            **design_options(cfgs),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            bin_num=args.bin_num,
            is_training=False,
            use_obs_depth=False,
            pose_depth_mode=args.pose_depth_mode,
            camera_pose_key="camera_pose_vec",
            camera_gravity_key="camera_gravity_vec",
            pose_hidden_dim=64,
            ray_gravity_hidden_dim=64,
            ray_gravity_mid_dim=32,
            load_pretrained_depth_backbone=False,
            use_depth_comp=False,
            use_cdf=True,
            vis_dir=None,
            vis_every=1000,
        ).to(self.device)

        validate_checkpoint_design(self.model, checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.checkpoint_sha256 = checkpoint_sha
        self._inference_lock = threading.Lock()

        LOGGER.info(
            "Loaded Point RayRoPE checkpoint: %s (epoch=%s, sha256=%s)",
            args.checkpoint,
            checkpoint.get("epoch", "unknown"),
            checkpoint_sha,
        )
        LOGGER.info(
            "grasp_design=%s; pose_depth_mode=%s; device=%s",
            design,
            args.pose_depth_mode,
            self.device,
        )

    def _prepare_batch(self, request: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        validated = _validate_request(request)
        rgb_pil = Image.fromarray(validated["rgb"], mode="RGB")
        image_tensor = self.transform(rgb_pil).unsqueeze(0).to(self.device)
        K_tensor = torch.from_numpy(validated["K_resized"]).unsqueeze(0).to(
            self.device, dtype=torch.float32
        )
        pose_tensor = torch.from_numpy(validated["camera_pose_vec"]).unsqueeze(0).to(
            self.device, dtype=torch.float32
        )
        batch = {
            "img": image_tensor,
            "K": K_tensor,
            "camera_pose_vec": pose_tensor,
        }
        return batch, validated

    def infer(self, request: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        batch, validated = self._prepare_batch(request)
        metadata = validated["metadata"]
        request_id = str(metadata.get("request_id", ""))
        timings: dict[str, float] = {}
        start = time.perf_counter()

        with self._inference_lock:
            with torch.inference_mode():
                end_points = self.model(batch)
                predictions = self._decode(end_points, use_cdf=True)
        timings["model_ms"] = (time.perf_counter() - start) * 1000.0

        if len(predictions) != 1:
            raise RuntimeError(
                f"Single-scene request must decode one grasp group, got {len(predictions)}."
            )

        from graspnetAPI import GraspGroup

        raw_pred = predictions[0].detach().cpu().numpy()
        gg = GraspGroup(np.asarray(raw_pred, dtype=np.float64))
        raw_count = len(gg)

        finite = np.isfinite(gg.grasp_group_array).all(axis=1)
        gg = gg[finite]
        width_values = gg.grasp_group_array[:, 1]
        valid_width = (width_values > 0.0) & (
            width_values <= float(self.args.grasp_max_width) + 1e-6
        )
        gg = gg[valid_width]
        after_validity = len(gg)

        collision_removed = 0
        if self.args.collision_thresh > 0 and len(gg) > 0:
            cloud = _depth_crop_to_cloud(
                validated["depth"],
                validated["depth_scale_to_m"],
                validated["K_crop"],
                self.args.collision_min_depth,
                self.args.collision_max_depth,
                self.args.max_collision_points,
            )
            if cloud.shape[0] < int(self.args.min_collision_points):
                raise ValueError(
                    "Not enough valid cropped-depth points for collision filtering: "
                    f"{cloud.shape[0]} < {self.args.min_collision_points}. "
                    "Set --collision-thresh 0 only if collision filtering is "
                    "intentionally disabled."
                )
            detector = self._collision_detector_cls(
                cloud.reshape(-1, 3),
                voxel_size=float(self.args.collision_voxel_size),
            )
            collision = detector.detect(
                gg,
                approach_dist=float(self.args.approach_dist),
                collision_thresh=float(self.args.collision_thresh),
            )
            if torch.is_tensor(collision):
                collision = collision.detach().cpu().numpy()
            collision = np.asarray(collision, dtype=bool).reshape(-1)
            if collision.shape[0] != len(gg):
                raise RuntimeError(
                    "Collision detector returned a mask with an unexpected length."
                )
            collision_removed = int(collision.sum())
            gg = gg[~collision]

        after_collision = len(gg)
        if len(gg):
            # Follow the AUBO demo's score ordering and grasp NMS stage.
            gg = gg.sort_by_score().nms().sort_by_score()
            gg = gg[: int(self.args.top_k)]
        timings["postprocess_ms"] = (time.perf_counter() - start) * 1000.0 - timings["model_ms"]

        grasp_array = (
            np.asarray(gg.grasp_group_array, dtype=np.float32).reshape(-1, 17)
            if len(gg)
            else np.empty((0, 17), dtype=np.float32)
        )
        if grasp_array.shape[0]:
            scores = grasp_array[:, 0].copy()
            widths = grasp_array[:, 1].copy()
            rotations = grasp_array[:, 4:13].reshape(-1, 3, 3).copy()
            translations = grasp_array[:, 13:16].copy()
        else:
            empty = _empty_grasp_arrays()
            scores = empty["scores"]
            widths = empty["widths_m"]
            rotations = empty["rotation_matrices_camera"]
            translations = empty["translations_camera_m"]

        response_arrays: dict[str, np.ndarray] = {
            "grasps_camera": grasp_array,
            "scores": scores.astype(np.float32, copy=False),
            "widths_m": widths.astype(np.float32, copy=False),
            "rotation_matrices_camera": rotations.astype(np.float32, copy=False),
            "translations_camera_m": translations.astype(np.float32, copy=False),
        }

        T_base_from_camera = validated["T_base_from_camera"]
        if T_base_from_camera is not None:
            if grasp_array.shape[0]:
                R_base_from_camera = T_base_from_camera[:3, :3]
                t_base_from_camera = T_base_from_camera[:3, 3]
                rotations_base = np.einsum(
                    "ij,njk->nik", R_base_from_camera, rotations
                )
                translations_base = (
                    np.einsum("ij,nj->ni", R_base_from_camera, translations)
                    + t_base_from_camera[None, :]
                )
                grasps_base = grasp_array.copy()
                grasps_base[:, 4:13] = rotations_base.reshape(-1, 9)
                grasps_base[:, 13:16] = translations_base
            else:
                rotations_base = np.empty((0, 3, 3), dtype=np.float32)
                translations_base = np.empty((0, 3), dtype=np.float32)
                grasps_base = np.empty((0, 17), dtype=np.float32)
            response_arrays.update({
                "grasps_base": grasps_base.astype(np.float32, copy=False),
                "rotation_matrices_base": rotations_base.astype(np.float32, copy=False),
                "translations_base_m": translations_base.astype(np.float32, copy=False),
            })

        response_metadata = {
            "status": "ok",
            "request_id": request_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "coordinate_frame": str(metadata.get("camera_frame_id", "camera_optical_frame")),
            "base_frame_id": (
                str(metadata.get("base_frame_id", "base_link"))
                if T_base_from_camera is not None else None
            ),
            "pose_depth_mode": self.args.pose_depth_mode,
            "camera_pose_source": validated["pose_source"],
            "depth_role": "sensor depth is used only for model-free collision filtering; grasp geometry is predicted from RGB by DPT",
            "raw_candidates": int(raw_count),
            "after_validity_filter": int(after_validity),
            "collision_removed": int(collision_removed),
            "after_collision_filter": int(after_collision),
            "returned_top_k": int(grasp_array.shape[0]),
            "top_k": int(self.args.top_k),
            "timings_ms": timings,
        }
        response_arrays["metadata_json"] = np.frombuffer(
            json.dumps(response_metadata, separators=(",", ":"), allow_nan=False).encode("utf-8"),
            dtype=np.uint8,
        )
        return response_arrays

    @staticmethod
    def error_response(exc: Exception) -> dict[str, np.ndarray]:
        metadata = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        result = _empty_grasp_arrays()
        result["metadata_json"] = np.frombuffer(
            json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
            dtype=np.uint8,
        )
        return result


class _RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        service: RayRoPEInferenceService = self.server.service  # type: ignore[attr-defined]
        max_bytes: int = self.server.max_request_bytes  # type: ignore[attr-defined]
        try:
            payload = recv_frame(self.request, max_bytes=max_bytes)
            arrays = decode_npz(payload)
            metadata = _metadata_json(arrays)
            expected_token = service.args.auth_token
            if expected_token and metadata.get("auth_token") != expected_token:
                raise PermissionError("LAN demo authentication failed.")
            response = service.infer(arrays)
        except Exception as exc:
            LOGGER.exception("Request failed")
            response = service.error_response(exc)
        try:
            send_frame(self.request, encode_npz(response))
        except Exception:
            LOGGER.exception("Failed to send response to client")


class _ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address, handler, service, max_request_bytes):
        super().__init__(address, handler)
        self.service = service
        self.max_request_bytes = int(max_request_bytes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EconomicGrasp Point RayRoPE LAN inference server."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("ECONOMICGRASP_RAYROPE_CKPT", ""),
        help="Path to the verified 50/50 Point RayRoPE checkpoint (or set ECONOMICGRASP_RAYROPE_CKPT).",
    )
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--pose-depth-mode", choices=("global_film",), default="global_film")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--collision-thresh", type=float, default=0.05)
    parser.add_argument("--collision-voxel-size", type=float, default=0.01)
    parser.add_argument("--approach-dist", type=float, default=0.05)
    parser.add_argument("--collision-min-depth", type=float, default=0.05)
    parser.add_argument("--collision-max-depth", type=float, default=2.0)
    parser.add_argument("--max-collision-points", type=int, default=0)
    parser.add_argument("--min-collision-points", type=int, default=32)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=1.0)
    parser.add_argument("--bin-num", type=int, default=256)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--num-angle", type=int, default=12)
    parser.add_argument("--num-depth", type=int, default=4)
    parser.add_argument("--m-points", type=int, default=1024)
    parser.add_argument("--graspness-threshold", type=float, default=0.1)
    parser.add_argument("--grasp-max-width", type=float, default=0.1)
    parser.add_argument("--max-request-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("ECONOMICGRASP_LAN_TOKEN", ""),
        help="Optional shared token checked against metadata.auth_token.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.collision_voxel_size <= 0:
        raise ValueError("--collision-voxel-size must be positive.")
    if args.collision_thresh < 0:
        raise ValueError("--collision-thresh cannot be negative.")

    service = RayRoPEInferenceService(args)
    server = _ThreadedTCPServer(
        (args.host, args.port),
        _RequestHandler,
        service=service,
        max_request_bytes=args.max_request_bytes,
    )
    LOGGER.info(
        "Listening on %s:%d; top_k=%d; collision_thresh=%.4f",
        args.host, args.port, args.top_k, args.collision_thresh,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
