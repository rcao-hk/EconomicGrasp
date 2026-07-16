#!/usr/bin/env python3
"""Mine exact GraspNet evaluator supervision for a CVA joint utility scorer.

This program freezes an existing first-generation CVA checkpoint, enumerates an
on-policy top-K-view / multi-angle / all-depth candidate lattice, extracts the
CVA hidden feature, and labels every retained candidate with the official
GraspNet CAD collision + DexNet force-closure evaluator.

The output is an offline cache consumed by ``train_cva_joint_utility_ddp.py``.
GT CAD models and poses are used only while mining training/validation labels;
they are not used by the deployed scorer.

Required repository patch
-------------------------
``models/kview_query_transformer.py`` must expose
``end_points['cva_angle_feature']`` with shape [B,Q,A,C] after the CVA angle
self-attention and before the legacy branch heads.  Apply the included patch.

The custom ``--ju_*`` options are consumed before importing the repository's
``utils.arguments`` parser, so the existing project parser does not need to be
edited.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _consume_custom_args() -> argparse.Namespace:
    # ``ProcessPoolExecutor(..., mp_context=spawn)`` re-imports this module in
    # evaluator workers.  The parent process has already removed the custom
    # ``--ju_*`` arguments from ``sys.argv`` before those workers are spawned,
    # so a required argparse option would fail during child import.
    #
    # Persist the parsed custom namespace in an environment variable.  Spawned
    # evaluator/DataLoader workers recover the same values from that snapshot,
    # while the top-level process can still override them with explicit CLI
    # options.
    env_key = "CVA_JOINT_UTILITY_ARGS_JSON"
    env_defaults = {}
    raw_env = os.environ.get(env_key, "").strip()
    if raw_env:
        try:
            loaded = json.loads(raw_env)
            if isinstance(loaded, dict):
                env_defaults = loaded
        except Exception as exc:
            raise RuntimeError(f"Invalid {env_key}: {exc}") from exc

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ju_cache_dir", default="")
    p.add_argument("--ju_dataset_split", default="train")
    p.add_argument("--ju_scene_ids", default="", help="Absolute scene ids, comma-separated.")
    p.add_argument("--ju_anno_ids", default="", help="Optional exact annotation ids.")
    p.add_argument("--ju_sample_interval", type=float, default=0.1)
    p.add_argument("--ju_max_samples", type=int, default=-1)
    p.add_argument("--ju_top_views", type=int, default=4)
    p.add_argument("--ju_top_centers", type=int, default=24)
    p.add_argument("--ju_random_centers", type=int, default=8)
    p.add_argument("--ju_top_angles", type=int, default=6)
    p.add_argument("--ju_random_angles", type=int, default=2)
    p.add_argument("--ju_all_angles", type=int, default=0)
    p.add_argument("--ju_eval_chunk", type=int, default=512)
    p.add_argument("--ju_eval_workers", type=int, default=4,
                   help="CPU evaluator processes per GPU producer. 0 keeps serial evaluation.")
    p.add_argument("--ju_max_pending", type=int, default=8,
                   help="Maximum frames queued for asynchronous evaluator workers.")
    p.add_argument("--ju_eval_threads", type=int, default=1,
                   help="OMP/MKL threads used inside each evaluator process.")
    p.add_argument("--ju_prefetch_factor", type=int, default=2)
    p.add_argument("--ju_persistent_workers", type=int, default=1)
    p.add_argument("--ju_pin_memory", type=int, default=1)
    p.add_argument("--ju_empty_cache_each_view", type=int, default=0,
                   help="Call torch.cuda.empty_cache after forced-view passes; normally keep disabled.")
    p.add_argument("--ju_tf32", type=int, default=1)
    p.add_argument("--ju_skip_force_closure", type=int, default=0)
    p.add_argument(
        "--ju_fc_mode",
        choices=["official", "reuse_contacts", "reuse_contacts_binary"],
        default="reuse_contacts",
        help=(
            "Force-closure scoring implementation. 'official' calls the stock "
            "get_grasp_score and repeatedly closes the fingers. 'reuse_contacts' "
            "closes once and exactly reuses the contacts for the official descending "
            "friction sweep. 'reuse_contacts_binary' also uses a monotonic binary "
            "search over the six official friction thresholds."
        ),
    )
    p.add_argument(
        "--ju_fc_verify_n", type=int, default=0,
        help=(
            "Per frame, compare the first N non-colliding optimized force-closure "
            "scores against stock get_grasp_score. Use 8-32 for a smoke test and 0 "
            "for production."
        ),
    )
    p.add_argument("--ju_feature_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--ju_compress", type=int, default=0)
    p.add_argument("--ju_overwrite", type=int, default=0)
    p.add_argument("--ju_strict", type=int, default=1)
    p.add_argument("--ju_seed", type=int, default=0)
    p.add_argument("--ju_worker_tag", default="worker")
    p.add_argument("--ju_override_key", default="oracle_view_inds_override")
    if env_defaults:
        known_dests = {action.dest for action in p._actions}
        p.set_defaults(**{k: v for k, v in env_defaults.items() if k in known_dests})

    custom, rest = p.parse_known_args(sys.argv[1:])

    # Make the custom configuration available to future spawn children before
    # stripping the arguments for the repository-wide parser.
    os.environ[env_key] = json.dumps(vars(custom), sort_keys=True)
    sys.argv = [sys.argv[0], *rest]
    return custom


JU = _consume_custom_args()


class _StopRequested(KeyboardInterrupt):
    """Raised in the producer process when SIGINT/SIGTERM is received."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(f"stop requested by signal {self.signum}")


def _install_main_signal_handlers():
    """Convert SIGINT/SIGTERM into a catchable stop request in the producer."""
    previous = {}

    def _handler(signum, _frame):
        name = signal.Signals(signum).name if signum in (signal.SIGINT, signal.SIGTERM) else str(signum)
        print(f"\n[STOP] received {name}; cancelling evaluator workers...", flush=True)
        raise _StopRequested(int(signum))

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, _handler)
    return previous


def _restore_signal_handlers(previous) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


def _shutdown_loader_iterator(loader_iter) -> None:
    """Stop persistent DataLoader workers without waiting for interpreter exit."""
    if loader_iter is None:
        return
    shutdown = getattr(loader_iter, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as exc:
            print(f"[STOP][WARN] DataLoader worker shutdown failed: {exc}", flush=True)


def _abort_process_pool(executor, pending, grace_sec: float = 2.0) -> None:
    """Cancel queued work and forcibly terminate running spawn workers.

    ProcessPoolExecutor.shutdown(wait=False) does not terminate tasks that are
    already running.  Capture the worker Process objects first, then terminate
    and finally kill any process that ignores SIGTERM.
    """
    if executor is None:
        pending.clear()
        return

    processes = []
    try:
        proc_map = getattr(executor, "_processes", None)
        if proc_map:
            processes = list(proc_map.values())
    except Exception:
        processes = []

    for future in list(pending):
        try:
            future.cancel()
        except Exception:
            pass
    pending.clear()

    for proc in processes:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:  # Python < 3.9 fallback
        executor.shutdown(wait=False)
    except Exception as exc:
        print(f"[STOP][WARN] executor shutdown failed: {exc}", flush=True)

    deadline = time.monotonic() + max(0.0, float(grace_sec))
    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.join(timeout=remaining)
        except Exception:
            pass

    survivors = []
    for proc in processes:
        try:
            if proc.is_alive():
                survivors.append(proc)
                if hasattr(proc, "kill"):
                    proc.kill()
                else:
                    os.kill(proc.pid, signal.SIGKILL)
        except Exception:
            pass

    for proc in survivors:
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass

    if processes:
        print(
            f"[STOP] evaluator pool terminated: workers={len(processes)} "
            f"forced_kill={len(survivors)}",
            flush=True,
        )


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from graspnetAPI import GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    collision_detection,
    compute_closest_points,
    create_table_points,
    get_grasp_score,
    transform_points,
    voxel_sample_points,
)
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
from graspnetAPI.utils.dexnet.grasping.quality import PointGraspMetrics3D

from utils.arguments import cfgs
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn
from utils.label_generation import batch_viewpoint_params_to_matrix
from models.economicgrasp_bip3d import economicgrasp_dpt


THRESHOLDS = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float64)


def _csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _save_json(payload: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(dict(payload)), f, indent=2)
    os.replace(tmp, path)


def _move_batch_to_device(batch: MutableMapping[str, Any], device: torch.device, non_blocking: bool = False) -> None:
    for key in list(batch.keys()):
        value = batch[key]
        if "list" in key:
            for i in range(len(value)):
                for j in range(len(value[i])):
                    value[i][j] = value[i][j].to(device, non_blocking=non_blocking)
        elif "graph" in key:
            for i in range(len(value)):
                value[i] = value[i].to(device, non_blocking=non_blocking)
        elif torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=non_blocking)


def _capture_rng() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "cpu": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _copy_inputs(batch: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(batch)



class DeterministicIndexDataset(Dataset):
    """Index subset with per-dataset-index NumPy/Python RNG seeding."""
    def __init__(self, dataset, indices: Sequence[int], seed: int) -> None:
        self.dataset = dataset
        self.indices = [int(x) for x in indices]
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int):
        dataset_index = self.indices[local_index]
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            np.random.seed((self.seed + dataset_index * 1000003) % (2**32 - 1))
            random.seed(self.seed + dataset_index * 1000003)
            return self.dataset[dataset_index]
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

def _build_dataset():
    cls = GraspNetMultiDataset if cfgs.multi_modal else GraspNetDataset
    return cls(
        cfgs.dataset_root,
        split=str(JU.ju_dataset_split),
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=bool(getattr(cfgs, "use_gt_depth", False)),
        use_fuse_depth=bool(getattr(cfgs, "use_fuse_depth", False)),
        min_depth=float(cfgs.min_depth),
        max_depth=float(cfgs.max_depth),
        bin_num=int(cfgs.bin_num),
        extend_angle=bool(getattr(cfgs, "extend_angle", False)),
    )


def _select_indices(dataset) -> List[int]:
    scene_filter = set(_csv_ints(JU.ju_scene_ids))
    anno_filter = set(_csv_ints(JU.ju_anno_ids))
    interval = float(JU.ju_sample_interval)
    if interval <= 0:
        raise ValueError("--ju_sample_interval must be > 0")
    stride = 1 if interval >= 1.0 else max(1, int(round(1.0 / interval)))
    selected: List[int] = []
    for index, scene_name in enumerate(dataset.scene_list()):
        scene_id = int(str(scene_name).split("_")[-1])
        anno_id = index % 256
        if scene_filter and scene_id not in scene_filter:
            continue
        if anno_filter:
            if anno_id not in anno_filter:
                continue
        elif anno_id % stride != 0:
            continue
        selected.append(index)
        if int(JU.ju_max_samples) > 0 and len(selected) >= int(JU.ju_max_samples):
            break
    if not selected:
        raise RuntimeError("No samples selected; check --ju_scene_ids/--ju_anno_ids")
    return selected


def _load_model(device: torch.device) -> torch.nn.Module:
    if not cfgs.multi_modal:
        raise ValueError("Joint utility mining targets economicgrasp_dpt; pass --multi_modal")
    if int(getattr(cfgs, "kview_k", 1)) != 1:
        raise ValueError("This miner targets first-generation CVA (kview_k=1), not RotNet-CVA")
    net = economicgrasp_dpt(
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        is_training=False,
        use_obs_depth=bool(getattr(cfgs, "use_obs_depth", False)),
        vis_dir=getattr(cfgs, "vis_dir", None),
        vis_every=int(getattr(cfgs, "vis_every", 1000)),
    )
    checkpoint = torch.load(cfgs.checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        net.load_state_dict(state)
    except RuntimeError:
        if state and all(str(k).startswith("module.") for k in state):
            net.load_state_dict({str(k)[7:]: v for k, v in state.items()})
        else:
            raise
    return net.to(device).eval()


def _canonical_view_score(end_points: Mapping[str, Any]) -> torch.Tensor:
    score = end_points["view_score"]
    q = int(end_points["xyz_graspable"].shape[1])
    if score.dim() != 3:
        raise ValueError(f"view_score must be 3D, got {tuple(score.shape)}")
    if score.shape[1] == q:
        return score
    if score.shape[2] == q:
        return score.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot canonicalize view_score={tuple(score.shape)} for Q={q}")


def _expected_quality(score_logits: torch.Tensor) -> torch.Tensor:
    # [6,Q,A] -> [Q,A]
    bins = torch.linspace(0.0, 1.0, score_logits.shape[0], device=score_logits.device)
    return (F.softmax(score_logits.float(), dim=0) * bins[:, None, None]).sum(dim=0)


def _assert_same_centers(base: Mapping[str, Any], other: Mapping[str, Any], tag: str) -> None:
    if torch.is_tensor(base.get("token_sel_idx")) and torch.is_tensor(other.get("token_sel_idx")):
        if not torch.equal(base["token_sel_idx"], other["token_sel_idx"]):
            raise RuntimeError(f"{tag} changed token_sel_idx")
    if not torch.allclose(base["xyz_graspable"], other["xyz_graspable"], atol=1e-7, rtol=0.0):
        raise RuntimeError(f"{tag} changed xyz_graspable")


def _run_forced_view(
    net: torch.nn.Module,
    batch: Mapping[str, Any],
    rng: Mapping[str, Any],
    forced_view: torch.Tensor,
    base: Mapping[str, Any],
    tag: str,
) -> Dict[str, Any]:
    _restore_rng(rng)
    inp = _copy_inputs(batch)
    inp[str(JU.ju_override_key)] = forced_view
    out = net(inp)
    _assert_same_centers(base, out, tag)
    used = out.get("grasp_top_view_inds")
    if not torch.is_tensor(used):
        raise KeyError("Forced pass did not expose grasp_top_view_inds")
    ratio = float(used.long().eq(forced_view.long()).float().mean().item())
    if ratio < 0.999999:
        msg = f"Forced view override not respected for {tag}: ratio={ratio:.6f}"
        if bool(JU.ju_strict):
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")
    return out


@dataclass
class EvalResult:
    assigned_obj: np.ndarray
    collision_or_empty: np.ndarray
    empty: np.ndarray
    pure_collision: np.ndarray
    friction: np.ndarray
    stats: Dict[str, Any]


class RawCandidateEvaluator:
    """Official CAD collision plus evaluator-equivalent force-closure labels.

    The stock ``get_grasp_score`` calls ``grasp.close_fingers`` again for every
    friction threshold. Contacts do not depend on the friction coefficient, so
    this repeats the dominant SDF ray-casting work up to six times per grasp.

    ``reuse_contacts`` performs exactly one ``close_fingers`` call and then
    follows the same descending threshold logic while passing the cached
    contacts to ``PointGraspMetrics3D.grasp_quality``. The resulting labels are
    evaluator-equivalent, subject to the deterministic DexNet contact routine.
    """

    def __init__(
        self,
        root: str,
        camera: str,
        split: str,
        chunk: int,
        skip_fc: bool,
        fc_mode: str = "reuse_contacts",
        fc_verify_n: int = 0,
        strict: bool = True,
    ) -> None:
        self.eval = GraspNetEval(root, camera, split=split)
        self.config = get_config()
        self.chunk = max(1, int(chunk))
        self.skip_fc = bool(skip_fc)
        self.fc_mode = str(fc_mode)
        self.fc_verify_n = max(0, int(fc_verify_n))
        self.strict = bool(strict)
        if self.fc_mode not in {"official", "reuse_contacts", "reuse_contacts_binary"}:
            raise ValueError(f"Unknown fc_mode={self.fc_mode}")
        self.table = create_table_points(
            1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008
        )
        self.scene_cache: Dict[int, Tuple[List[np.ndarray], List[Any]]] = {}
        # Keep the official order for evaluator-equivalent output.
        self.fc_list = np.asarray([1.2, 1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)
        self.fc_ascending = self.fc_list[::-1].copy()
        self.fc_configs: Dict[float, Any] = {}
        if not self.skip_fc:
            for mu in self.fc_list:
                key = round(float(mu), 2)
                self.config["metrics"]["force_closure"]["friction_coef"] = key
                self.fc_configs[key] = GraspQualityConfigFactory.create_config(
                    self.config["metrics"]["force_closure"]
                )

    def _scene_models(self, scene_id: int) -> Tuple[List[np.ndarray], List[Any]]:
        if scene_id not in self.scene_cache:
            models, dexmodels, _ = self.eval.get_scene_models(scene_id, ann_id=0)
            self.scene_cache[scene_id] = (
                [voxel_sample_points(x, 0.008) for x in models],
                dexmodels,
            )
        return self.scene_cache[scene_id]

    def _quality_with_contacts(
        self,
        grasp: Any,
        obj: Any,
        contacts: Sequence[Any],
        mu: float,
    ) -> bool:
        key = round(float(mu), 2)
        value = PointGraspMetrics3D.grasp_quality(
            grasp,
            obj,
            self.fc_configs[key],
            contacts=contacts,
        )
        return bool(value)

    def _score_reuse_contacts(
        self,
        grasp: Any,
        obj: Any,
        binary: bool,
    ) -> Tuple[float, int, bool, bool]:
        """Return score, number of quality calls, contacts_found, binary_fallback."""
        # Contact acquisition is friction-independent and is the expensive part.
        base_cfg = self.fc_configs[round(float(self.fc_list[0]), 2)]
        found, contacts = grasp.close_fingers(
            obj,
            check_approach=bool(getattr(base_cfg, "check_approach", False)),
            vis=False,
        )
        if not found:
            return -1.0, 0, False, False

        # Parallel-jaw close_fingers normally returns two contacts. For unusual
        # contact cardinalities, preserve the official descending sweep rather
        # than assuming monotonicity of a QP fallback.
        use_binary = bool(binary and len(contacts) == 2)
        calls = 0
        if use_binary:
            lo, hi = 0, len(self.fc_ascending)
            while lo < hi:
                mid = (lo + hi) // 2
                calls += 1
                if self._quality_with_contacts(grasp, obj, contacts, float(self.fc_ascending[mid])):
                    hi = mid
                else:
                    lo = mid + 1
            if lo >= len(self.fc_ascending):
                return -1.0, calls, True, False
            return round(float(self.fc_ascending[lo]), 2), calls, True, False

        tmp, is_force_closure = False, False
        quality = -1.0
        for ind, value_fc in enumerate(self.fc_list):
            value_fc = round(float(value_fc), 2)
            tmp = is_force_closure
            calls += 1
            is_force_closure = self._quality_with_contacts(grasp, obj, contacts, value_fc)
            if tmp and not is_force_closure:
                quality = round(float(self.fc_list[ind - 1]), 2)
                break
            if is_force_closure and value_fc == round(float(self.fc_list[-1]), 2):
                quality = value_fc
                break
            if value_fc == round(float(self.fc_list[0]), 2) and not is_force_closure:
                break
        return float(quality), calls, True, bool(binary and not use_binary)

    def _score_grasp(
        self,
        grasp: Any,
        obj: Any,
    ) -> Tuple[float, int, bool, bool]:
        if self.fc_mode == "official":
            return (
                float(get_grasp_score(grasp, obj, self.fc_list, self.fc_configs)),
                -1,
                True,
                False,
            )
        return self._score_reuse_contacts(
            grasp,
            obj,
            binary=(self.fc_mode == "reuse_contacts_binary"),
        )

    def evaluate(self, scene_id: int, anno_id: int, grasps: np.ndarray) -> EvalResult:
        n = len(grasps)
        empty_stats = {
            "collision_sec": 0.0,
            "force_closure_sec": 0.0,
            "fc_candidates": 0,
            "fc_quality_calls": 0,
            "fc_contacts_not_found": 0,
            "fc_binary_fallbacks": 0,
            "fc_verify_count": 0,
            "fc_verify_mismatches": 0,
            "fc_mode": self.fc_mode,
        }
        if n == 0:
            zi = np.zeros(0, dtype=np.int64)
            zb = np.zeros(0, dtype=bool)
            zf = np.zeros(0, dtype=np.float32)
            return EvalResult(zi, zb, zb, zb, zf, empty_stats)

        models_obj, dexmodels = self._scene_models(scene_id)
        _, poses, camera_pose, align_mat = self.eval.get_model_poses(scene_id, anno_id)
        models_cam = [transform_points(m, poses[i]) for i, m in enumerate(models_obj)]
        scene = np.concatenate(models_cam, axis=0)
        seg = np.concatenate(
            [np.full(len(m), i, dtype=np.int64) for i, m in enumerate(models_cam)], axis=0
        )
        nearest = compute_closest_points(grasps[:, 13:16], scene)
        assigned = seg[nearest]
        table_cam = transform_points(self.table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        scene_with_table = np.concatenate([scene, table_cam], axis=0)

        collision = np.zeros(n, dtype=bool)
        empty = np.zeros(n, dtype=bool)
        friction = np.full(n, -1.0, dtype=np.float32)
        collision_sec = 0.0
        fc_sec = 0.0
        fc_candidates = 0
        fc_quality_calls = 0
        contacts_not_found = 0
        binary_fallbacks = 0
        verify_count = 0
        verify_mismatches = 0

        for obj_idx in range(len(models_cam)):
            obj_rows = np.flatnonzero(assigned == obj_idx)
            for start in range(0, len(obj_rows), self.chunk):
                ids = obj_rows[start : start + self.chunk]
                if len(ids) == 0:
                    continue
                chunk_grasps = grasps[ids]
                t0 = time.perf_counter()
                coll_l, empty_l, dex_l = collision_detection(
                    [chunk_grasps],
                    [models_cam[obj_idx]],
                    [dexmodels[obj_idx]],
                    [poses[obj_idx]],
                    scene_with_table,
                    outlier=0.05,
                    return_dexgrasps=True,
                )
                collision_sec += time.perf_counter() - t0
                c = np.asarray(coll_l[0], dtype=bool)
                e = np.asarray(empty_l[0], dtype=bool)
                collision[ids] = c
                empty[ids] = e
                if self.skip_fc:
                    friction[ids] = np.where(c, -1.0, 1.2).astype(np.float32)
                    continue

                dexgrasps = dex_l[0]
                t1 = time.perf_counter()
                for local_i, global_i in enumerate(ids):
                    if c[local_i] or dexgrasps[local_i] is None:
                        friction[global_i] = -1.0
                        continue
                    fc_candidates += 1
                    score, calls, contacts_found, binary_fallback = self._score_grasp(
                        dexgrasps[local_i], dexmodels[obj_idx]
                    )
                    friction[global_i] = float(score)
                    if calls > 0:
                        fc_quality_calls += int(calls)
                    contacts_not_found += int(not contacts_found)
                    binary_fallbacks += int(binary_fallback)

                    if (
                        self.fc_mode != "official"
                        and verify_count < self.fc_verify_n
                    ):
                        official = float(
                            get_grasp_score(
                                dexgrasps[local_i],
                                dexmodels[obj_idx],
                                self.fc_list,
                                self.fc_configs,
                            )
                        )
                        verify_count += 1
                        if not np.isclose(official, score, atol=1e-6, rtol=0.0):
                            verify_mismatches += 1
                            msg = (
                                f"FC verification mismatch scene={scene_id} anno={anno_id} "
                                f"obj={obj_idx} row={int(global_i)} optimized={score} official={official}"
                            )
                            if self.strict:
                                raise RuntimeError(msg)
                            print(f"[FC][WARN] {msg}", flush=True)
                fc_sec += time.perf_counter() - t1

        stats = {
            "collision_sec": float(collision_sec),
            "force_closure_sec": float(fc_sec),
            "fc_candidates": int(fc_candidates),
            "fc_quality_calls": int(fc_quality_calls),
            "fc_contacts_not_found": int(contacts_not_found),
            "fc_binary_fallbacks": int(binary_fallbacks),
            "fc_verify_count": int(verify_count),
            "fc_verify_mismatches": int(verify_mismatches),
            "fc_mode": self.fc_mode,
        }
        return EvalResult(
            assigned_obj=assigned,
            collision_or_empty=collision,
            empty=empty,
            pure_collision=collision & (~empty),
            friction=friction,
            stats=stats,
        )


_EVAL_WORKER: Optional[RawCandidateEvaluator] = None


def _eval_worker_init(
    root: str,
    camera: str,
    split: str,
    chunk: int,
    skip_fc: bool,
    threads: int,
    fc_mode: str,
    fc_verify_n: int,
    strict: bool,
) -> None:
    """Initialize one CPU-only exact-evaluator process.

    SIGINT is handled only by the producer.  The producer then explicitly
    terminates this worker pool, avoiding multiple workers raising independent
    KeyboardInterrupt exceptions and leaving orphaned spawn processes.
    """
    global _EVAL_WORKER
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    threads = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _EVAL_WORKER = RawCandidateEvaluator(
        root, camera, split, chunk, skip_fc,
        fc_mode=fc_mode, fc_verify_n=fc_verify_n, strict=strict,
    )


def _eval_worker_task(scene_id: int, anno_id: int, grasps: np.ndarray) -> Dict[str, Any]:
    if _EVAL_WORKER is None:
        raise RuntimeError("Evaluator worker was not initialized")
    started = time.perf_counter()
    result = _EVAL_WORKER.evaluate(int(scene_id), int(anno_id), np.asarray(grasps, dtype=np.float32))
    return {
        "assigned_obj": result.assigned_obj,
        "collision_or_empty": result.collision_or_empty,
        "empty": result.empty,
        "pure_collision": result.pure_collision,
        "friction": result.friction,
        "eval_sec": time.perf_counter() - started,
        "worker_pid": os.getpid(),
        **result.stats,
    }


@dataclass
class PendingFrame:
    scene_id: int
    anno_id: int
    dataset_idx: int
    out_path: str
    arrays: Dict[str, np.ndarray]
    num_centers: int
    submitted_at: float
    gpu_sec: float


def _result_from_payload(payload: Mapping[str, Any]) -> EvalResult:
    return EvalResult(
        assigned_obj=np.asarray(payload["assigned_obj"], dtype=np.int64),
        collision_or_empty=np.asarray(payload["collision_or_empty"], dtype=bool),
        empty=np.asarray(payload["empty"], dtype=bool),
        pure_collision=np.asarray(payload["pure_collision"], dtype=bool),
        friction=np.asarray(payload["friction"], dtype=np.float32),
        stats={},
    )


def _finalize_pending(
    item: PendingFrame,
    payload: Mapping[str, Any],
    summaries: List[Dict[str, Any]],
    global_start: float,
) -> None:
    result = _result_from_payload(payload)
    n = len(result.friction)
    arrays: Dict[str, np.ndarray] = {
        **item.arrays,
        "assigned_obj": result.assigned_obj.astype(np.int16),
        "collision_or_empty": result.collision_or_empty.astype(np.uint8),
        "pure_collision": result.pure_collision.astype(np.uint8),
        "empty": result.empty.astype(np.uint8),
        "friction": result.friction.astype(np.float16),
        "scene_id": np.asarray([item.scene_id], dtype=np.int16),
        "anno_id": np.asarray([item.anno_id], dtype=np.int16),
        "dataset_idx": np.asarray([item.dataset_idx], dtype=np.int32),
    }
    save_started = time.perf_counter()
    _save_cache(item.out_path, arrays)
    save_sec = time.perf_counter() - save_started

    utility = ((result.friction[:, None] > 0.0) & (
        result.friction[:, None] <= THRESHOLDS[None]
    )).mean(axis=1)
    summary = {
        "scene_id": item.scene_id,
        "anno_id": item.anno_id,
        "num_candidates": n,
        "num_centers": int(item.num_centers),
        "valid_ratio": float(np.mean(utility > 0.0)),
        "safe08_ratio": float(np.mean(utility >= 0.5)),
        "pure_collision_ratio": float(result.pure_collision.mean()),
        "empty_ratio": float(result.empty.mean()),
        "gpu_candidate_sec": float(item.gpu_sec),
        "eval_sec": float(payload.get("eval_sec", float("nan"))),
        "queue_latency_sec": float(time.perf_counter() - item.submitted_at),
        "save_sec": float(save_sec),
        "eval_worker_pid": int(payload.get("worker_pid", -1)),
        "collision_sec": float(payload.get("collision_sec", float("nan"))),
        "force_closure_sec": float(payload.get("force_closure_sec", float("nan"))),
        "fc_candidates": int(payload.get("fc_candidates", 0)),
        "fc_quality_calls": int(payload.get("fc_quality_calls", 0)),
        "fc_contacts_not_found": int(payload.get("fc_contacts_not_found", 0)),
        "fc_binary_fallbacks": int(payload.get("fc_binary_fallbacks", 0)),
        "fc_verify_count": int(payload.get("fc_verify_count", 0)),
        "fc_verify_mismatches": int(payload.get("fc_verify_mismatches", 0)),
        "fc_mode": str(payload.get("fc_mode", "unknown")),
        "cache_path": item.out_path,
    }
    summaries.append(summary)
    elapsed = time.time() - global_start
    print(
        f"[MINE] {item.scene_id:04d}/{item.anno_id:04d} N={n} "
        f"valid={summary['valid_ratio']:.3f} safe08={summary['safe08_ratio']:.3f} "
        f"gpu={summary['gpu_candidate_sec']:.1f}s eval={summary['eval_sec']:.1f}s "
        f"(col={summary['collision_sec']:.1f}s fc={summary['force_closure_sec']:.1f}s "
        f"fcN={summary['fc_candidates']}) "
        f"queue={summary['queue_latency_sec']:.1f}s elapsed={elapsed/60.0:.1f}m",
        flush=True,
    )

def _choose_centers(base: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    score_logits = base["grasp_score_pred_angle"][0].float()
    quality = _expected_quality(score_logits)
    center_score = quality.max(dim=-1).values
    q = center_score.numel()
    top_n = min(max(0, int(JU.ju_top_centers)), q)
    random_n = min(max(0, int(JU.ju_random_centers)), max(q - top_n, 0))
    top = torch.topk(center_score, k=top_n, largest=True, sorted=True).indices.cpu().numpy()
    remaining = np.setdiff1d(np.arange(q, dtype=np.int64), top, assume_unique=False)
    rand = rng.choice(remaining, size=random_n, replace=False) if random_n and remaining.size else np.zeros(0, dtype=np.int64)
    return np.concatenate([top.astype(np.int64), rand.astype(np.int64)])


def _choose_angles(
    expected: torch.Tensor,
    center_id: int,
    rng: np.random.Generator,
) -> Tuple[List[int], set]:
    a = int(expected.shape[-1])
    if bool(JU.ju_all_angles):
        ids = list(range(a))
        return ids, set(ids)
    top_n = min(max(1, int(JU.ju_top_angles)), a)
    top = torch.topk(expected[center_id], k=top_n, largest=True, sorted=True).indices.cpu().tolist()
    top_set = {int(x) for x in top}
    remaining = [x for x in range(a) if x not in top_set]
    random_n = min(max(0, int(JU.ju_random_angles)), len(remaining))
    rand = rng.choice(np.asarray(remaining, dtype=np.int64), size=random_n, replace=False).tolist() if random_n else []
    return [int(x) for x in top + rand], top_set


def _extract_view_candidates(
    end_points: Mapping[str, Any],
    base_view_score: torch.Tensor,
    center_ids: np.ndarray,
    view_rank: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    hidden = end_points.get("cva_angle_feature")
    if not torch.is_tensor(hidden):
        raise KeyError(
            "Model did not expose end_points['cva_angle_feature']. Apply "
            "patches/expose_cva_angle_feature.patch first."
        )
    if hidden.dim() != 4:
        raise ValueError(f"cva_angle_feature must be [B,Q,A,C], got {tuple(hidden.shape)}")

    hidden = hidden[0].float()  # [Q,A,C]
    score_logits = end_points["grasp_score_pred_angle"][0].float()  # [6,Q,A]
    depth_logits = end_points["grasp_depth_pred_angle"][0].float()  # [D+1,Q,A]
    width = end_points["grasp_width_pred_angle"][0, 0].float()      # [Q,A]
    collision = end_points.get("grasp_collision_pred_angle")
    collision = collision[0, 0].float() if torch.is_tensor(collision) else torch.zeros_like(width)
    centers = end_points["xyz_graspable"][0].float()
    view_xyz = end_points["grasp_top_view_xyz"][0].float()
    view_id = end_points["grasp_top_view_inds"][0].long()
    expected = _expected_quality(score_logits)
    d_physical = min(int(getattr(cfgs, "num_depth", depth_logits.shape[0] - 1)), depth_logits.shape[0] - 1)
    a_total = int(score_logits.shape[-1])

    rows: Dict[str, List[Any]] = {
        "angle_feature": [], "score_logits": [], "depth_logits": [],
        "width_raw": [], "legacy_collision_logit": [], "angle_id": [],
        "depth_id": [], "view_rank": [], "view_id": [], "view_score": [],
        "center_xyz": [], "view_xyz": [], "center_id": [],
        "legacy_quality": [], "legacy_depth_prob": [], "legacy_score": [],
        "angle_is_top": [], "grasp": [],
    }

    depth_prob = F.softmax(depth_logits, dim=0)
    for qid_np in center_ids:
        qid = int(qid_np)
        angles, top_set = _choose_angles(expected, qid, rng)
        vid = int(view_id[qid].item())
        original_view_score = float(base_view_score[0, qid, vid].item())
        for aid in angles:
            angle_rad = float(aid) * math.pi / float(a_total)
            rotation = batch_viewpoint_params_to_matrix(
                -view_xyz[qid : qid + 1],
                torch.tensor([angle_rad], device=centers.device, dtype=centers.dtype),
            ).reshape(9)
            width_raw = width[qid, aid]
            width_m = torch.clamp(
                1.2 * width_raw / 10.0,
                min=0.0,
                max=float(getattr(cfgs, "grasp_max_width", 0.1)),
            )
            for did in range(d_physical):
                quality = expected[qid, aid]
                dprob = depth_prob[did, qid, aid]
                legacy_score = quality * dprob
                depth_m = (float(did) + 1.0) * 0.01
                grasp = torch.cat(
                    [
                        legacy_score.reshape(1),
                        width_m.reshape(1),
                        torch.tensor([0.02, depth_m], device=centers.device, dtype=centers.dtype),
                        rotation,
                        centers[qid],
                        torch.tensor([-1.0], device=centers.device, dtype=centers.dtype),
                    ]
                )
                rows["angle_feature"].append(hidden[qid, aid])
                rows["score_logits"].append(score_logits[:, qid, aid])
                rows["depth_logits"].append(depth_logits[:, qid, aid])
                rows["width_raw"].append(width_raw)
                rows["legacy_collision_logit"].append(collision[qid, aid])
                rows["angle_id"].append(aid)
                rows["depth_id"].append(did)
                rows["view_rank"].append(view_rank)
                rows["view_id"].append(vid)
                rows["view_score"].append(original_view_score)
                rows["center_xyz"].append(centers[qid])
                rows["view_xyz"].append(view_xyz[qid])
                rows["center_id"].append(qid)
                rows["legacy_quality"].append(quality)
                rows["legacy_depth_prob"].append(dprob)
                rows["legacy_score"].append(legacy_score)
                rows["angle_is_top"].append(int(aid in top_set))
                rows["grasp"].append(grasp)

    feature_dtype = torch.float16 if JU.ju_feature_dtype == "float16" else torch.float32
    out: Dict[str, np.ndarray] = {
        "angle_feature": torch.stack(rows["angle_feature"]).to(feature_dtype).cpu().numpy(),
        "score_logits": torch.stack(rows["score_logits"]).to(torch.float16).cpu().numpy(),
        "depth_logits": torch.stack(rows["depth_logits"]).to(torch.float16).cpu().numpy(),
        "width_raw": torch.stack(rows["width_raw"]).to(torch.float16).cpu().numpy(),
        "legacy_collision_logit": torch.stack(rows["legacy_collision_logit"]).to(torch.float16).cpu().numpy(),
        "angle_id": np.asarray(rows["angle_id"], dtype=np.int16),
        "depth_id": np.asarray(rows["depth_id"], dtype=np.int8),
        "view_rank": np.asarray(rows["view_rank"], dtype=np.int8),
        "view_id": np.asarray(rows["view_id"], dtype=np.int16),
        "view_score": np.asarray(rows["view_score"], dtype=np.float16),
        "center_xyz": torch.stack(rows["center_xyz"]).to(torch.float16).cpu().numpy(),
        "view_xyz": torch.stack(rows["view_xyz"]).to(torch.float16).cpu().numpy(),
        "center_id": np.asarray(rows["center_id"], dtype=np.int16),
        "legacy_quality": torch.stack(rows["legacy_quality"]).to(torch.float16).cpu().numpy(),
        "legacy_depth_prob": torch.stack(rows["legacy_depth_prob"]).to(torch.float16).cpu().numpy(),
        "legacy_score": torch.stack(rows["legacy_score"]).to(torch.float16).cpu().numpy(),
        "angle_is_top": np.asarray(rows["angle_is_top"], dtype=np.uint8),
        "grasp": torch.stack(rows["grasp"]).float().cpu().numpy(),
    }
    return out


def _concat_candidate_dicts(parts: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not parts:
        raise ValueError("No candidate parts")
    keys = parts[0].keys()
    return {key: np.concatenate([p[key] for p in parts], axis=0) for key in keys}


def _save_cache(path: str, arrays: Mapping[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    saver = np.savez_compressed if bool(JU.ju_compress) else np.savez
    # np.savez appends .npz when the provided name lacks it. Use an explicit
    # file handle so the atomic temporary path remains exact.
    with open(tmp, "wb") as f:
        saver(f, **arrays)
    os.replace(tmp, path)


def main() -> None:
    if not str(JU.ju_cache_dir).strip():
        raise SystemExit(
            "Missing required --ju_cache_dir. When launching through the provided "
            "Bash script, set CACHE_DIR=/path/to/cache; when running Python "
            "directly, pass --ju_cache_dir /path/to/cache."
        )

    torch.manual_seed(int(JU.ju_seed))
    np.random.seed(int(JU.ju_seed))
    random.seed(int(JU.ju_seed))

    if bool(JU.ju_tf32) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dataset = _build_dataset()
    indices = _select_indices(dataset)
    loader_workers = int(getattr(cfgs, "num_workers", 2))
    loader_kwargs: Dict[str, Any] = dict(
        dataset=DeterministicIndexDataset(dataset, indices, int(JU.ju_seed)),
        batch_size=1,
        shuffle=False,
        num_workers=loader_workers,
        collate_fn=collate_fn,
        pin_memory=bool(JU.ju_pin_memory) and torch.cuda.is_available(),
    )
    if loader_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(JU.ju_prefetch_factor))
        loader_kwargs["persistent_workers"] = bool(JU.ju_persistent_workers)
    loader = DataLoader(**loader_kwargs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = _load_model(device)

    eval_workers = max(0, int(JU.ju_eval_workers))
    max_pending = max(1, int(JU.ju_max_pending))
    executor: Optional[cf.ProcessPoolExecutor] = None
    serial_evaluator: Optional[RawCandidateEvaluator] = None
    if eval_workers > 0:
        ctx = mp.get_context("spawn")
        executor = cf.ProcessPoolExecutor(
            max_workers=eval_workers,
            mp_context=ctx,
            initializer=_eval_worker_init,
            initargs=(
                cfgs.dataset_root,
                cfgs.camera,
                str(JU.ju_dataset_split),
                int(JU.ju_eval_chunk),
                bool(JU.ju_skip_force_closure),
                int(JU.ju_eval_threads),
                str(JU.ju_fc_mode),
                int(JU.ju_fc_verify_n),
                bool(JU.ju_strict),
            ),
        )
        print(
            f"[ASYNC] evaluator_workers={eval_workers} max_pending={max_pending} "
            f"eval_threads={int(JU.ju_eval_threads)} fc_mode={JU.ju_fc_mode} "
            f"fc_verify_n={int(JU.ju_fc_verify_n)}",
            flush=True,
        )
    else:
        serial_evaluator = RawCandidateEvaluator(
            cfgs.dataset_root,
            cfgs.camera,
            str(JU.ju_dataset_split),
            int(JU.ju_eval_chunk),
            bool(JU.ju_skip_force_closure),
            fc_mode=str(JU.ju_fc_mode),
            fc_verify_n=int(JU.ju_fc_verify_n),
            strict=bool(JU.ju_strict),
        )
        print("[ASYNC] disabled; using serial exact evaluator", flush=True)

    output_root = os.path.abspath(JU.ju_cache_dir)
    os.makedirs(output_root, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    pending: Dict[cf.Future, PendingFrame] = {}
    start_wall = time.time()
    loader_iter = None
    previous_signal_handlers = _install_main_signal_handlers()

    def drain(block: bool) -> None:
        if not pending:
            return
        mode = cf.FIRST_COMPLETED if block else cf.FIRST_COMPLETED
        timeout = None if block else 0.0
        done, _ = cf.wait(list(pending.keys()), timeout=timeout, return_when=mode)
        for future in done:
            item = pending.pop(future)
            payload = future.result()
            _finalize_pending(item, payload, summaries, start_wall)

    try:
        loader_iter = iter(loader)
        for local_idx, batch in enumerate(loader_iter):
            # Bound host memory and keep the GPU producer from outrunning CPU labelers.
            while executor is not None and len(pending) >= max_pending:
                drain(block=True)

            dataset_idx = indices[local_idx]
            scene_name = dataset.scene_list()[dataset_idx]
            scene_id = int(str(scene_name).split("_")[-1])
            anno_id = int(dataset_idx % 256)
            out_path = os.path.join(output_root, f"scene_{scene_id:04d}", f"ann_{anno_id:04d}.npz")
            if os.path.isfile(out_path) and not bool(JU.ju_overwrite):
                print(f"[SKIP] {out_path}", flush=True)
                continue

            gpu_started = time.perf_counter()
            _move_batch_to_device(
                batch,
                device,
                non_blocking=bool(JU.ju_pin_memory) and torch.cuda.is_available(),
            )
            rng_state = _capture_rng()
            with torch.inference_mode():
                base = net(_copy_inputs(batch))
            if "cva_angle_feature" not in base:
                raise KeyError(
                    "Missing cva_angle_feature. Apply patches/expose_cva_angle_feature.patch "
                    "to models/kview_query_transformer.py."
                )
            base_view_score = _canonical_view_score(base).detach()
            top_k = min(max(1, int(JU.ju_top_views)), int(base_view_score.shape[-1]))
            base_used = base["grasp_top_view_inds"].long()

            # ViewNet and torch.topk can choose different indices when multiple
            # views have exactly (or numerically) tied maximum scores.  The base
            # forward is authoritative for rank 0; using topk(...)[..., 0]
            # directly can therefore trigger a false strict failure and can also
            # duplicate the base view among the extra hypotheses.
            if not torch.isfinite(base_view_score).all():
                bad = int((~torch.isfinite(base_view_score)).sum().item())
                raise RuntimeError(f"view_score contains {bad} non-finite entries")

            argmax_view = torch.argmax(base_view_score, dim=-1)
            index_match = base_used.eq(argmax_view)
            top1_agreement = float(index_match.float().mean().item())

            row_max = base_view_score.amax(dim=-1)
            base_score = base_view_score.gather(-1, base_used.unsqueeze(-1)).squeeze(-1)
            score_span = (row_max - base_view_score.amin(dim=-1)).abs()
            tie_tol = torch.maximum(
                torch.full_like(score_span, 1.0e-6),
                score_span * 1.0e-6,
            )
            score_equivalent = (row_max - base_score).abs() <= tie_tol
            true_mismatch = (~index_match) & (~score_equivalent)
            tie_mismatch = (~index_match) & score_equivalent

            n_query = int(base_used.numel())
            n_tie = int(tie_mismatch.sum().item())
            n_true = int(true_mismatch.sum().item())
            if n_tie > 0:
                print(
                    f"[WARN] Base view differs from argmax index for {n_tie}/{n_query} "
                    "queries, but the selected scores are tied with the row maximum. "
                    "Preserving the model-selected base view as rank 0.",
                    flush=True,
                )
            if n_true > 0:
                max_gap = float((row_max - base_score)[true_mismatch].max().item())
                msg = (
                    "Base view is not score-equivalent to view_score argmax: "
                    f"count={n_true}/{n_query}, agreement={top1_agreement:.9f}, "
                    f"max_score_gap={max_gap:.6g}"
                )
                if bool(JU.ju_strict):
                    raise RuntimeError(msg)
                print(f"[WARN] {msg}; preserving base forward indices.", flush=True)

            # Keep the exact view used by the base forward at rank 0.  Select
            # the remaining K-1 hypotheses after masking that index, preventing
            # duplicate base-view candidates when argmax/topk tie-breaking differs.
            if top_k == 1:
                top_view = base_used.unsqueeze(-1)
            else:
                extra_score = base_view_score.clone()
                extra_score.scatter_(-1, base_used.unsqueeze(-1), float("-inf"))
                extra_view = torch.topk(
                    extra_score,
                    k=top_k - 1,
                    dim=-1,
                    largest=True,
                    sorted=True,
                ).indices
                top_view = torch.cat([base_used.unsqueeze(-1), extra_view], dim=-1)

            sample_rng = np.random.default_rng(int(JU.ju_seed) + scene_id * 1000 + anno_id)
            center_ids = _choose_centers(base, sample_rng)
            parts: List[Dict[str, np.ndarray]] = []
            for rank in range(top_k):
                if rank == 0:
                    current = base
                else:
                    forced = top_view[:, :, rank].long()
                    with torch.inference_mode():
                        current = _run_forced_view(
                            net, batch, rng_state, forced, base,
                            f"scene{scene_id}_ann{anno_id}_rank{rank}",
                        )
                parts.append(_extract_view_candidates(current, base_view_score, center_ids, rank, sample_rng))
                if rank > 0:
                    del current
                    if torch.cuda.is_available() and bool(JU.ju_empty_cache_each_view):
                        torch.cuda.empty_cache()

            candidates = _concat_candidate_dicts(parts)
            grasps = np.asarray(candidates.pop("grasp"), dtype=np.float32)
            # Metadata needed by the cache dataset; labels are attached after exact evaluation.
            candidates["num_angles"] = np.asarray(
                [base["grasp_score_pred_angle"].shape[-1]], dtype=np.int16
            )
            candidates["num_depths"] = np.asarray(
                [min(int(getattr(cfgs, "num_depth", 4)), base["grasp_depth_pred_angle"].shape[1] - 1)],
                dtype=np.int16,
            )
            candidates["top_views"] = np.asarray([top_k], dtype=np.int16)
            gpu_sec = time.perf_counter() - gpu_started
            item = PendingFrame(
                scene_id=scene_id,
                anno_id=anno_id,
                dataset_idx=dataset_idx,
                out_path=out_path,
                arrays=candidates,
                num_centers=int(len(center_ids)),
                submitted_at=time.perf_counter(),
                gpu_sec=gpu_sec,
            )

            if executor is None:
                assert serial_evaluator is not None
                eval_started = time.perf_counter()
                result = serial_evaluator.evaluate(scene_id, anno_id, grasps)
                payload = {
                    "assigned_obj": result.assigned_obj,
                    "collision_or_empty": result.collision_or_empty,
                    "empty": result.empty,
                    "pure_collision": result.pure_collision,
                    "friction": result.friction,
                    "eval_sec": time.perf_counter() - eval_started,
                    "worker_pid": os.getpid(),
                }
                _finalize_pending(item, payload, summaries, start_wall)
            else:
                future = executor.submit(_eval_worker_task, scene_id, anno_id, grasps)
                pending[future] = item
                # Opportunistically save completed frames without stalling the producer.
                drain(block=False)

            del batch, base, parts, candidates, grasps

        while pending:
            drain(block=True)
    except _StopRequested as exc:
        # Do not wait for running DexNet evaluations.  Cancel queued futures,
        # terminate active spawn workers, and exit with the conventional
        # 128+signal status.  Completed .npz files remain valid for resume.
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except Exception:
            pass
        _abort_process_pool(executor, pending)
        executor = None
        interrupted = {
            "worker_tag": str(JU.ju_worker_tag),
            "signal": int(exc.signum),
            "signal_name": signal.Signals(exc.signum).name,
            "num_completed_files": len(summaries),
            "elapsed_sec": time.time() - start_wall,
            "status": "interrupted",
        }
        _save_json(
            interrupted,
            os.path.join(output_root, f"_mine_interrupted_{JU.ju_worker_tag}.json"),
        )
        print(
            f"[STOP] producer {JU.ju_worker_tag} exited after saving "
            f"{len(summaries)} completed files",
            flush=True,
        )
        raise SystemExit(128 + int(exc.signum))
    except KeyboardInterrupt:
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except Exception:
            pass
        _abort_process_pool(executor, pending)
        executor = None
        _save_json(
            {
                "worker_tag": str(JU.ju_worker_tag),
                "signal": int(signal.SIGINT),
                "signal_name": "SIGINT",
                "num_completed_files": len(summaries),
                "elapsed_sec": time.time() - start_wall,
                "status": "interrupted",
            },
            os.path.join(output_root, f"_mine_interrupted_{JU.ju_worker_tag}.json"),
        )
        raise SystemExit(130)
    except BaseException:
        # Unexpected producer errors must not leave evaluator/DataLoader workers
        # running in the background.
        _abort_process_pool(executor, pending)
        executor = None
        raise
    else:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
            executor = None
    finally:
        _shutdown_loader_iterator(loader_iter)
        _restore_signal_handlers(previous_signal_handlers)

    elapsed = time.time() - start_wall
    completion = {
        "worker_tag": str(JU.ju_worker_tag),
        "checkpoint_path": str(cfgs.checkpoint_path),
        "dataset_split": str(JU.ju_dataset_split),
        "scene_ids": _csv_ints(JU.ju_scene_ids),
        "sample_interval": float(JU.ju_sample_interval),
        "top_views": int(JU.ju_top_views),
        "top_centers": int(JU.ju_top_centers),
        "random_centers": int(JU.ju_random_centers),
        "top_angles": int(JU.ju_top_angles),
        "random_angles": int(JU.ju_random_angles),
        "all_angles": bool(JU.ju_all_angles),
        "eval_workers": eval_workers,
        "max_pending": max_pending,
        "eval_threads": int(JU.ju_eval_threads),
        "num_new_files": len(summaries),
        "elapsed_sec": elapsed,
        "mean_gpu_candidate_sec": float(np.mean([x["gpu_candidate_sec"] for x in summaries])) if summaries else None,
        "mean_eval_sec": float(np.mean([x["eval_sec"] for x in summaries])) if summaries else None,
        "mean_queue_latency_sec": float(np.mean([x["queue_latency_sec"] for x in summaries])) if summaries else None,
        "summaries": summaries,
    }
    _save_json(completion, os.path.join(output_root, f"_mine_complete_{JU.ju_worker_tag}.json"))
    print(f"[DONE] mined {len(summaries)} files into {output_root} in {elapsed/60.0:.1f}m", flush=True)


if __name__ == "__main__":
    main()
