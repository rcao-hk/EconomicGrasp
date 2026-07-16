#!/usr/bin/env python3
"""Validate and quantify the first-generation CVA view override/oracle path.

For every selected RGB-D frame this program performs four deterministic passes
with identical center queries:

- ``base``: normal predicted view;
- ``identity``: explicitly force the predicted view (must reproduce base);
- ``farthest``: force the angularly farthest view (must change downstream);
- ``label_oracle``: force the label-best view.

It validates control flow, compares downstream endpoints and module-hook
signatures, reports view-label coverage/regret at K={1,4,8}, saves optional
GraspNet dumps, and optionally computes official per-sample AP for all modes.

This is a diagnostic program using GT labels. It targets kview_k=1 and requires
the stage-wise-oracle model patch that accepts
``end_points['oracle_view_inds_override']`` before grouping/CVA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _consume_custom_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--view_output_dir", default=None)
    p.add_argument("--view_scene_ids", default="")
    p.add_argument("--view_anno_ids", default="")
    p.add_argument("--view_max_samples", type=int, default=16)
    p.add_argument("--view_eval_official", type=int, default=1)
    p.add_argument("--view_save_dumps", type=int, default=1)
    p.add_argument("--view_topks", default="1,4,8")
    p.add_argument("--view_label_min", type=float, default=1e-6)
    p.add_argument("--view_hook_patterns", default="group,crop,patch,transformer,head")
    p.add_argument("--view_hook_sample_size", type=int, default=4096)
    p.add_argument("--view_disable_hooks", type=int, default=0)
    p.add_argument("--view_top_k_eval", type=int, default=50)
    p.add_argument("--view_seed", type=int, default=0)
    p.add_argument("--view_strict", type=int, default=1)
    p.add_argument(
        "--view_override_injection",
        choices=["score", "selector", "endpoint"],
        default="score",
        help=(
            "score: force the requested view both at the selector and by minimally "
            "pinning its view_score above the current maximum. This survives wrappers "
            "that recompute argmax(view_score). selector: replace only the selector "
            "return value. endpoint: use the model's explicit endpoint override path."
        ),
    )
    p.add_argument("--view_identity_atol", type=float, default=1e-6)
    p.add_argument("--view_identity_rtol", type=float, default=1e-5)
    custom, remaining = p.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    return custom


VIEW = _consume_custom_args()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from graspnetAPI import GraspGroup, GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    create_table_points,
    eval_grasp,
    transform_points,
    voxel_sample_points,
)

from utils.arguments import cfgs
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn


FRICTIONS = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float32)
MODES = ["base", "identity", "farthest", "label_oracle"]


def _csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _save_json(payload: Mapping[str, Any], path: str) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(conv(dict(payload)), f, indent=2)
    os.replace(tmp, path)


def _save_csv(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted(set().union(*(set(r.keys()) for r in rows)))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _move_batch_to_device(batch: MutableMapping[str, Any], device: torch.device) -> None:
    for key in list(batch.keys()):
        value = batch[key]
        if "list" in key:
            for i in range(len(value)):
                for j in range(len(value[i])):
                    value[i][j] = value[i][j].to(device)
        elif "graph" in key:
            for i in range(len(value)):
                value[i] = value[i].to(device)
        elif torch.is_tensor(value):
            batch[key] = value.to(device)


def _worker_init(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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


def _copy_inputs(batch: Mapping[str, Any], override: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    out = dict(batch)
    if override is not None:
        out["oracle_view_inds_override"] = override
    return out


def _build_dataset():
    cls = GraspNetMultiDataset if cfgs.multi_modal else GraspNetDataset
    return cls(
        cfgs.dataset_root,
        split=str(cfgs.test_mode),
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=True,
        extend_angle=cfgs.extend_angle
    )


def _select_indices(dataset) -> Tuple[List[int], List[str]]:
    scenes = set(_csv_ints(VIEW.view_scene_ids))
    annos = set(_csv_ints(VIEW.view_anno_ids))
    scene_list = list(dataset.scene_list())
    interval = float(getattr(cfgs, "sample_interval", 1.0))
    stride = 1 if interval >= 1.0 else max(1, int(round(1.0 / interval)))
    out: List[int] = []
    for idx, scene_name in enumerate(scene_list):
        try:
            sid = int(str(scene_name).split("_")[-1])
        except ValueError:
            continue
        anno = idx % 256
        if scenes and sid not in scenes:
            continue
        if annos:
            if anno not in annos:
                continue
        elif anno % stride:
            continue
        out.append(idx)
        if VIEW.view_max_samples > 0 and len(out) >= VIEW.view_max_samples:
            break
    if not out:
        raise RuntimeError("No samples selected for view-oracle validation.")
    return out, scene_list


def _load_model(device: torch.device):
    if not cfgs.multi_modal:
        raise ValueError("Pass --multi_modal for the first-generation CVA model.")
    if int(getattr(cfgs, "kview_k", 1)) != 1:
        raise ValueError("This validator targets --kview_k 1, not RotNet/top-L.")
    from models.economicgrasp_bip3d import economicgrasp_dpt

    try:
        net = economicgrasp_dpt(
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            bin_num=cfgs.bin_num,
            is_training=False,
            use_obs_depth=bool(getattr(cfgs, "use_obs_depth", False)),
            vis_dir=getattr(cfgs, "vis_dir", None),
            vis_every=int(getattr(cfgs, "vis_every", 1000)),
            oracle_diag=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "The model lacks oracle_diag support. Apply the stage-wise-oracle patch first."
        ) from exc
    ckpt = torch.load(cfgs.checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        net.load_state_dict(state)
    except RuntimeError:
        if state and all(str(k).startswith("module.") for k in state):
            net.load_state_dict({str(k)[7:]: v for k, v in state.items()})
        else:
            raise
    return net.to(device).eval()


def _find_view_dirs(net: torch.nn.Module) -> torch.Tensor:
    candidates: List[Any] = [
        getattr(net, "view_dirs", None),
        getattr(getattr(net, "view", None), "view_dirs", None),
        getattr(getattr(getattr(net, "kview_grasp_module", None), "view_net", None), "view_dirs", None),
        getattr(getattr(net, "kview_grasp_module", None), "view_dirs", None),
    ]
    for x in candidates:
        if torch.is_tensor(x) and x.dim() == 2 and x.shape[-1] == 3:
            return x
    for name, buffer in net.named_buffers():
        if "view" in name.lower() and buffer.dim() == 2 and buffer.shape[-1] == 3:
            return buffer
    raise KeyError("Could not locate the [V,3] view-direction tensor in the model.")



class DirectViewSelectionOverride:
    """Force the active ViewNet selection at its actual decision boundary.

    ``selector`` replaces only ``_select_top_view_inds``.  Some CVA wrappers,
    however, later recompute ``argmax(view_score)`` and overwrite the ViewNet's
    selected indices.  ``score`` therefore performs two consistent actions:

      1. return the requested forced indices from the selector; and
      2. for rows whose requested view differs from the normal prediction,
         minimally raise that view's score above the current row maximum.

    The second action makes any downstream re-selection from ``view_score``
    choose the same forced view.  Identity passes are not modified because
    ``forced == normal`` for every row, so they remain exact controls.
    """

    def __init__(self, net: torch.nn.Module, mode: str = "score"):
        self.mode = str(mode)
        self.current: Optional[torch.Tensor] = None
        self.calls = 0
        self.forced_calls = 0
        self.score_pinned_rows = 0
        self.score_argmax_respected_rows = 0
        self.last_shape: Optional[Tuple[int, ...]] = None
        self.view_net = self._locate_view_net(net)
        self._original_select = None

        if self.mode in {"selector", "score"}:
            select = getattr(self.view_net, "_select_top_view_inds", None)
            if not callable(select):
                raise AttributeError(
                    "The active ViewNet has no callable _select_top_view_inds; "
                    "cannot apply direct selector/score override."
                )
            self._original_select = select

            def wrapped_select(view_score: torch.Tensor) -> torch.Tensor:
                normal = self._original_select(view_score)
                self.calls += 1
                self.last_shape = tuple(int(x) for x in view_score.shape)
                forced = self.current
                if forced is None:
                    return normal
                forced = forced.to(device=view_score.device, dtype=torch.long)
                expected = tuple(int(x) for x in view_score.shape[:2])
                if tuple(forced.shape) != expected:
                    raise ValueError(
                        "Forced view indices must match ViewNet [B,Q], got "
                        f"{tuple(forced.shape)} vs {expected}."
                    )
                if ((forced < 0) | (forced >= view_score.shape[-1])).any():
                    raise ValueError("Forced view indices contain out-of-range entries.")

                self.forced_calls += 1
                if self.mode == "score":
                    changed = forced.ne(normal)
                    changed_count = int(changed.sum().item())
                    if changed_count:
                        # Pin only changed rows.  A relative + absolute margin is
                        # used to remain effective across arbitrary score scales.
                        # The operation is diagnostic and runs under no_grad().
                        row_max = view_score.detach().amax(dim=-1)
                        row_min = view_score.detach().amin(dim=-1)
                        span = (row_max - row_min).abs()
                        margin = torch.maximum(
                            span * 1.0e-4,
                            torch.full_like(span, 1.0e-4),
                        )
                        forced_old = view_score.gather(
                            -1, forced.unsqueeze(-1)
                        ).squeeze(-1)
                        forced_new = torch.where(
                            changed,
                            torch.maximum(forced_old, row_max + margin),
                            forced_old,
                        )
                        view_score.scatter_(
                            -1, forced.unsqueeze(-1), forced_new.unsqueeze(-1)
                        )
                        self.score_pinned_rows += changed_count

                    argmax_after = view_score.argmax(dim=-1)
                    self.score_argmax_respected_rows += int(
                        argmax_after.eq(forced).sum().item()
                    )
                return forced

            # Assigning to the instance intentionally installs a callable that
            # receives only view_score, matching the existing bound call site.
            setattr(self.view_net, "_select_top_view_inds", wrapped_select)

    @staticmethod
    def _locate_view_net(net: torch.nn.Module) -> torch.nn.Module:
        kview = getattr(net, "kview_grasp_module", None)
        candidates = [
            getattr(kview, "view_net", None),
            getattr(kview, "view", None),
            getattr(net, "view", None),
            getattr(net, "view_net", None),
        ]
        for candidate in candidates:
            if isinstance(candidate, torch.nn.Module):
                return candidate
        raise AttributeError("Could not locate the active ViewNet module.")

    def set(self, forced: Optional[torch.Tensor]) -> None:
        self.current = forced

    def outer_input(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        # Keep endpoint mode for comparison/backward compatibility.
        if self.mode == "endpoint" and self.current is not None:
            return _copy_inputs(batch, self.current)
        return _copy_inputs(batch)

    def close(self) -> None:
        self.current = None
        if self.mode in {"selector", "score"} and self._original_select is not None:
            setattr(self.view_net, "_select_top_view_inds", self._original_select)


def _as_bqv(x: torch.Tensor, q: int, v: int, name: str) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"{name} must be 3D, got {tuple(x.shape)}")
    if x.shape[1:] == (q, v):
        return x
    if x.shape[1:] == (v, q):
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot canonicalize {name} to [B,Q,V]=[B,{q},{v}], got {tuple(x.shape)}")


def _derive_label_oracle(end_points: Mapping[str, Any], view_dirs: torch.Tensor):
    pred = end_points["grasp_top_view_inds"].long()
    b, q = pred.shape
    v = view_dirs.shape[0]
    score = _as_bqv(end_points["view_score"].float(), q, v, "view_score")
    label = _as_bqv(end_points["batch_grasp_view_graspness"].float(), q, v, "batch_grasp_view_graspness")
    finite = torch.isfinite(label)
    neg = torch.finfo(label.dtype).min
    safe = torch.where(finite, label, torch.full_like(label, neg))
    oracle_value, oracle_raw = safe.max(dim=-1)
    valid = finite.any(dim=-1) & (oracle_value > float(VIEW.view_label_min))
    oracle = torch.where(valid, oracle_raw.long(), pred)
    pred_label = label.gather(-1, pred.clamp(0, v - 1).unsqueeze(-1)).squeeze(-1)
    return pred, oracle, valid, score, label, pred_label, oracle_value


def _farthest_views(pred: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
    dirs = view_dirs.to(pred.device)
    pred_dir = dirs.index_select(0, pred.reshape(-1)).reshape(*pred.shape, 3)
    dots = torch.einsum("bqc,vc->bqv", pred_dir, dirs)
    return dots.argmin(dim=-1)


def _angular_deg(a: torch.Tensor, b: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
    dirs = view_dirs.to(a.device)
    da = dirs.index_select(0, a.reshape(-1)).reshape(*a.shape, 3)
    db = dirs.index_select(0, b.reshape(-1)).reshape(*b.shape, 3)
    cos = (da * db).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    if a.shape != b.shape:
        return {"shape_equal": 0.0, "exact_equal": 0.0, "max_abs": float("inf"), "rel_l2": float("inf"), "changed_ratio": 1.0}
    af = a.detach().float()
    bf = b.detach().float()
    diff = af - bf
    denom = torch.linalg.vector_norm(af.reshape(-1)).clamp_min(1e-12)
    changed = (diff.abs() > (float(VIEW.view_identity_atol) + float(VIEW.view_identity_rtol) * af.abs())).float().mean()
    return {
        "shape_equal": 1.0,
        "exact_equal": float(torch.equal(a, b)),
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "rel_l2": float((torch.linalg.vector_norm(diff.reshape(-1)) / denom).item()) if diff.numel() else 0.0,
        "changed_ratio": float(changed.item()) if diff.numel() else 0.0,
    }


class HookSignatures:
    """Sample deterministic signatures from downstream modules for each pass."""

    def __init__(
        self,
        net: torch.nn.Module,
        patterns: Sequence[str],
        sample_size: int,
        disabled: bool = False,
    ):
        normalized = [x.strip().lower() for x in patterns if x and x.strip()]
        self.disabled = bool(disabled) or any(x in {"none", "off", "disable", "disabled"} for x in normalized)
        self.patterns = [x for x in normalized if x not in {"none", "off", "disable", "disabled"}]
        self.sample_size = max(16, int(sample_size))
        self.current_pass = ""
        self.data: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.handles = []

        if self.disabled:
            return

        def tensors(obj: Any) -> List[torch.Tensor]:
            if torch.is_tensor(obj):
                return [obj]
            if isinstance(obj, Mapping):
                out: List[torch.Tensor] = []
                for v in obj.values():
                    out.extend(tensors(v))
                return out
            if isinstance(obj, (list, tuple)):
                out = []
                for v in obj:
                    out.extend(tensors(v))
                return out
            return []

        for name, module in net.named_modules():
            low = name.lower()
            if not name or "kview_grasp_module" not in low:
                continue
            if self.patterns and not any(p in low for p in self.patterns):
                continue

            def make_hook(module_name: str):
                def hook(_module, _inputs, output):
                    if not self.current_pass:
                        return
                    ts = tensors(output)
                    if not ts:
                        return
                    largest = max(ts, key=lambda z: z.numel())
                    # Sample before converting the full tensor to float.  The previous
                    # implementation used CUDA float32 linspace followed by .long().
                    # For tensors with >2^24 elements, float32 cannot exactly represent
                    # ``numel - 1`` and can round an endpoint to ``numel``, producing an
                    # out-of-range index and a device-side assert in index_select.
                    flat = largest.detach().reshape(-1)
                    n = int(flat.numel())
                    if n == 0:
                        sample = flat.to(dtype=torch.float32)
                    else:
                        k = min(self.sample_size, n)
                        if k == n:
                            sample = flat.to(dtype=torch.float32)
                        elif k == 1:
                            sample = flat[:1].to(dtype=torch.float32)
                        else:
                            # Pure int64 arithmetic: indices are guaranteed to lie in
                            # [0, n-1], even for very large CUDA tensors.
                            pos = torch.arange(k, device=flat.device, dtype=torch.int64)
                            idx = torch.div(pos * (n - 1), k - 1, rounding_mode="floor")
                            idx.clamp_(0, n - 1)
                            sample = flat.index_select(0, idx).to(dtype=torch.float32)

                    # Statistics over the deterministic sample are sufficient for a
                    # signature and avoid an expensive full-tensor float32 copy/reduce.
                    self.data[self.current_pass][module_name] = {
                        "shape": list(largest.shape),
                        "numel": n,
                        "sample": sample.cpu().numpy(),
                        "mean": float(sample.mean().item()) if sample.numel() else 0.0,
                        "std": float(sample.std(unbiased=False).item()) if sample.numel() else 0.0,
                        "stats_scope": "deterministic_sample",
                    }
                return hook

            self.handles.append(module.register_forward_hook(make_hook(name)))

    def begin(self, pass_name: str) -> None:
        self.current_pass = pass_name
        self.data[pass_name] = {}

    def end(self) -> None:
        self.current_pass = ""

    def compare(self, ref: str, other: str) -> List[Dict[str, Any]]:
        rows = []
        common = sorted(set(self.data.get(ref, {})) & set(self.data.get(other, {})))
        for name in common:
            a = self.data[ref][name]
            b = self.data[other][name]
            sa = np.asarray(a["sample"], dtype=np.float64)
            sb = np.asarray(b["sample"], dtype=np.float64)
            if sa.shape != sb.shape:
                rel = float("inf")
                mx = float("inf")
            else:
                diff = sa - sb
                rel = float(np.linalg.norm(diff) / max(np.linalg.norm(sa), 1e-12))
                mx = float(np.max(np.abs(diff))) if len(diff) else 0.0
            rows.append(
                {
                    "reference": ref,
                    "other": other,
                    "module": name,
                    "shape_ref": str(a["shape"]),
                    "shape_other": str(b["shape"]),
                    "sample_rel_l2": rel,
                    "sample_max_abs": mx,
                    "mean_ref": a["mean"],
                    "mean_other": b["mean"],
                }
            )
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()


class OfficialEvaluator:
    def __init__(self, root: str, camera: str, split: str, top_k: int):
        self.evaluator = GraspNetEval(root, camera, split=split)
        self.config = get_config()
        self.top_k = int(top_k)
        self.table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        self.scene_cache: Dict[int, Tuple[List[np.ndarray], List[Any]]] = {}

    @staticmethod
    def _acc(scores: np.ndarray, top_k: int) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        acc = np.zeros((top_k, len(FRICTIONS)), dtype=np.float32)
        for j, mu in enumerate(FRICTIONS):
            success = ((scores > 0.0) & (scores <= mu)).astype(np.float32)
            prefix = np.cumsum(success)
            for k in range(1, top_k + 1):
                n = min(k, len(scores))
                acc[k - 1, j] = (float(prefix[n - 1]) if n else 0.0) / float(k)
        return acc

    def evaluate(self, scene_id: int, anno_id: int, grasp_tensor: torch.Tensor) -> Dict[str, float]:
        if scene_id not in self.scene_cache:
            models, dex, _ = self.evaluator.get_scene_models(scene_id, ann_id=0)
            self.scene_cache[scene_id] = ([voxel_sample_points(m, 0.008) for m in models], dex)
        models, dex = self.scene_cache[scene_id]
        _, poses, camera_pose, align_mat = self.evaluator.get_model_poses(scene_id, anno_id)
        table = transform_points(self.table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        arr = grasp_tensor.detach().cpu().numpy().astype(np.float32)
        if len(arr):
            arr[:, 1] = np.clip(arr[:, 1], 0.0, 0.1)
        gg = GraspGroup(arr)
        gl, sl, cl = eval_grasp(
            gg,
            models,
            dex,
            poses,
            self.config,
            table=table,
            voxel_size=0.008,
            TOP_K=self.top_k,
        )
        nonempty = [i for i, x in enumerate(gl) if len(x)]
        if not nonempty:
            return {"ap": 0.0, "ap04": 0.0, "ap08": 0.0, "top10_collision": 0.0, "top10_fail": 0.0}
        grasps = np.concatenate([gl[i] for i in nonempty])
        scores = np.concatenate([sl[i] for i in nonempty])
        collision = np.concatenate([cl[i] for i in nonempty]).astype(bool)
        order = np.argsort(-grasps[:, 0], kind="stable")
        scores = scores[order]
        collision = collision[order]
        acc = self._acc(scores, self.top_k)
        k = min(10, len(scores))
        return {
            "ap": float(acc.mean()),
            "ap04": float(acc[:, 1].mean()),
            "ap08": float(acc[:, 3].mean()),
            "top10_collision": float(collision[:k].mean()) if k else 0.0,
            "top10_fail": float((scores[:k] <= 0.0).mean()) if k else 0.0,
        }


def _save_dump(out_dir: str, mode: str, scene_id: int, anno_id: int, grasp: torch.Tensor) -> None:
    path = os.path.join(out_dir, "dumps", mode, f"scene_{scene_id:04d}", cfgs.camera, f"{anno_id:04d}.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    GraspGroup(grasp.detach().cpu().numpy()).save_npy(path)


def _assert_same_queries(base: Mapping[str, Any], other: Mapping[str, Any], name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ["token_sel_idx", "xyz_graspable"]:
        if not (torch.is_tensor(base.get(key)) and torch.is_tensor(other.get(key))):
            raise KeyError(f"Passes must expose {key}")
        d = _tensor_diff(base[key], other[key])
        out[f"{name}_{key}_exact"] = d["exact_equal"]
        out[f"{name}_{key}_rel_l2"] = d["rel_l2"]
        if key == "token_sel_idx" and not torch.equal(base[key], other[key]):
            raise RuntimeError(f"{name} pass selected different token queries despite RNG restoration.")
        if key == "xyz_graspable" and not torch.allclose(base[key], other[key], atol=1e-7, rtol=1e-6):
            raise RuntimeError(f"{name} pass selected different center XYZ despite RNG restoration.")
    return out


def main() -> None:
    random.seed(VIEW.view_seed)
    np.random.seed(VIEW.view_seed)
    torch.manual_seed(VIEW.view_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(VIEW.view_seed)

    out_dir = VIEW.view_output_dir or os.path.join(cfgs.save_dir, f"view_oracle_check_{cfgs.test_mode}")
    os.makedirs(out_dir, exist_ok=True)
    dataset = _build_dataset()
    indices, scene_list = _select_indices(dataset)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(cfgs.batch_size),
        shuffle=False,
        num_workers=int(getattr(cfgs, "num_workers", 2)),
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = _load_model(device)
    view_dirs = _find_view_dirs(net).to(device)
    override_controller = DirectViewSelectionOverride(net, VIEW.view_override_injection)
    topks = _csv_ints(VIEW.view_topks)

    from models.economicgrasp_bip3d import pred_decode_center_view_angle

    hook = HookSignatures(
        net,
        [x.strip() for x in VIEW.view_hook_patterns.split(",") if x.strip()],
        VIEW.view_hook_sample_size,
        disabled=bool(VIEW.view_disable_hooks),
    )
    official = OfficialEvaluator(cfgs.dataset_root, cfgs.camera, str(cfgs.test_mode), VIEW.view_top_k_eval) if VIEW.view_eval_official else None

    per_sample: List[Dict[str, Any]] = []
    endpoint_rows: List[Dict[str, Any]] = []
    hook_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    start = time.time()

    prediction_keys = [
        "grasp_top_view_inds",
        "grasp_top_view_xyz",
        "grasp_top_view_rot",
        "grasp_score_pred_angle",
        "grasp_depth_pred_angle",
        "grasp_width_pred_angle",
        "grasp_collision_pred_angle",
    ]

    try:
        sample_counter = 0
        for batch_idx, batch in enumerate(loader):
            _move_batch_to_device(batch, device)
            rng = _capture_rng()

            override_controller.set(None)
            hook.begin("base")
            with torch.no_grad():
                base = net(override_controller.outer_input(batch))
            hook.end()
            pred, oracle_idx, oracle_valid, view_score, view_label, pred_gt, oracle_gt = _derive_label_oracle(base, view_dirs)
            farthest_idx = _farthest_views(pred, view_dirs)

            passes: Dict[str, Mapping[str, Any]] = {"base": base}
            for name, forced in [("identity", pred), ("farthest", farthest_idx), ("label_oracle", oracle_idx)]:
                _restore_rng(rng)
                override_controller.set(forced)
                hook.begin(name)
                with torch.no_grad():
                    passes[name] = net(override_controller.outer_input(batch))
                hook.end()
                _assert_same_queries(base, passes[name], name)
            override_controller.set(None)

            decoded = {name: pred_decode_center_view_angle(ep) for name, ep in passes.items()}
            bs = pred.shape[0]
            q = pred.shape[1]
            v = view_dirs.shape[0]
            if view_label.shape != (bs, q, v):
                raise RuntimeError(f"View label is not full [B,Q,V]: {tuple(view_label.shape)} expected {(bs,q,v)}")

            # Module-level downstream signatures are batch-level; attach batch index.
            for name in ["identity", "farthest", "label_oracle"]:
                for r in hook.compare("base", name):
                    hook_rows.append({"batch_idx": batch_idx, **r})

            for b in range(bs):
                subset_pos = batch_idx * int(cfgs.batch_size) + b
                if subset_pos >= len(indices):
                    continue
                data_idx = indices[subset_pos]
                scene_id = int(str(scene_list[data_idx]).split("_")[-1])
                anno_id = int(data_idx % 256)
                row: Dict[str, Any] = {
                    "split": str(cfgs.test_mode),
                    "scene_id": scene_id,
                    "anno_id": anno_id,
                    "num_queries": q,
                    "oracle_valid_ratio": float(oracle_valid[b].float().mean().item()),
                    "pred_eq_oracle_ratio": float((pred[b] == oracle_idx[b]).float().mean().item()),
                    "oracle_forced_change_ratio": float((oracle_valid[b] & (pred[b] != oracle_idx[b])).float().mean().item()),
                    "pred_oracle_angle_deg_mean": float(_angular_deg(pred[b:b+1], oracle_idx[b:b+1], view_dirs)[0][oracle_valid[b]].mean().item()) if oracle_valid[b].any() else float("nan"),
                    "pred_gt_view_score_mean": float(pred_gt[b][oracle_valid[b]].mean().item()) if oracle_valid[b].any() else float("nan"),
                    "oracle_gt_view_score_mean": float(oracle_gt[b][oracle_valid[b]].mean().item()) if oracle_valid[b].any() else float("nan"),
                    "view_gt_regret_mean": float((oracle_gt[b] - pred_gt[b])[oracle_valid[b]].clamp_min(0).mean().item()) if oracle_valid[b].any() else float("nan"),
                    "farthest_angle_deg_mean": float(_angular_deg(pred[b:b+1], farthest_idx[b:b+1], view_dirs)[0].mean().item()),
                }

                for k in topks:
                    kk = min(k, v)
                    inds = torch.topk(view_score[b], k=kk, dim=-1).indices
                    vals = view_label[b].gather(-1, inds)
                    best_topk = vals.max(dim=-1).values
                    row[f"pred_top{k}_positive_coverage"] = float((best_topk > float(VIEW.view_label_min)).float().mean().item())
                    row[f"pred_top{k}_utility_regret"] = float((oracle_gt[b] - best_topk)[oracle_valid[b]].clamp_min(0).mean().item()) if oracle_valid[b].any() else float("nan")
                    row[f"oracle_in_pred_top{k}"] = float((inds == oracle_idx[b, :, None]).any(-1)[oracle_valid[b]].float().mean().item()) if oracle_valid[b].any() else float("nan")

                for name in ["identity", "farthest", "label_oracle"]:
                    ep = passes[name]
                    used = ep["grasp_top_view_inds"][b].long()
                    target = {"identity": pred[b], "farthest": farthest_idx[b], "label_oracle": oracle_idx[b]}[name]
                    mask = oracle_valid[b] if name == "label_oracle" else torch.ones_like(target, dtype=torch.bool)
                    row[f"{name}_override_respected_ratio"] = float((used[mask] == target[mask]).float().mean().item()) if mask.any() else float("nan")
                    row[f"{name}_used_changed_vs_base"] = float((used != pred[b]).float().mean().item())

                    expected_xyz = view_dirs.index_select(0, used.reshape(-1)).reshape(q, 3)
                    xyz = ep["grasp_top_view_xyz"][b].float()
                    row[f"{name}_view_xyz_consistency_cos"] = float((torch.nn.functional.normalize(expected_xyz, dim=-1) * torch.nn.functional.normalize(xyz, dim=-1)).sum(-1).mean().item())

                    for key in prediction_keys:
                        if torch.is_tensor(base.get(key)) and torch.is_tensor(ep.get(key)):
                            d = _tensor_diff(base[key][b], ep[key][b])
                            endpoint_rows.append(
                                {
                                    "scene_id": scene_id,
                                    "anno_id": anno_id,
                                    "comparison": f"base_vs_{name}",
                                    "key": key,
                                    **d,
                                }
                            )
                            row[f"{name}_{key}_rel_l2"] = d["rel_l2"]
                            row[f"{name}_{key}_changed_ratio"] = d["changed_ratio"]

                    pred_diff = _tensor_diff(decoded["base"][b], decoded[name][b])
                    row[f"{name}_decoded_rel_l2"] = pred_diff["rel_l2"]
                    row[f"{name}_decoded_max_abs"] = pred_diff["max_abs"]
                    row[f"{name}_decoded_changed_ratio"] = pred_diff["changed_ratio"]

                if official is not None:
                    for name in MODES:
                        metrics = official.evaluate(scene_id, anno_id, decoded[name][b])
                        for key, value in metrics.items():
                            row[f"{name}_{key}"] = value

                if VIEW.view_save_dumps:
                    for name in MODES:
                        _save_dump(out_dir, name, scene_id, anno_id, decoded[name][b])

                # Strict smoke-test checks.
                if row["identity_override_respected_ratio"] < 0.999999:
                    failures.append(f"scene {scene_id} anno {anno_id}: identity override not respected")
                if row["identity_decoded_max_abs"] > float(VIEW.view_identity_atol) * 10:
                    failures.append(f"scene {scene_id} anno {anno_id}: identity decoded output changed ({row['identity_decoded_max_abs']:.3e})")
                if row["farthest_override_respected_ratio"] < 0.999:
                    failures.append(f"scene {scene_id} anno {anno_id}: farthest override not respected")
                if row["farthest_used_changed_vs_base"] < 0.99:
                    failures.append(f"scene {scene_id} anno {anno_id}: farthest view did not change nearly all queries")
                if row["farthest_grasp_top_view_xyz_changed_ratio"] < 0.5:
                    failures.append(f"scene {scene_id} anno {anno_id}: farthest view did not propagate to view XYZ")
                if oracle_valid[b].any() and row["label_oracle_override_respected_ratio"] < 0.999:
                    failures.append(f"scene {scene_id} anno {anno_id}: label oracle override not respected")

                per_sample.append(row)
                sample_counter += 1
                print(
                    f"[{sample_counter}/{len(indices)}] scene={scene_id:04d} anno={anno_id:04d} "
                    f"pred=oracle={row['pred_eq_oracle_ratio']:.3f} "
                    f"far_changed={row['farthest_used_changed_vs_base']:.3f} "
                    f"oracle_changed={row['oracle_forced_change_ratio']:.3f} "
                    + (f"AP base/oracle={row['base_ap']:.3f}/{row['label_oracle_ap']:.3f} " if official is not None else "")
                    + f"elapsed={(time.time()-start)/60:.1f}m"
                )
                _save_csv(per_sample, os.path.join(out_dir, "view_oracle_per_sample.partial.csv"))
                _save_csv(endpoint_rows, os.path.join(out_dir, "view_oracle_endpoint_diffs.partial.csv"))
                _save_csv(hook_rows, os.path.join(out_dir, "view_oracle_hook_diffs.partial.csv"))

    finally:
        override_controller.close()
        hook.close()

    _save_csv(per_sample, os.path.join(out_dir, "view_oracle_per_sample.csv"))
    _save_csv(endpoint_rows, os.path.join(out_dir, "view_oracle_endpoint_diffs.csv"))
    _save_csv(hook_rows, os.path.join(out_dir, "view_oracle_hook_diffs.csv"))

    summary: Dict[str, Any] = {
        "split": str(cfgs.test_mode),
        "num_samples": len(per_sample),
        "num_failures": len(failures),
        "failures": failures,
        "elapsed_minutes": (time.time() - start) / 60.0,
        "override_injection_mode": VIEW.view_override_injection,
        "view_selector_calls": int(override_controller.calls),
        "view_selector_forced_calls": int(override_controller.forced_calls),
        "view_score_pinned_rows": int(override_controller.score_pinned_rows),
        "view_score_argmax_respected_rows": int(override_controller.score_argmax_respected_rows),
        "view_selector_last_score_shape": override_controller.last_shape,
        "means": {},
        "interpretation_checks": {},
    }
    if per_sample:
        numeric = [k for k in per_sample[0] if k not in {"split", "scene_id", "anno_id"}]
        for key in numeric:
            vals = []
            for r in per_sample:
                try:
                    value = float(r[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    vals.append(value)
            if vals:
                summary["means"][key] = float(np.mean(vals))

    summary["interpretation_checks"] = {
        "identity_control_pass": bool(
            summary["means"].get("identity_override_respected_ratio", 0.0) > 0.999
            and summary["means"].get("identity_decoded_max_abs", float("inf")) < float(VIEW.view_identity_atol) * 10
        ),
        "farthest_counterfactual_pass": bool(
            summary["means"].get("farthest_override_respected_ratio", 0.0) > 0.999
            and summary["means"].get("farthest_used_changed_vs_base", 0.0) > 0.99
            and summary["means"].get("farthest_grasp_top_view_xyz_changed_ratio", 0.0) > 0.5
        ),
        "label_oracle_override_pass": bool(
            summary["means"].get("label_oracle_override_respected_ratio", 0.0) > 0.999
        ),
        "label_is_full_300_view": bool(view_dirs.shape[0] == int(getattr(cfgs, "num_view", 300))),
    }
    if official is not None:
        summary["official_ap_delta"] = {
            "identity_minus_base": summary["means"].get("identity_ap", float("nan")) - summary["means"].get("base_ap", float("nan")),
            "farthest_minus_base": summary["means"].get("farthest_ap", float("nan")) - summary["means"].get("base_ap", float("nan")),
            "label_oracle_minus_base": summary["means"].get("label_oracle_ap", float("nan")) - summary["means"].get("base_ap", float("nan")),
        }
    _save_json(summary, os.path.join(out_dir, "view_oracle_summary.json"))

    print(f"\n[DONE] Outputs: {out_dir}")
    print(json.dumps(summary["interpretation_checks"], indent=2))
    if failures:
        print(f"[WARN] {len(failures)} strict-check failures. See view_oracle_summary.json.")
        if VIEW.view_strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
