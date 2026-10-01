#!/usr/bin/env python3
"""P0 local-action selection x global cross-query scoring 2x2 intervention.

The four conditions are produced from the SAME frozen-source forward:

    lbase_gbase : Base chooses (angle,depth), Base scores that exact action.
    lfull_gbase : Full chooses (angle,depth), Base scores that exact action.
    lbase_gfull : Base chooses (angle,depth), Full scores that exact action.
    lfull_gfull : Full chooses (angle,depth), Full scores that exact action.

Only the local (angle,depth) choice and the scalar score assigned to the chosen
physical action are intervened. Centres, selected views, source-predicted width
map, and all upstream proposals are identical. No cache is generated.

Collision-on/off dumps are derived from the same forward. For conditions sharing
the same local selector, the physical grasp arrays and collision masks are
asserted identical; only the first score column may differ.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


EXPERIMENT_VERSION = "mgf_p0_local_global_2x2_v1"
CONDITIONS = (
    "lbase_gbase",
    "lfull_gbase",
    "lbase_gfull",
    "lfull_gfull",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--full-control-checkpoint", required=True)
    p.add_argument("--dataset-root", default="/data/robotarm/dataset/graspnet")
    p.add_argument("--output-root", required=True)
    p.add_argument(
        "--split",
        choices=("test_seen", "test_similar", "test_novel"),
        required=True,
    )
    p.add_argument("--eval-fraction", type=float, default=0.1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--collision",
        choices=("off", "on", "both"),
        default="both",
    )
    p.add_argument("--collision-thresh", type=float, default=0.01)
    p.add_argument("--voxel-size", type=float, default=0.01)
    p.add_argument("--approach-dist", type=float, default=0.05)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p


def code_digest():
    from mgf_p0_online import code_digest as p0_digest

    h = hashlib.sha256(p0_digest().encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def _decode_local_global(
    end_points,
    local_logits,
    global_logits,
    rotation_fn,
    max_width,
):
    """Decode local action from one scorer and rank it with another.

    Args:
        local_logits:  [B,T,Q,A,D], used ONLY for argmax over A x D.
        global_logits: [B,T,Q,A,D], evaluated at the local-selected A,D and
                       written as grasp score for cross-query/global ranking.

    The physical width is always read from the frozen source width tensor at
    the LOCAL-selected (angle,depth). No alternate width head is introduced.
    """
    from models.economicgrasp_bip3d import _cva_decode_query_indices

    if local_logits.shape != global_logits.shape or local_logits.ndim != 5:
        raise ValueError(
            "Expected shape-matched local/global CDF logits [B,T,Q,A,D], got "
            f"{tuple(local_logits.shape)} vs {tuple(global_logits.shape)}"
        )
    if not bool(torch.isfinite(local_logits).all()):
        raise FloatingPointError("Non-finite local logits")
    if not bool(torch.isfinite(global_logits).all()):
        raise FloatingPointError("Non-finite global logits")

    centers = end_points["xyz_graspable"]
    views = end_points["grasp_top_view_xyz"]
    widths = end_points["grasp_width_pred_angle_depth"]
    if centers.ndim != 3 or centers.shape[-1] != 3:
        raise ValueError(f"Malformed centers: {tuple(centers.shape)}")
    if views.shape != centers.shape:
        raise ValueError("View/center shape mismatch")

    B, T, Q, A, D = local_logits.shape
    if centers.shape[:2] != (B, Q):
        raise ValueError("Logit/center query mismatch")
    if widths.shape != (B, D, Q, A):
        raise ValueError(
            "Frozen width tensor must be [B,D,Q,A], got "
            f"{tuple(widths.shape)} expected {(B,D,Q,A)}"
        )

    local_u = torch.sigmoid(local_logits.float()).mean(dim=1)
    global_u = torch.sigmoid(global_logits.float()).mean(dim=1)
    outputs = []

    for bi in range(B):
        query_idx = _cva_decode_query_indices(
            end_points=end_points,
            batch_i=bi,
            total_q=Q,
            use_top4_view=False,
        )
        qn = int(query_idx.numel())
        row = torch.arange(qn, device=query_idx.device)

        lu = local_u[bi].index_select(0, query_idx)
        gu = global_u[bi].index_select(0, query_idx)
        joint = lu.reshape(qn, A * D).argmax(dim=-1)
        angle_idx = torch.div(joint, D, rounding_mode="floor")
        depth_idx = torch.remainder(joint, D)

        # Global score is evaluated at the EXACT local-selected action.
        global_flat = gu.reshape(qn, A * D)
        score = global_flat[row, joint].unsqueeze(-1)

        width_qad = widths[bi].float().index_select(1, query_idx)
        width_qad = width_qad.permute(1, 2, 0).contiguous()
        width = width_qad[row, angle_idx, depth_idx].unsqueeze(-1)
        width = torch.clamp(1.2 * width / 10.0, min=0.0, max=float(max_width))

        center = centers[bi].float().index_select(0, query_idx)
        approach = -views[bi].float().index_select(0, query_idx)
        angle = angle_idx.float() * (np.pi / float(A))
        rot = rotation_fn(approach, angle).reshape(qn, 9)
        depth = (depth_idx.float().unsqueeze(-1) + 1.0) * 0.01
        height = torch.full_like(score, 0.02)
        obj = torch.full_like(score, -1.0)

        pred = torch.cat(
            (score, width, height, depth, rot, center, obj),
            dim=-1,
        )
        if pred.shape != (qn, 17) or not bool(torch.isfinite(pred).all()):
            raise RuntimeError(f"Invalid hybrid decode: {tuple(pred.shape)}")
        outputs.append(pred)
    return outputs


def _assert_endpoints_match_reference(
    ep,
    base_logits,
    full_logits,
    decoded,
    rotation_fn,
):
    """Endpoint contract: BB/FF must reproduce native decodes exactly enough."""
    from models.economicgrasp_bip3d import pred_decode_center_view_angle

    e = dict(ep)
    e["grasp_cdf_pred_angle_depth"] = base_logits
    ref_base = pred_decode_center_view_angle(
        e,
        use_cdf=True,
        batch_viewpoint_params_to_matrix_fn=rotation_fn,
    )
    e["grasp_cdf_pred_angle_depth"] = full_logits
    ref_full = pred_decode_center_view_angle(
        e,
        use_cdf=True,
        batch_viewpoint_params_to_matrix_fn=rotation_fn,
    )
    for i in range(len(ref_base)):
        torch.testing.assert_close(
            decoded["lbase_gbase"][i],
            ref_base[i],
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            decoded["lfull_gfull"][i],
            ref_full[i],
            atol=2e-6,
            rtol=2e-6,
        )
        # Same local selector => EXACT same physical action. Only score may differ.
        torch.testing.assert_close(
            decoded["lbase_gbase"][i][:, 1:],
            decoded["lbase_gfull"][i][:, 1:],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            decoded["lfull_gbase"][i][:, 1:],
            decoded["lfull_gfull"][i][:, 1:],
            atol=0,
            rtol=0,
        )


@torch.no_grad()
def main():
    a = parser().parse_args()
    if (
        a.num_shards < 1
        or not 0 <= a.shard_id < a.num_shards
        or a.batch_size < 1
        or a.workers < 0
        or a.max_frames < 0
    ):
        raise ValueError("Invalid inference counts")
    if not 0 < a.eval_fraction <= 1:
        raise ValueError("--eval-fraction must lie in (0,1]")
    if not all(
        np.isfinite(x) and x > 0
        for x in (a.collision_thresh, a.voxel_size, a.approach_dist)
    ):
        raise ValueError("Collision parameters must be finite and positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GraspNet inference environment required")

    from mgf_p0_online import FrozenSource, load_control, safe_batch
    from metric_field_runtime import (
        VERSION,
        make_dataset,
        dataset_schedule,
        ensure_manifest,
        digest,
        sha256_file,
        atomic_json,
        seed_all,
        worker_init,
    )
    from inference_metric_grasp_field import atomic_npy
    from dataset.graspnet_dataset import collate_fn

    seed_all(0)
    source = FrozenSource(a.source_checkpoint, "cuda:0")
    full, full_ck = load_control(source, a.full_control_checkpoint)
    if full.variant != "full":
        raise ValueError(
            "--full-control-checkpoint must be the P0-1 full control, got "
            f"{full.variant!r}"
        )
    full_protocol = full_ck["protocol"]
    full_epoch = int(full_ck["epoch"])
    if bool(full_protocol.get("partial_run", True)):
        raise ValueError("Formal 2x2 inference rejects partial Full training")
    if abs(float(full_protocol["eval_fraction"]) - float(a.eval_fraction)) > 1e-12:
        raise ValueError("Evaluation fraction differs from Full-control protocol")
    full.eval()

    dataset, _, indices = make_dataset(
        a.dataset_root,
        a.split,
        a.eval_fraction,
        labels=False,
        max_frames=a.max_frames,
    )
    schedule = dataset_schedule(dataset, indices)
    modes = ["off", "on"] if a.collision == "both" else [a.collision]

    out_root = Path(a.output_root)
    roots = {
        (cond, mode): out_root / cond / f"test_collision_{mode}"
        for cond in CONDITIONS
        for mode in modes
    }

    protocols = {}
    locks = []
    import fcntl

    for cond in CONDITIONS:
        local_source = "full" if cond.startswith("lfull") else "base"
        global_source = "full" if cond.endswith("gfull") else "base"
        for mode in modes:
            root = roots[(cond, mode)]
            training = copy.deepcopy(source.protocol)
            training["eval_fraction"] = a.eval_fraction
            training["partial_run"] = bool(
                a.max_frames or full_protocol.get("partial_run", False)
            )
            training["p0_full_finetuning"] = full_protocol
            collision = (
                "none"
                if mode == "off"
                else dict(
                    type="model_free_original_sensor",
                    threshold=a.collision_thresh,
                    voxel_size_m=a.voxel_size,
                    approach_dist_m=a.approach_dist,
                    network_input=False,
                )
            )
            protocol = dict(
                version=VERSION,
                experiment_version=EXPERIMENT_VERSION,
                split=a.split,
                schedule=schedule,
                code_sha256=code_digest(),
                checkpoint_sha256=source.checkpoint_sha,
                checkpoint_epoch=source.epoch,
                source_checkpoint_epoch=source.epoch,
                full_control_epoch=full_epoch,
                full_control_sha256=sha256_file(a.full_control_checkpoint),
                training_protocol=training,
                score_source=cond,
                local_action_source=local_source,
                global_score_source=global_source,
                intervention=(
                    "Local scorer selects joint (angle,depth); global scorer "
                    "scores that exact selected physical action for cross-query ranking. "
                    "Centre/view/source width map are frozen and shared."
                ),
                collision_filter=collision,
                primary_collision="on",
                max_frames=a.max_frames,
                evaluation_fraction=a.eval_fraction,
                batch_size=a.batch_size,
                num_shards=a.num_shards,
                seed=0,
                preprocessing=(
                    "Original GraspNet crop/workspace; RGB-only network input; "
                    "sensor cloud used only by optional collision post-filter"
                ),
            )
            ensure_manifest(root / a.split / "protocol.json", protocol)
            lock = open(root / a.split / f".shard_{a.shard_id}.lock", "a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
            protocols[(cond, mode)] = (protocol, digest(protocol))

    def paths(root, sid, ann):
        return (
            root / "dump" / f"scene_{sid:04d}" / "realsense" / f"{ann:04d}.npy",
            root / a.split / "completed" / f"{sid:04d}_{ann:04d}.json",
        )

    pending = []
    for pos, (idx, (sid, ann)) in enumerate(zip(indices, schedule)):
        if pos % a.num_shards != a.shard_id:
            continue
        complete = True
        for cond in CONDITIONS:
            for mode in modes:
                path, mark = paths(roots[(cond, mode)], sid, ann)
                if not a.resume and (path.exists() or mark.exists()):
                    raise FileExistsError(path)
                ok = False
                if mark.is_file():
                    row = json.loads(mark.read_text())
                    if row["signature"] != protocols[(cond, mode)][1]:
                        raise RuntimeError(f"Stale marker {mark}")
                    ok = (
                        path.is_file()
                        and sha256_file(path) == row["output_sha256"]
                    )
                complete &= ok
        if not complete:
            pending.append(idx)

    loader = DataLoader(
        Subset(dataset, pending),
        batch_size=a.batch_size,
        num_workers=a.workers,
        collate_fn=collate_fn,
        shuffle=False,
        worker_init_fn=worker_init,
    )

    if "on" in modes:
        from graspnetAPI import GraspGroup
        from utils.collision_detector import ModelFreeCollisionDetectorTorch

    verified = False
    cursor = 0
    for raw in loader:
        ctx = source(safe_batch(raw, "cuda:0", False))
        ep = ctx["ep"]
        base_logits = ep["grasp_cdf_pred_angle_depth"].detach()
        full_logits = full(ctx).detach()

        decoded = {
            "lbase_gbase": _decode_local_global(
                ep,
                base_logits,
                base_logits,
                source.model.rotation_fn,
                source.model.max_width,
            ),
            "lfull_gbase": _decode_local_global(
                ep,
                full_logits,
                base_logits,
                source.model.rotation_fn,
                source.model.max_width,
            ),
            "lbase_gfull": _decode_local_global(
                ep,
                base_logits,
                full_logits,
                source.model.rotation_fn,
                source.model.max_width,
            ),
            "lfull_gfull": _decode_local_global(
                ep,
                full_logits,
                full_logits,
                source.model.rotation_fn,
                source.model.max_width,
            ),
        }
        if not verified:
            _assert_endpoints_match_reference(
                ep,
                base_logits,
                full_logits,
                decoded,
                source.model.rotation_fn,
            )
            verified = True

        batch_n = len(decoded["lbase_gbase"])
        for bi in range(batch_n):
            idx = pending[cursor]
            sid, ann = dataset_schedule(dataset, [idx])[0]
            cursor += 1

            arrays_off = {
                cond: decoded[cond][bi].float().cpu().numpy()
                for cond in CONDITIONS
            }
            for cond, arr in arrays_off.items():
                if arr.ndim != 2 or arr.shape[1] != 17 or not np.isfinite(arr).all():
                    raise RuntimeError(f"Invalid {cond} decoded grasps")

            # Physical-action invariants: same local selector => same action.
            if not np.array_equal(
                arrays_off["lbase_gbase"][:, 1:],
                arrays_off["lbase_gfull"][:, 1:],
            ):
                raise RuntimeError("Base-local physical actions differ by global scorer")
            if not np.array_equal(
                arrays_off["lfull_gbase"][:, 1:],
                arrays_off["lfull_gfull"][:, 1:],
            ):
                raise RuntimeError("Full-local physical actions differ by global scorer")

            arrays_by_mode = {"off": arrays_off}
            if "on" in modes:
                cloud, _ = dataset.get_data(idx, return_raw_cloud=True)
                detector = ModelFreeCollisionDetectorTorch(
                    np.asarray(cloud, np.float32).reshape(-1, 3),
                    voxel_size=a.voxel_size,
                )
                # Compute one mask per PHYSICAL action family, then reuse it for
                # the alternate global score. This is stricter than four
                # independent detector calls.
                gg_base = GraspGroup(arrays_off["lbase_gbase"])
                coll_base = detector.detect(
                    gg_base,
                    approach_dist=a.approach_dist,
                    collision_thresh=a.collision_thresh,
                ).cpu().numpy()
                gg_full = GraspGroup(arrays_off["lfull_gbase"])
                coll_full = detector.detect(
                    gg_full,
                    approach_dist=a.approach_dist,
                    collision_thresh=a.collision_thresh,
                ).cpu().numpy()
                arrays_by_mode["on"] = {
                    "lbase_gbase": arrays_off["lbase_gbase"][~coll_base],
                    "lbase_gfull": arrays_off["lbase_gfull"][~coll_base],
                    "lfull_gbase": arrays_off["lfull_gbase"][~coll_full],
                    "lfull_gfull": arrays_off["lfull_gfull"][~coll_full],
                }

            for mode in modes:
                arrays = arrays_by_mode[mode]
                if len(arrays["lbase_gbase"]) != len(arrays["lbase_gfull"]):
                    raise RuntimeError("Collision count differs within Base-local pair")
                if len(arrays["lfull_gbase"]) != len(arrays["lfull_gfull"]):
                    raise RuntimeError("Collision count differs within Full-local pair")

                for cond in CONDITIONS:
                    root = roots[(cond, mode)]
                    path, mark = paths(root, sid, ann)
                    arr = arrays[cond].astype(np.float32, copy=False)
                    if a.resume and mark.is_file() and path.is_file():
                        saved = json.loads(mark.read_text())
                        if (
                            saved["signature"] == protocols[(cond, mode)][1]
                            and sha256_file(path) == saved["output_sha256"]
                        ):
                            prior = np.load(path, allow_pickle=False)
                            if prior.shape != arr.shape or not np.allclose(
                                prior, arr, atol=1e-6, rtol=1e-6
                            ):
                                raise RuntimeError(
                                    "Resume re-forward changed an existing 2x2 dump; "
                                    "use a fresh output root"
                                )
                            continue

                    atomic_npy(path, arr)
                    atomic_json(
                        mark,
                        dict(
                            signature=protocols[(cond, mode)][1],
                            output_sha256=sha256_file(path),
                            grasps=len(arr),
                            grasps_before_collision=len(arrays_off[cond]),
                            grasps_after_collision=len(arr),
                            local_action_source=protocols[(cond, mode)][0][
                                "local_action_source"
                            ],
                            global_score_source=protocols[(cond, mode)][0][
                                "global_score_source"
                            ],
                        ),
                    )
        del ctx, ep, base_logits, full_logits, decoded

    for cond in CONDITIONS:
        for mode in modes:
            root = roots[(cond, mode)]
            atomic_json(
                root / a.split / f"shard_{a.shard_id}.json",
                dict(
                    signature=protocols[(cond, mode)][1],
                    written=cursor,
                    endpoint_equivalence_verified=verified,
                ),
            )


if __name__ == "__main__":
    main()
