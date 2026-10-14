# Mixed-v2 feasibility: Point RayRoPE with or without GVAR

**Branch:** `exp/moge-rayrope-grasp`. **Scope:** *only* RayRoPE Point (not U3, not MoGe) + GVAR volume_rel. This is a separate matched training suite; it does not modify the archived Point/U3 experiment scripts, their checkpoint source fingerprints, nor pinned `main` files.

## Goal / hypothesis

Does adding an action-conditioned GVAR readout after a Point RayRoPE center–view–angle grouping improve grasp AP, Novel AP and physical quality on *the same* 10% GraspNet RealSense + 10% GN-Trans Mixed-v2 training source? This is a feasibility check, not yet a complete factorial proof of synergy. No pretrained grasp-task checkpoint or GVAR checkpoint warm start.

### Architecture

```text
RGB 448x448 + crop-adjusted K + camera pose
 ├── frozen pretrained DAV2 ViT-B / pose-aware metric depth (global_film)
 │      metric loss from Mixed-v2 GT; numeric geometry detached from grasp loss
 ├── DPT proposal/objectness/graspness; image-FPS 1024 grasp centers
 └── view selection (A1 Top-1)
          |
  GraspRayRoPEGrouping (ray_encoding=point, 7x7, 40px, V/O=on)
          |
  Center–View–Angle features [B, Q, A, hidden]
          ├── rayrope_point: original monotonic CDF/depth-wise width decoder
          └── rayrope_point_gvar:
              action depth adapter → GVAR volume_rel (36 role probes per
              insertion; RGB from pre-geometry feature, relative numeric
              geometry from predicted depth, all E/Q/C numeric-depth detached)
              → SAME CDF threshold labels + width head/physical depth bins
          |
  Native GraspNet decoded grasp groups; sensor-cloud collision ON/OFF
```

The ported GVAR reader and CDF decoder originated from the `exp/gripper-volume-action-reader` implementation (review against its own source; *not* a claim of equivalent checkpoints). The original GVAR's 60mm support envelope is not equal to the final model-predicted grasp width, so this remains evidence reading rather than exact-action geometric evaluation.

## Fixed dataset contract

- Train 100 GraspNet scenes, frames 0,10,...,250, **2,600 RealSense Original RGB + 2,600 GN-Trans RGB**; mixed concat, same pair indices.
- RealSense **GT supervision**: rendered (virtual_scenes) foreground + TSDF background, no fallback on invalid pixels.
- GN-Trans **GT supervision**: corresponding full rendered depth (including background), no TSDF fallback.
- Seen validation: 780 Original RealSense frames; same GT policy. No Similar/Novel AP checkpoint selection.
- The model forward receives **RGB, K, pose and predicted depth**, never observed/GT depth. Original sensor depth may still be used by the GraspNet preprocessing/crop and optional model-free collision filter; do not claim a completely sensor-depth-free pipeline.

## Matched training and inference

| Setting | Value |
|---|---|
| Variants | `rayrope_point`, `rayrope_point_gvar` |
| Initialization | SAME frozen pretrained DAV2 ViT-B, **fresh** depth / task / GVAR heads |
| Epochs/checkpoint | **20 epochs**, final e19 checkpoint; no validation-AP model selection |
| GPUs/batch | **3 GPUs × 3 samples/GPU**, global9, seed0 |
| Optimizer | AdamW LR 3e-4, WD 1e-3, epoch cosine, grad clip 1 |
| Proposal/grouping | image-FPS 1024, top-1 view, RayRoPE Point, 7×7, 40px radius |
| CDF / width | Original 6-threshold CDF, depth-wise width and compact CDF labels |
| GVAR | `volume_rel`, 36 role probes, `reader_dim=64`, heads4 |
| Initial GPU-memory controls | `GROUP_CHUNK=32`, `GVAR_ACTION_CHUNK=256`, `GVAR_ACTIVATION_CHECKPOINT=1` |
| Inference | batch=3, Original test Seen/Similar/Novel 780 frames each, Top-1 view |
| Collision | model-free **on** primary (0.01 threshold, 0.01m voxel, 0.05m approach), **off** diagnostic; same raw Original sensor cloud |

GVAR adds model capacity and reads additional visual evidence; therefore this comparison measures the full proposed addition, not the independent effect of fixed parameter count or one geometric prior.

### Server preflight (mandatory)

Inspect `AGENTS.md`, source SHA/worktrees, GPU occupancy, disk capacity, DAV2 weight hash, data path audit and past experimental processes first. **Do not** `reset`, `stash`, stop other GPU jobs, overwrite outputs, or auto-merge `main`. Create a detached worktree pinned to a commit of the target branch; no in-place changes while training.

```bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export SUITE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rayrope_point_gvar_mixed_v2
# Use 3 actually idle allowed GPUs; GPU4 was excluded by earlier server allocation.
export GPUS=0,1,2

# Full paired dataset audit (2,600 pairs, reader/crop checks):
"$PYTHON_BIN" scripts/diagnose_rayrope_paired_depth.py \
  --dataset-root "$DATASET_ROOT" --gntrans-rgb-root "$GNTRANS_RGB_ROOT" \
  --split train --fraction 0.1 --max-pairs 0 --loader-audit-pairs 8 \
  --output-prefix "$SUITE_ROOT/mixed_v2_depth_audit"

# Independent CPU contracts:
"$PYTHON_BIN" -m pytest -q tests/test_rayrope_point_gvar.py
bash -n scripts/run_rayrope_point_gvar_mixed_v2.sh scripts/run_rayrope_point_gvar_suite.sh
```

### GPU smoke (never report official AP from smoke)

```bash
for v in rayrope_point rayrope_point_gvar; do
  VARIANT="$v" PHASES=train EPOCHS=1 \
    MAX_TRAIN_FRAMES=18 MAX_VAL_FRAMES=9 MAX_STEPS=2 \
    SEEDS=32 GPUS=0 BATCH_SIZE=3 \
    WORK_ROOT="$SUITE_ROOT/smoke/$v" \
    bash scripts/run_rayrope_point_gvar_mixed_v2.sh
done
```

Preflight must confirm finite CDF/width/metric loss, nonzero RayRoPE task gradients and (for combined) nonzero action-adapter/reader gradients, **zero grasp→depth gradients**, and no DAV2 backbone gradients. Both variants must finish train → distributed validation → checkpoint. Then run a full-shape 1024-seed short smoke for the combined model before using 3×3090 full training. If OOM, **first reduce GVAR_ACTION_CHUNK** (e.g., 128) and group chunk uniformly for the whole matched suite; do not silently change effective batch or weights.

### Matched full suite (do not run until audits and smoke pass)

```bash
GPUS=0,1,2 PHASES=train,infer,eval \
  BATCH_SIZE=3 INFER_BATCH_SIZE=3 EPOCHS=20 SEED=0 \
  SUITE_ROOT="$SUITE_ROOT" \
  bash scripts/run_rayrope_point_gvar_suite.sh
```

This runs the two variants **sequentially on the same GPU pool**. No alternative checkpoint initialization or extra GVAR-specific quality loss. Separate result roots are mandatory. The training script records each protocol, sampling SHA, source fingerprint, gradient contract, optimizer/RNG state, all epochs, VRAM peaks and per-split inference manifests; restart only with an unchanged valid checkpoint and `RESUME=1`.

Comparison is automatic after the full suite, or:

```bash
"$PYTHON_BIN" compare_rayrope_gvar.py \
  --root "$SUITE_ROOT" --output "$SUITE_ROOT/comparison" \
  --bootstrap 50000
```

Outputs `absolute.csv`, `paired_effects.csv`, `scene_effects.csv`, `report.md`, `report.json`. The comparator verifies same dataset/sampling, initialization source, training budget, eval schedule, epoch19, inference batch3, finite official `[30,26,50,6]` arrays; evaluates collision-on main AP, off diagnostics and friction μ=.4/.8; uses paired **scene** bootstrap rather than treating 26 frames per scene as IID.

## Decision rules and caveats

1. If Point+GVAR improves Similar/Novel and mean AP under the *same* Mixed-v2 GT, this supports a complementary **action-conditioned readout** after Point RoPE. It does not prove super-additive synergy until the other cells are trained in a matched setup.
2. If there is no reproducible improvement, do not expand to learned U3 or add an uncertainty gate by default. Examine compute cost and stop if the reader adds no task value.
3. Be explicit that the GVAR decoder changes both candidate predictions and rankings; not an exact-action fixed-candidate rerank. One seed and scene-bootstrap confidence intervals do not establish train-seed stability.
4. Full benchmark AP, model memory/latency, code SHA, dataset provenance, historical checkpoint isolation and real robot execution status are separate acceptance criteria.

**Current status:** code prepared; no new formal joint training/official AP claims. All GPU runs are the user's next step.
