# RayRoPE mixed-depth confidence study (U0--U4)

## Scope and data policy

This study continues from `exp/moge-rayrope-grasp` and keeps **MoGe off**.
All variants use the exact same image-FPS proposals, RayRoPE support,
CVA view/angle selection, CDF/width decoder, optimizer/epochs, train
scene/frame schedule and evaluation protocol.

Two source domains are used in **equal sample count**:

| Source | RGB | Depth target (full scene) |
|---|---|---|
| GraspNet RealSense (2600 train frames) | Original RealSense RGB | `tsdf_depth/scene_xxxx/realsense/xxxx_depth.png` |
| GN-Trans (2600 paired train frames) | Rendered GN-Trans RGB | `virtual_scenes/scene_xxxx/realsense/xxxx_depth.png` |

GN-Trans **never** substitutes GraspNet TSDF for its background.
RealSense uses the **full** TSDF depth target, rather than the older
rendered-object + TSDF-background mixture. In both streams, depth target
is **supervision only**; the grasp model receives RGB, crop-adjusted K
and camera pose metadata as in the existing RayRoPE protocol.
Non-RGB metadata and sampling/crop dependencies must still be disclosed
when describing the strictness of RGB-only evaluation.

The two targets represent different geometry reconstructions and are
**not assumed pixelwise identical**. Their source-specific masks and
differences are audited by the paired-depth diagnostic.

Training: 10% RS (2600 frames) + 10% GN-Trans (2600 paired frames);
20 epochs, seed 0, 3 GPUs × batch/GPU 3. Validation uses 10% RS Seen
with the same TSDF target. Official grasp evaluation remains on original
GraspNet RealSense test data, three splits × 780 frames; collision on
(primary) and off (diagnostic).

## U0--U4

| Variant | RoPE | Halfwidth h | Supervision |
|---|---|---|---|
| U0 | Point | zero | Ordinary depth L1 (no sigma head) |
| U1 | Expected interval | fixed 20mm | Ordinary depth L1 |
| U2 | Expected interval | learned 1–80mm | 90% central interval score; depth detached |
| U3 | Expected interval | learned 1–80mm | Laplace NLL, depth detached; original L1 retained |
| U4 | Expected interval | learned 1–80mm | Laplace confidence-weighted depth regression replaces L1 |

U2 interval score uses `2h + 20·relu(abs(depth-GT)-h)`.
U3/U4 use the bounded predicted **90% halfwidth** `h`, with
Laplace scale `b = h/log(10)`. The confidence objective is
`b0*(abs(depth-GT)/b + log(b/b0))`, where `b0=20mm/log(10)`.
At h=20mm, U4 metric-depth gradient initially matches plain L1
(after the same original loss weight). **Only U4 changes metric-depth
gradients**; U3 evaluates residuals with a stop-gradient.
Every variant keeps grasp -> numeric geometry/sigma detached.

Use the checkpoint-embedded flags and source fingerprint; an older
checkpoint lacking the mixed-depth contract is not a U0--U4 run.
Checkpoint files and inference outputs should be kept separate by U-ID.

## Commands

Set explicit data roots first:

```bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export GPUS=0,1,2
```

Audit sources first (CSV + JSON, default all 2600 pairs; crop/K audit
for the first 8 pairs):

```bash
P1_VARIANT=U0 PHASES=diagnose bash scripts/run_rayrope_mixed_p1.sh
```

For a short DDP training smoke (partial checkpoints are rejected by
official inference):

```bash
P1_VARIANT=U2 PHASES=train EPOCHS=1 MAX_TRAIN_FRAMES=18 \
  MAX_VAL_FRAMES=9 MAX_STEPS=2 \
  WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_smoke/U2 \
  bash scripts/run_rayrope_mixed_p1.sh
```

Then run one U-ID at a time, reserving the GPU set for that run:

```bash
P1_VARIANT=U0 PHASES=train,infer,eval bash scripts/run_rayrope_mixed_p1.sh
P1_VARIANT=U1 PHASES=train,infer,eval bash scripts/run_rayrope_mixed_p1.sh
P1_VARIANT=U2 PHASES=train,infer,eval bash scripts/run_rayrope_mixed_p1.sh
P1_VARIANT=U3 PHASES=train,infer,eval bash scripts/run_rayrope_mixed_p1.sh
P1_VARIANT=U4 PHASES=train,infer,eval bash scripts/run_rayrope_mixed_p1.sh
```

Use `RESUME=1` only with the exact same experiment signature and
completed checkpoint. Do not silently fall back to fused-depth targets
on missing TSDF/rendered assets. Do not modify an in-progress run's
source or change a data-root path behind an existing checkpoint.

## Diagnostics / acceptance criteria

1. Paired scene/frame identities, source paths and full-depth policy
   must match the report. Compare all/foreground/background validity
   fractions, support IoU, overlap MAE/p90/p99 and crop/K alignment.
2. Geometry supervision and the sigma head must receive finite gradients;
   zero grasp-to-geometry gradient leakage is mandatory for U0--U4.
3. Track depth MAE (all-valid/foreground), empirical h coverage, mean
   halfwidth, upper-bound saturation, CDF ranking metrics and losses.
   A sharp but miscalibrated confidence map is not evidence of reliability.
4. Assert inference is RGB-only; the sensor/TSDF/rendered depth cannot
   appear in the neural network input. External collision filtering and
   crop/workspace dependencies must remain fixed across U-ID variants.
5. Each split: 780 frames, epoch-20 model, seed 0, same collision
   threshold, exact NPY/summary consistency, same inference batch.
6. Do not claim a gain from a single seed is statistically significant.
   U2-vs-U1 tests learned vs fixed intervals; U3-vs-U2 tests
   confidence objective; U4-vs-U3 tests gradient coupling.

## Implementation / verification status

The repository-side code is sufficient to define a full protocol but
still requires CUDA/server smoke, data-path audit and full training.
Do not fill AP fields or claim full integration tests without running
them on the GraspNet server.
