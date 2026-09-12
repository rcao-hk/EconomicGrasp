# P5 — Geometry-Error-Aware Grasp Repair

Branch: `exp/p5-geometry-aware-grasp-repair`

P5 implements two lessons from P1–P4 without inheriting their experiment code.
It starts from the controlled Stage-1 RGB checkpoint on `main` and keeps the
entire Stage-1 model frozen.

## Scientific question

Can a small RGB repair module recover a physically coherent grasp center when
monocular metric geometry is wrong, if training (1) exposes deployment-like
geometry errors and (2) supplies gripper-centric and neighboring-ray evidence?

P5 deliberately does **not** treat absence of a nearby annotation as grasp
failure.  It also does not change the Stage-1 view, angle, insertion depth,
width or score in v1.  This makes translation repair the only deployment
intervention.

## Route 1 — geometry-error-aware repair

### Structured training corruption

Only during `model.train()`, the native Stage-1 predicted dense depth is
corrupted by a spatially correlated field:

```
D_tilde = clamp(D_pred * (1 + scale) + scene_bias + smooth_region_bias)
```

Defaults:

- corruption probability: 0.8
- scene additive sigma: 12 mm
- multiplicative scale sigma: 0.025
- smooth regional sigma: 15 mm
- regional latent grid: 7 x 7

The repair module never receives the uncorrupted depth as an input.  The same
corrupted dense map is used to:

1. backproject the sparse proposal center;
2. rebuild the frozen geometry-conditioned feature map through the existing
   spatial enhancer;
3. compute gripper-keypoint depth residuals.

Validation and inference use the native RGB-predicted depth with **no synthetic
corruption**.

### Label semantics

For a corrupted proposal, `process_grasp_labels_cdf_width` is used to obtain its
nearest annotated grasp point and the cached operation labels associated with
that point.  Importantly, the original 5-mm matcher validity is **not** reused as
"success/failure".

A center-repair target is supervised only if:

1. the nearest annotated grasp center lies within `P5_TARGET_RADIUS_M` (default
   60 mm), and
2. the exact frozen Stage-1 `(predicted view, predicted angle, predicted
   insertion-depth)` operation has a positive CDF label at that annotated
   center.

The target is then the vector from the corrupted proposal center to that
annotated center, represented in the current grasp coordinate frame.

Every other query is **unknown**.  It is not assigned a zero grasp-quality
label.  Unknown queries receive only a weak identity regularizer on the
predicted repair magnitude; default weight is 0.02.

Thus P5 does not assume:

```
no annotation within 5 mm == grasp failure
```

## Route 2 — gripper-centric evidence and local cross-ray consistency

For each native Stage-1 grasp, P5 builds 11 physical keypoints spanning:

- grasp center;
- left/right inner finger tips;
- left/right outer finger tips;
- left/right finger bases;
- palm/back point;
- forward approach point;
- upper/lower contact-height points.

At each projected keypoint, P5 samples:

- RGB values;
- the pre-geometry DPT proposal feature captured by a forward hook;
- a frozen geometry-conditioned feature recomputed from `D_tilde`;
- signed dense-depth residual relative to the physical keypoint;
- gripper-local xyz;
- visibility.

A small Transformer aggregates the 11 keypoint tokens into one grasp-evidence
vector.

P5 then constructs an 8-neighbor image-space graph over the 1024 image-FPS
queries.  Multi-head attention exchanges evidence among nearby rays while a
relative `(du,dv,dz)` embedding describes local geometric consistency.  This is
intended to expose regional depth-bias evidence without imposing a hard
smoothness constraint across object boundaries.

The final repair head predicts a bounded 3-D residual in the grasp frame.  Its
last layer is zero initialized, so a fresh P5 model starts as the identity
mapping.

## Frozen/deployable contract

Stage-1 outputs kept unchanged in P5-v1:

- image-FPS query ownership;
- Top-1 approach view;
- in-plane angle;
- insertion-depth class;
- width;
- raw CDF score.

Only translation is replaced by the P5 repaired center at inference.  The
`native` inference mode is provided as an internal control.

No clean depth, CDF cache or analytic evaluator enters network inference.
Collision filtering remains the existing captured-cloud post-processing when
`collision_thresh > 0`.

## Fresh training

Use the canonical controlled Stage-1 checkpoint, not P1/P2/P3/P4:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
INIT_CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/p5_repair_10pct \
TRAIN_SAMPLE_INTERVAL=0.1 \
EVAL_SAMPLE_INTERVAL=0.1 \
BATCH_SIZE=1 \
MAX_EPOCH=20 \
bash run_p5_train.sh
```

Smoke first:

```bash
GPUS=0 MAX_EPOCH=1 P5_MAX_BATCHES=2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/p5_repair_smoke \
DATASET_ROOT=/data/robotarm/dataset/graspnet \
INIT_CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
bash run_p5_train.sh
```

Training outputs:

- `log_train_p5.txt`
- `p5_epochs.jsonl`
- `p5_training_protocol.json`
- `checkpoint_latest.tar`
- `checkpoint_best_distance.tar`
- `checkpoint_best_5mm.tar`

Primary held-out diagnostics:

- `p5_target_known_ratio`
- `p5_native_target_dist_m`
- `p5_repaired_target_dist_m`
- `p5_repair_improvement_m`
- `p5_native_within5mm`
- `p5_repaired_within5mm`
- `p5_pred_delta_abs_m`
- `p5_target_delta_abs_m`
- train-only query corruption magnitude

The first mechanism gate is strict: validation must improve repaired-target
distance and/or within-5mm rate over the native proposal.  Training-only gains
are not enough.

## Inference / AP

Primary repaired run:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/p5_repair_10pct/checkpoint_best_distance.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/p5_repair_eval/repair \
SAMPLE_INTERVAL=0.1 \
MODE=repair \
COLLISION_THRESH=0.01 \
RUN_EVAL=1 \
bash run_p5_inference.sh
```

Internal native control using the same checkpoint:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/p5_repair_10pct/checkpoint_best_distance.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/p5_repair_eval/native \
SAMPLE_INTERVAL=0.1 \
MODE=native \
COLLISION_THRESH=0.01 \
RUN_EVAL=1 \
bash run_p5_inference.sh
```

`repair - native` isolates the effect of translation repair because P5-v1 keeps
all other decoded grasp fields unchanged.

## Interpretation

A useful P5 result requires both levels of evidence:

1. held-out repair diagnostics improve on naturally erroneous predicted depth;
2. official GraspNet AP improves relative to the internal native control,
   especially Similar/Novel.

If training repair improves but held-out repair does not, the synthetic error
model still fails to approximate deployment error.  If held-out center repair
improves but AP does not, translation is being repaired toward annotation-space
centers without improving executable grasp utility; the next change should be
a full-action repair target rather than a larger P5 network.
