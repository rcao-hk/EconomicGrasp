# Ray-conditioned grasp confidence calibration

Branch: `exp/p4-ray-confidence-calibration`  
Base: `main@3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2`

## Motivation

The P3 B/C/D decomposition isolated the strongest positive mechanism:

- Stage-1 / zero-center baseline: mean AP 46.095
- P3 B (cross-depth aggregation, zero center, raw ranking): 45.791
- P3 C (cross-depth aggregation, joint center selection, raw ranking): 45.826
- P3 D (same pose selection, joint viability ranking): 46.462

Thus the main gain came from the viability-conditioned **final ranking score**, not
from changing the physical center or from contextualizing the CDF representation
itself. This experiment tests that mechanism in isolation.

## Controlled intervention

The complete controlled Stage-1 RGB grasp model is frozen. The native Stage-1
pipeline determines, exactly as before:

1. image-FPS query pixel;
2. predicted metric center;
3. Top-1 approach view;
4. CDF in-plane angle and insertion depth;
5. grasp width;
6. native raw CDF score.

Additional camera-z hypotheses

`[-40, -20, -10, 0, +10, +20, +40] mm`

are used only to extract frozen local grouping features for the same query/view.
No alternate center is decoded. No contextual feature is fed back to the CDF or
width decoder.

For the already selected native angle, K local evidence tokens are compared by a
small per-ray Transformer. Its only output is one scalar confidence gate:

`gate = sigmoid(confidence_logit)`

and the deployable intervention is

`score_calibrated = score_Stage1 * gate`.

Therefore calibrated and raw modes have identical grasp translation, rotation,
insertion depth, and width. Only score/ranking may differ.

## Training target

The target refers to the **unchanged Stage-1 decoded operation at the zero
center**. The existing compact CDF cache supplies evaluator-aligned utility.

- zero center known outside the 5-mm label-point support: target score = 0;
- zero center supported and selected `(angle, insertion-depth)` has a valid CDF
  label: target score = its compact CDF utility;
- supported center but selected operation/view is unlabelled: unknown, ignored.

No GraspNet/Dex-Net analytic evaluator is called during training.

Training objective:

1. balanced positive/zero BCE between calibrated score and target utility;
2. frame-wise listwise ranking CE among known native Stage-1 grasps.

Default weights are 1.0 + 1.0. The confidence head starts near identity with
output bias 4.0 (`sigmoid(4) ~= 0.982`), so a fresh model initially perturbs
Stage-1 ranking only slightly.

## Fresh training

Use the canonical controlled Stage-1 checkpoint, not P1/P2/P3:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
INIT_CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/ray_confidence_10pct \
TRAIN_SAMPLE_INTERVAL=0.1 \
EVAL_SAMPLE_INTERVAL=0.1 \
BATCH_SIZE=1 \
MAX_EPOCH=20 \
bash run_ray_confidence_train.sh
```

Smoke test:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
INIT_CKPT=/path/to/stage1_e15.tar \
GPUS=0 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/ray_confidence_smoke \
MAX_EPOCH=1 RC_MAX_BATCHES=2 \
bash run_ray_confidence_train.sh
```

Smoke checkpoints are rejected by official inference.

Outputs:

- `log_train_rc.txt`
- `rc_epochs.jsonl`
- `rc_training_protocol.json`
- `checkpoint_latest.tar`
- `checkpoint_best_rank.tar`
- `checkpoint_best_top10.tar`

Primary validation diagnostics:

- `rc_raw_top{1,10,50}_target`
- `rc_cal_top{1,10,50}_target`
- `rc_raw_bce` / `rc_calibrated_bce`
- `rc_gate_mean`, `rc_gate_positive`, `rc_gate_negative`
- `rc_zero_point_support`

`checkpoint_best_top10.tar` is the most AP-oriented checkpoint; the rank-loss
checkpoint is retained as a less task-specific control.

## Inference

Primary calibrated result:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/ray_confidence_10pct/checkpoint_best_top10.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_confidence_eval/calibrated \
SAMPLE_INTERVAL=0.1 \
SCORE_MODE=calibrated \
COLLISION_THRESH=0.01 \
RUN_EVAL=1 \
bash run_ray_confidence_inference.sh
```

Exact internal control using the same checkpoint and frozen Stage-1 pose:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/ray_confidence_10pct/checkpoint_best_top10.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_confidence_eval/raw \
SAMPLE_INTERVAL=0.1 \
SCORE_MODE=raw \
COLLISION_THRESH=0.01 \
RUN_EVAL=1 \
bash run_ray_confidence_inference.sh
```

`raw` and `calibrated` must produce identical grasp poses before collision
filtering. Only the score column differs. This is the decisive causal control.

## Interpretation

- `calibrated > raw`: multi-depth evidence is useful as grasp-confidence
  information even when metric center and grasp pose are untouched.
- `calibrated ~= raw`: P3 D-C likely depended on coupling between P3's joint
  field and its viability score rather than a transferable pure ranking signal.
- `calibrated < raw`: confidence supervision/ranking objective does not
  generalize; do not reintroduce center-selection complexity to hide the result.
