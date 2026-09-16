# Pairwise exact-action center selector

Branch: `exp/ray-bestofk-exact-action-diagnostic`

## Purpose

The frozen K=7 exact-action diagnostic established two facts:

1. actual decoded multi-center grasps have substantial best-of-K headroom;
2. the deployed raw CDF score does not reliably rank centers and can damage a
   successful native grasp.

This experiment asks the shortest remaining learnability question:

> Does the frozen Stage-1 center-conditioned representation contain enough
> information to predict whether an alternative physical center is better than
> the native zero-offset center?

The grasp network stays frozen.  Only a small pairwise selector MLP is trained.

## Data protocol and leakage control

Exact CAD/DexNet labels are mined only on the GraspNet **train** split.  With the
default protocol:

- scenes `0--79`: selector training;
- scenes `80--99`: selector validation and native-fallback threshold selection;
- `test_seen`, `test_similar`, `test_novel`: held-out testing only.

The analytic evaluator is not inside the selector training loop.  Cache mining
first freezes each physical action and stores its exact-action label.  Training
then reads only those cached labels and frozen features.

At test time the selector receives no CAD model, GT depth, collision label or
force-closure label.  Exact evaluation is performed only after the selector has
chosen a center.

## K-center representation

For every selected native image-FPS ray, use the historical grid

`[-40,-20,-10,0,+10,+20,+40] mm`.

For each center the corrected reread path keeps read center and emitted action
center coherent, freezes the Stage-1 approach view, and exports:

- local grouping feature at the CDF-selected in-plane angle;
- local grouping feature averaged across in-plane angles;
- deployed raw CDF score.

For candidate `k` relative to native candidate `0`, the selector input is

`[F0_sel, Fk_sel, Fk_sel-F0_sel, F0_mean, Fk_mean-F0_mean,
  score0, scorek, scorek-score0, normalized_offset]`.

This is intentionally a relative representation.  The selector predicts

`Delta U_k = U_exact(g_k) - U_exact(g_0)`.

The native candidate has defined relative score zero.

## Training objective

The lightweight MLP is trained with three complementary terms:

1. Smooth-L1 regression to exact utility difference `Delta U`;
2. balanced beneficial-vs-harmful BCE for non-tied candidates;
3. listwise cross-center CE using the exact K-way utility distribution, only on
   queries whose utility actually changes across centers.

Defaults:

`L = 1.0 L_reg + 0.5 L_sign + 0.5 L_list`.

After every epoch, validation searches a conservative native-fallback threshold.
The primary threshold objective is validation exact utility; ties prefer lower
Success@0.8 harm and then fewer center changes.  This threshold is stored in the
selector checkpoint and is fixed for all test splits.

## Files

- `mine_ray_pairwise_exact_cache.py`: frozen feature + exact-action train cache;
- `utils/ray_pairwise_selector.py`: feature contract, MLP, losses, fallback;
- `train_ray_pairwise_selector.py`: train/validation and threshold selection;
- `test_ray_pairwise_selector.py`: held-out exact-action test;
- `run_ray_pairwise_selector_train.sh`: multi-GPU mining + selector training;
- `run_ray_pairwise_selector_test.sh`: multi-split testing;
- `tests/test_ray_pairwise_selector.py`: CPU unit tests.

## Unit tests

```bash
pytest -q \
  tests/test_ray_pairwise_selector.py \
  tests/test_ray_bestofk_diagnostic.py \
  tests/test_cva_center_decoupling.py
```

## Training smoke

Use one GPU and a small number of frames per shard.  The cache must still contain
both a training scene (<80) and a validation scene (>=80), so for a practical
smoke it is usually easiest to mine a small 10% cache first rather than setting a
very small global `MINE_MAX_SAMPLES`.

A short selector-only smoke on an existing cache is:

```bash
PHASES=train \
CACHE_ROOT=/path/to/cache_train \
TRAIN_OUT=/tmp/ray_pairwise_selector_smoke \
EPOCHS=2 \
MAX_TRAIN_FRAMES=20 \
MAX_VAL_FRAMES=10 \
TRAIN_GPU=0 \
bash run_ray_pairwise_selector_train.sh
```

## Main training

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
STAGE1_CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector \
MINE_GPUS=0,1,2,3,4,5 \
MINE_SAMPLE_INTERVAL=0.1 \
MINE_QUERY_EVAL_NUM=64 \
OFFSETS_MM=-40,-20,-10,0,10,20,40 \
VAL_SCENE_START=80 \
EPOCHS=20 \
TRAIN_GPU=0 \
bash run_ray_pairwise_selector_train.sh
```

The expensive phase is exact-action cache mining.  It is resumable by default:
existing `scene_xxxx/ann_xxxx.npz` files are skipped unless
`MINE_OVERWRITE=1`.

After mining is complete, retraining selector variants does not call the analytic
evaluator:

```bash
PHASES=train bash run_ray_pairwise_selector_train.sh
```

## Held-out testing

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
STAGE1_CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector \
SELECTOR_CKPT=/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector/train/checkpoint_best.tar \
GPUS=0,1,2 \
SPLITS=test_seen,test_similar,test_novel \
SAMPLE_INTERVAL=0.1 \
QUERY_EVAL_NUM=128 \
QUERY_EVAL_MODE=topk_uniform \
OFFSETS_MM=-40,-20,-10,0,10,20,40 \
bash run_ray_pairwise_selector_test.sh
```

Use the same offset grid at cache mining and testing.  Do not tune
`SELECTOR_THRESHOLD` on test results; by default the validation-selected value in
`checkpoint_best.tar` is used.

## Primary test comparison

Every test query reports four policies:

- `native`: zero-offset Stage-1 action;
- `raw`: raw CDF cross-center selection;
- `learned`: pairwise selector with native fallback;
- `oracle`: post-hoc exact best-of-K upper bound.

Primary quantities are:

- `learned - native`: deployable relative-selection gain;
- `learned rescue08` and `learned harm08`: whether gain comes with acceptable
  no-harm behavior;
- `oracle - learned`: remaining selection/representation headroom;
- learned vs raw: whether actual-action supervision fixes the cross-center score
  mismatch identified by the frozen diagnostic.

## Interpretation gate

- **Learned > native and learned > raw on Similar/Novel with lower harm**:
  frozen reread features contain usable cross-center validity information;
  selective center correction is supported.
- **Validation improves but held-out tests do not**:
  the representation/label relation does not generalize; do not increase MLP
  capacity as the first response.
- **Learned remains near native while oracle is much higher**:
  frozen local evidence is insufficient for robust relative center selection;
  representation design, not selector capacity, is the next target.
- **Learned gains only by high intervention/harm rate**:
  no-harm calibration/fallback remains a deployment bottleneck.
