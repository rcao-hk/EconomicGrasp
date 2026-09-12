# P5-v1.1 — Repair to a Valid Center Set

P5-v1.1 is a training-protocol correction of P5-v1. The inference architecture is unchanged: RGB-only Stage-1 is frozen, the P5 gripper-centric/cross-ray module predicts a translation residual, and only grasp translation is modified at decode.

## Motivation

P5-v1 showed a clear mismatch:

- synthetic-corruption train: larger predicted repair correlated with better target distance;
- held-out native validation: larger predicted repair correlated with worse target distance;
- proposals already within the 5-mm label-valid region were still trained to snap exactly to the nearest annotation center.

P5-v1.1 therefore changes three things before any further architecture tuning.

## 1. Repair-to-valid-set supervision

Let `c*` be the coherent nearest annotated center for the frozen Stage-1 `(view, angle, insertion-depth)` operation and `d = ||c-c*||`.

With `r_safe = 5 mm` and `r_repair = 60 mm`:

- `d <= r_safe`: the proposal is already in the valid center set; target residual is **zero**;
- `r_safe < d <= r_repair` and the same frozen operation has a positive CDF label: repair toward `c*`;
- otherwise: **unknown**, never a grasp-quality negative; only weak identity regularization is applied.

The primary mechanism metric is distance outside the valid set:

`violation(c) = max(||c-c*|| - r_safe, 0)`.

## 2. Balanced native/corrupted training

Training alternates batches deterministically:

- even optimization step: native predicted geometry;
- odd optimization step: structured corruption active with probability 1.

Thus the deployment condition is exactly 50% of optimization batches instead of a minority under the previous 0.8 corruption probability.

The corruption model itself is unchanged:

- scene bias sigma 12 mm;
- scale sigma 2.5%;
- smooth regional bias sigma 15 mm on a 7x7 latent grid.

## 3. Dual validation

Every epoch evaluates the same sampled Seen validation frames twice:

1. **native validation** — exact deployment geometry, no synthetic corruption;
2. **deterministic corrupt validation** — corruption forced on, with RNG reset to the same fixed seed each epoch.

This separates:

- failure to generalize synthetic repair across scenes, from
- failure to transfer synthetic repair to the native deployment-error distribution.

## Checkpoint policy

`checkpoint_best_native.tar` is created **only if** both conditions hold:

1. repaired mean set violation is lower than the native-proposal baseline;
2. repaired within-safe ratio is not lower than the native-proposal baseline.

Therefore a run in which every checkpoint harms native validation will not produce a misleading `best_native` checkpoint.

`checkpoint_best_corrupt.tar` uses the same gate on deterministic corrupted validation and is diagnostic only.

## Files

- `utils/p5_v11_ops.py`
- `train_p5_v11.py`
- `inference_p5_v11.py`
- `run_p5_v11_train.sh`
- `run_p5_v11_inference.sh`
- `tests/test_p5_v11_ops.py`

P5-v1 files remain intact for historical reproduction.

## Recommended first run

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
INIT_CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/p5_v11_repair_10pct \
TRAIN_SAMPLE_INTERVAL=0.1 \
EVAL_SAMPLE_INTERVAL=0.1 \
BATCH_SIZE=1 \
MAX_EPOCH=20 \
bash run_p5_v11_train.sh
```

Before full training, run:

```bash
pytest -q tests/test_p5_v11_ops.py tests/test_p5_ops.py tests/test_p5_argparse.py
```

and a 2-batch smoke with `P5_MAX_BATCHES=2`.
