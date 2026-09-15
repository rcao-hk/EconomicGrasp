# E00/E01/E10/E11 post-selector center-decoupling diagnostic

Branch: `exp/e00-e11-center-decoupling-diagnostic`  
Base: `main`

## Question

The current RGB-only CVA-CDF model stores dense evidence on the image plane, but
the predicted metric depth at one image-FPS token is reused as both:

1. the 3D center that conditions local evidence reading/grouping; and
2. the final 6-DoF grasp translation.

This diagnostic asks whether these two roles should remain hard-coupled.

## Intervention point

The intervention happens **after the center-view selector and before angle
expansion/local grouping**. The following are fixed from the native Stage-1
forward pass:

- DINO/DPT image features and predicted metric depth map;
- proposal/objectness/graspness maps;
- image-FPS token indices;
- selected Top-1 approach view;
- all model weights.

Rendered/fused GT depth is used only to construct a counterfactual reference
center at the same selected image pixel. It is never fed into the native model
forward path.

For selected pixel `(u,v)` and reference depth `z_ref`, the reference center is
backprojected on the same camera ray:

`c_ref = z_ref * [(u-cx)/fx, (v-cy)/fy, 1]`.

Invalid reference depth falls back to the native center and is marked invalid.
With `EVAL_VALID_ONLY=1` (default launcher setting), such queries are removed
before exact-action evaluation.

## Four variants

| Variant | center used by local grouping/CDF read | final grasp translation | Purpose |
|---|---|---|---|
| E00 | predicted center | predicted center | native baseline |
| E01 | predicted center | reference center | translation-only intervention |
| E10 | reference center | predicted center | read-center-only mechanism diagnostic |
| E11 | reference center | reference center | coherent re-read + translation intervention |

E01 copies the native decoded grasp and changes only columns 13:16 (xyz).
E10/E11 share **one** counterfactual local grouping + CDF/width decode because
their read center is identical. E11 is then produced from E10 by replacing only
xyz. This removes a redundant second counterfactual forward without changing the
experiment.

The strict invariants are:

- E00 and E01 must be identical in score, width, height, insertion depth,
  rotation and object-id; only xyz may differ.
- E10 and E11 must satisfy the same invariant.
- all variants preserve native query identity/order before optional paired query
  subsampling.

## Fast exact-evaluation path

The expensive part of this diagnostic is usually the CPU GraspNet/DexNet exact
evaluator, not DPT inference. The fast path therefore preserves the causal
comparison while reducing unnecessary evaluator work.

### 1. One exact-evaluator call per frame

The evaluated E00/E01/E10/E11 arrays are concatenated and passed to
`ExactGraspNetActionEvaluator.evaluate()` once. The returned arrays are then
split back into four equal variant blocks. This avoids repeating scene/model
pose setup and object assignment four times.

### 2. Reference-valid-only evaluation

`EVAL_VALID_ONLY=1` evaluates only queries whose rendered/fused reference depth
is finite and inside the configured metric-depth range. These are exactly the
queries used by the paired center intervention.

### 3. Deterministic paired query subsampling

The launcher defaults to:

```text
QUERY_EVAL_MODE=topk_uniform
QUERY_EVAL_NUM=128
```

Selection uses **E00/native information only**, before any intervention, so E01,
E10 and E11 cannot influence which queries are evaluated. The same query ids are
used for all four variants.

Modes:

- `all`: all eligible native queries;
- `topk`: highest E00 decoded scores;
- `uniform`: deterministic evenly spaced native query ids;
- `topk_uniform`: half highest-score queries + half deterministic population
  coverage from the remaining queries.

`QUERY_EVAL_NUM=0` restores exhaustive evaluation of every eligible native query.

The fast 128-query setting is intended for mechanism diagnosis. Once a clear
paired effect is found, use `QUERY_EVAL_NUM=0` for exhaustive confirmation on
selected splits/scenes.

## Evaluation

The diagnostic uses `ExactGraspNetActionEvaluator` directly on every selected
same-query candidate, before model-free collision filtering, NMS or Top-K scene
ranking. This removes candidate-count/ranking confounds and reports:

- official-style friction result for each exact action;
- collision / pure-collision / empty;
- success at friction thresholds 0.4 and 0.8;
- reference-center displacement in mm.

Primary causal comparisons:

- `E01 - E00`: effect of changing **only output translation**;
- `E11 - E01`: additional effect of **re-reading evidence** at the same reference
  output center;
- `E10 - E00`: effect of changing the read center while holding translation;
- `E11 - E00`: combined effect.

Interpretation:

- `E01 > E00`, `E11 ~= E01`: translation anchoring is the main bottleneck;
- `E01 > E00`, `E11 > E01`: both translation and center-consistent evidence
  reading matter;
- `E01 > E00`, `E11 < E01`: the frozen Stage-1 local representation/head does
  not transfer cleanly to counterfactual centers;
- large reference-center headroom but weak E11: the candidate location is
  useful, but current RGB/local evidence cannot reliably evaluate it.

## Timing

With `PROFILE_TIMING=1`, `per_sample_summary.csv` and `summary.json` include:

- native Stage-1 forward + E00 decode time;
- shared counterfactual E10/E11 local re-read time;
- total exact-evaluator time;
- evaluator collision time;
- evaluator force-closure time.

This makes it explicit whether further optimization should target GPU inference
or CPU force-closure evaluation.

## Run

The canonical `economicgrasp_dpt_cva_cdf_distill_stage1` checkpoint uses
`POSE_DEPTH_MODE=global_film`; the launcher therefore defaults to this mode.

Recommended fast 10% diagnostic on three GPUs:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/center_decoupling_diag \
GPUS=0,1,2 \
SPLITS=test_seen,test_similar,test_novel \
SAMPLE_INTERVAL=0.1 \
QUERY_EVAL_MODE=topk_uniform \
QUERY_EVAL_NUM=128 \
EVAL_VALID_ONLY=1 \
PROFILE_TIMING=1 \
bash run_center_decoupling_diag.sh
```

Fast smoke:

```bash
GPUS=0 \
SPLITS=test_seen \
MAX_SAMPLES=2 \
NUM_WORKERS=0 \
QUERY_EVAL_NUM=32 \
VERIFY_N=4 \
bash run_center_decoupling_diag.sh
```

Exhaustive confirmation:

```bash
GPUS=0,1,2 \
QUERY_EVAL_MODE=all \
QUERY_EVAL_NUM=0 \
EVAL_VALID_ONLY=1 \
bash run_center_decoupling_diag.sh
```

Unit tests that do not require GraspNet data/GPU:

```bash
pytest -q tests/test_cva_center_decoupling.py
```

## Outputs

Each split directory contains:

- `per_query.csv`: one row for every evaluated `(scene, anno, native query,
  variant)`;
- `per_sample_summary.csv`: sample-level exact-action metrics and timing;
- `summary.json`: aggregate variant metrics, paired deltas, timing, evaluated
  fraction and invariant checks;
- optional `raw_grasps/<variant>/scene_xxxx/xxxx.npy` when
  `SAVE_RAW_GRASPS=1`.

## Scope / limitations

This is a causal diagnostic, not a deployable RGB-only method: rendered/fused GT
depth defines the counterfactual reference center. It must not be reported as an
RGB-only inference result. Its role is to identify whether the shortest next
method change should target translation anchoring, local evidence reading, or
both.
