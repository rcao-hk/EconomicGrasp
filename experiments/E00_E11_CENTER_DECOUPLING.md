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

Invalid reference depth falls back to the native center and is marked invalid;
such queries are excluded from paired aggregate statistics.

## Four variants

| Variant | center used by local grouping/CDF read | final grasp translation | Purpose |
|---|---|---|---|
| E00 | predicted center | predicted center | native baseline |
| E01 | predicted center | reference center | translation-only intervention |
| E10 | reference center | predicted center | read-center-only mechanism diagnostic |
| E11 | reference center | reference center | coherent re-read + translation intervention |

E01 copies the native decoded grasp and changes only columns 13:16 (xyz). E10
and E11 re-run only angle expansion, local cross-attention grouping and the
frozen CDF/width decoder. View prediction/selection is not re-run.

The strict invariants are:

- E00 and E01 must be identical in score, width, height, insertion depth,
  rotation and object-id; only xyz may differ.
- E10 and E11 must satisfy the same invariant.
- all four variants must contain exactly one grasp per native Top-1 image-FPS
  query and preserve query order.

## Stage-1 depth configuration

The canonical checkpoint used by this diagnostic,
`economicgrasp_dpt_cva_cdf_distill_stage1`, was trained with pose-conditioned
metric depth using `pose_depth_mode=global_film`. The launcher therefore defaults
to:

```bash
POSE_DEPTH_MODE=global_film
```

Do not switch this to `none` when evaluating that checkpoint. A different mode
should only be supplied for a checkpoint trained with the corresponding depth
architecture.

## Evaluation

The primary diagnostic uses `ExactGraspNetActionEvaluator` directly on every
raw same-query candidate, before model-free collision filtering, NMS or Top-K
ranking. This removes candidate-count/ranking confounds and reports:

- official-style friction result for each exact action;
- collision / pure-collision / empty;
- success at friction thresholds 0.4 and 0.8;
- the reference-center displacement in mm.

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

## Multi-GPU run

`run_center_decoupling_diag.sh` launches one independent split per GPU. Splits
are assigned to the GPUs listed in `GPUS` round-robin, and the launcher waits in
waves if there are more splits than GPUs. Each worker writes its own
`<OUTPUT_ROOT>/<split>/diagnostic.log`, so there is no shared model process or
output-file race.

Recommended 3-GPU 10% diagnostic:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/center_decoupling_diag \
GPUS=0,1,2 \
SPLITS=test_seen,test_similar,test_novel \
SAMPLE_INTERVAL=0.1 \
POSE_DEPTH_MODE=global_film \
bash run_center_decoupling_diag.sh
```

`POSE_DEPTH_MODE=global_film` is already the launcher default and can be omitted
for the canonical Stage-1 checkpoint.

With fewer GPUs, for example:

```bash
GPUS=0,1 \
SPLITS=test_seen,test_similar,test_novel \
SAMPLE_INTERVAL=0.1 \
bash run_center_decoupling_diag.sh
```

`test_seen` and `test_similar` run first; `test_novel` starts after that wave
finishes. Supplying more than three GPUs does not accelerate the default three
splits because the current launcher parallelizes across splits rather than
sharding one split across multiple GPUs.

Fast smoke:

```bash
GPUS=0 \
SPLITS=test_seen \
MAX_SAMPLES=2 \
NUM_WORKERS=0 \
VERIFY_N=4 \
bash run_center_decoupling_diag.sh
```

Unit tests that do not require GraspNet data/GPU:

```bash
pytest -q tests/test_cva_center_decoupling.py
```

## Outputs

Each split directory contains:

- `diagnostic.log`: stdout/stderr of that GPU worker;
- `per_query.csv`: one row for every `(scene, anno, query, variant)`;
- `per_sample_summary.csv`: sample-level exact-action means over valid reference
  queries;
- `summary.json`: aggregate variant metrics, paired deltas and intervention
  invariant checks;
- optional `raw_grasps/<variant>/scene_xxxx/xxxx.npy` when
  `SAVE_RAW_GRASPS=1`.

## Scope / limitations

This is a causal diagnostic, not a deployable RGB-only method: rendered/fused GT
depth defines the counterfactual reference center. It must not be reported as an
RGB-only inference result. Its role is to identify whether the shortest next
method change should target translation anchoring, local evidence reading, or
both.
