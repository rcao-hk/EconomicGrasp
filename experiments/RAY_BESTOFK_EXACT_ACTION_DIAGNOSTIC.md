# Frozen ray best-of-K exact-action diagnostic

Branch: `exp/ray-bestofk-exact-action-diagnostic`

## Question

Previous P2/P3 experiments already showed that expanding one Stage-1 image ray
into multiple camera-z centers increases geometric label-point support, while
learned cross-depth selection did not reliably convert that support into AP.
The missing diagnostic is stricter:

> Do the actual frozen Stage-1 grasps decoded at those alternative centers contain
> a substantially better exact-action candidate, and if so, how much of that
> headroom is recovered by the current raw CDF score?

This experiment answers that question without training.

## Frozen protocol

Use the canonical RGB-only Stage-1 CVA-CDF checkpoint.  The native model fixes:

- RGB/DINO-DPT image evidence;
- image-FPS query pixel;
- Top-1 approach view;
- all network weights.

For every selected image ray, construct camera-z offsets

`[-40,-20,-10,0,+10,+20,+40] mm`

around the Stage-1 predicted metric center.  Every alternative center is
backprojected on the same image ray.  The corrected center-decoupling helper then
re-runs only angle expansion, local grouping, and frozen CDF/width decoding at
that center.  Read center and emitted action center are always the same physical
hypothesis.

The zero-offset candidate is the native Stage-1 grasp.  The first frames also
run a native-center no-op replay and require exact CDF/width/decoded parity before
any result is accepted.

## Three policies

For each ray with K decoded grasps:

1. `native`: always use the zero-offset Stage-1 grasp;
2. `raw_selected`: choose the valid center with the largest frozen decoded CDF
   score;
3. `oracle`: evaluate every valid candidate with the CAD/DexNet exact-action
   evaluator and choose the largest official-threshold utility.  Ties use the
   frozen raw score only for deterministic selection.

Exact utility is the fraction of official friction thresholds
`{0.2,0.4,0.6,0.8,1.0,1.2}` satisfied by the physical grasp.  Collision, empty,
no-contact and force-closure failures have zero utility.

The oracle is post-hoc diagnostic information.  It is not a deployable RGB-only
result and is never used by the network.

## Primary quantities

The experiment separates two gaps:

- **candidate headroom**: `oracle - native`;
- **selection gap/regret**: `oracle - raw_selected`.

It also reports:

- Success@0.4 / Success@0.8;
- collision, pure collision and empty rates;
- raw-score/oracle center-selection histograms;
- raw-score vs oracle exact center match ratio;
- native->raw and native->oracle rescue/harm rates;
- per-offset exact-action quality;
- per-frame timing.

The main scientific gate is:

- high oracle headroom + low raw recovery: candidate expansion is useful, but
  cross-center ranking remains unsolved;
- high oracle headroom + strong raw recovery: a simple frozen center beam may be
  enough and no learned selector is immediately necessary;
- low oracle headroom: do not spend more effort on selector complexity for this
  ray-only center grid.

## Run

Smoke first:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_bestofk_exact_smoke \
GPUS=0 SPLITS=test_seen MAX_SAMPLES=2 NUM_WORKERS=0 \
QUERY_EVAL_NUM=32 VERIFY_N=4 \
bash run_ray_bestofk_exact_diag.sh
```

Then check `test_seen/summary.json`; all `noop_replay_max` values should be at or
below the configured tolerance.

Main 10% protocol:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_bestofk_exact_diag \
GPUS=0,1,2 \
SPLITS=test_seen,test_similar,test_novel \
SAMPLE_INTERVAL=0.1 \
OFFSETS_MM=-40,-20,-10,0,10,20,40 \
QUERY_EVAL_NUM=128 \
QUERY_EVAL_MODE=topk_uniform \
POSE_DEPTH_MODE=global_film \
NOOP_CHECK_SAMPLES=2 \
NOOP_ATOL=5e-5 \
PROFILE_TIMING=1 \
bash run_ray_bestofk_exact_diag.sh
```

The launcher defaults `NUMPY_MADVISE_HUGEPAGE=0` and limits BLAS/OpenMP thread
counts to one per split worker.

Unit tests:

```bash
pytest -q tests/test_ray_bestofk_diagnostic.py tests/test_cva_center_decoupling.py
```

## Outputs

Each split directory contains:

- `summary.json`: native/raw/oracle aggregate and gap metrics;
- `per_query.csv`: one row per evaluated Stage-1 ray with all three policies;
- `per_candidate.csv.gz`: exact outcome for every K center candidate;
- `offset_summary.csv`: per-offset validity, quality and selection frequencies;
- `per_sample_summary.csv`: frame-level paired metrics;
- optional `raw_grids/scene_xxxx/xxxx.npz` when `SAVE_RAW_GRASPS=1`.

## Scope

This diagnostic deliberately does not reuse P2-v1/P2-v2 trainable ray heads or
P3 contextual aggregation.  It asks only whether the corrected frozen Stage-1
local reread produces actual physical grasp headroom on the historical K=7 ray
grid and whether the unmodified CDF score can exploit that headroom.
