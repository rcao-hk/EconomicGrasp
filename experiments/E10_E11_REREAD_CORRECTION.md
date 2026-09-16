# Corrected E10/E11 center-reread rerun

## Why E10/E11 must be rerun

The original counterfactual reread reconstructed the grouping graspness map from
`end_points["graspness_score"]` without applying the same `[0,1]` clamp used by
the native forward. The native model actually computes

```python
grasp_raw = graspness_score.squeeze(1)
grasp_sel = grasp_raw.clamp(0.0, 1.0)
...
graspness_map = grasp_sel.view(B, 1, H, W)
```

and stores the exact flattened map as `dbg_grasp_sel`.

The corrected `utils/cva_center_decoupling.py` now reuses `dbg_grasp_sel`
directly. For compatibility with an older endpoint lacking this key, it applies
the identical clamp to `graspness_score`.

This issue affects only the counterfactual local reread used by **E10/E11**.
E00/E01 from the previous run remain valid because both use the native forward.

## Fail-fast no-op replay audit

Before trusting E10/E11, the corrected rerun performs a native-center replay:

```text
read_center   = native_xyz
output_center = native_xyz
```

and compares the replayed tensors against the native forward:

- `grasp_cdf_pred_angle_depth`
- `grasp_width_pred_angle_depth`
- final decoded `[N,17]` grasps

The default absolute tolerance is `5e-5`. If the audit fails, the script exits
before exact-action evaluation. The maximum observed differences are written to
`summary.json -> noop_replay_max`.

## E10/E11-only rerun

The dedicated rerun computes E00 internally only to reproduce the original
native query selection. It sends only E10/E11 to the expensive exact evaluator.
Thus the previous E00/E01 results do not need to be recomputed.

Use exactly the same `SAMPLE_INTERVAL`, `QUERY_EVAL_MODE`, `QUERY_EVAL_NUM`,
`EVAL_VALID_ONLY`, checkpoint, camera, and split settings as the original run.
With deterministic image-FPS and E00-based query selection, corrected rows join
the previous E00/E01 rows by:

```text
(split, scene_id, anno_id, query_id)
```

### Smoke test

```bash
GPUS=0 \
SPLITS=test_seen \
MAX_SAMPLES=2 \
NUM_WORKERS=0 \
QUERY_EVAL_MODE=topk_uniform \
QUERY_EVAL_NUM=128 \
EVAL_VALID_ONLY=1 \
NOOP_CHECK_SAMPLES=2 \
bash run_center_decoupling_reread.sh
```

Inspect `test_seen/summary.json`. `noop_replay_max` must be below the configured
`NOOP_ATOL`.

### Full three-split rerun

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/center_decoupling_reread_corrected \
GPUS=0,1,2 \
SPLITS=test_seen,test_similar,test_novel \
SAMPLE_INTERVAL=0.1 \
QUERY_EVAL_MODE=topk_uniform \
QUERY_EVAL_NUM=128 \
EVAL_VALID_ONLY=1 \
POSE_DEPTH_MODE=global_film \
NOOP_CHECK_SAMPLES=2 \
bash run_center_decoupling_reread.sh
```

The launcher defaults to `NUMPY_MADVISE_HUGEPAGE=0` and one BLAS/OpenMP thread
per worker to avoid the memory-management/thread oversubscription seen on the
evaluation server.

## Outputs

Each split directory contains:

- `per_query.csv`: corrected E10/E11 exact-action rows only;
- `per_sample_summary.csv`: corrected per-frame E10/E11 metrics;
- `summary.json`: E10/E11 aggregate metrics, timing, and no-op replay audit;
- optional `raw_grasps/E10` and `raw_grasps/E11`.

Do not overwrite the old E00/E01 directory. Keep the corrected reread output in
a separate root and join results after the run.
