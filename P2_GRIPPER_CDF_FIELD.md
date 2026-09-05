# P2: Capacity-Matched Gripper-Conditioned CDF Field

## 1. Question

P1 showed that exact-action evaluator supervision is only weakly decodable from the frozen Stage-1 pre-CDF representation using the existing linear CDF head. P2 tests whether richer action-conditioned evidence makes the same target more learnable.

P2 fixes:

- the untouched original Stage-1 RGB student;
- the exact physical actions and evaluator labels mined for P1;
- Top-1 view inference;
- ordinary unbalanced CDF BCE as the only loss;
- the optimizer, seed, hidden width, and three-layer MLP architecture;
- model-free collision filtering at `0.01` during test inference.

P2 changes only the evidence made available to the CDF predictor.

## 2. No P1 checkpoint and no residual

P2 does **not** load the trained P1 exact-action checkpoint and does not add a residual to either the Stage-1 or P1 CDF output.

Each variant is randomly initialized and predicts the full depth-by-friction CDF lattice directly:

\[
\hat{S}_\theta(g) = \operatorname{MonotonicCDF}(\operatorname{MLP}_\theta(x_g)).
\]

The only required inputs are:

```text
BASE_CHECKPOINT
EXACT_ACTION_CACHE_DIR
```

`EXACT_ACTION_CACHE_DIR` is the cache produced by `mine_cva_exact_action_cdf_cache.py`; it contains the fixed Stage-1 actions, pre-CDF features, and clean-evaluator friction labels.

## 3. Four cumulative variants

The predictor operates on one center-angle row and outputs all four depth anchors and six friction thresholds, `[R, D, T]`. This preserves the depth-specific output structure of the original CDF head without giving P2-0 an explicit pose input.

### P2-0: `p2_0`

```text
pre-CDF feature only
```

This is the nonlinear-capacity control. It asks whether a three-layer MLP trained on exact-action labels is sufficient without adding explicit physical evidence.

### P2-A: `p2_a`

```text
P2-0 + exact action pose
```

For every depth anchor, the 12-D descriptor contains:

- normalized metric grasp center;
- center ray coordinates;
- approach view;
- `cos(2a), sin(2a)` for the parallel-jaw in-plane angle;
- approach depth;
- predicted physical width.

### P2-B: `p2_b`

```text
P2-A + projected gripper-region DPT evidence
```

Thirty-two canonical points are distributed over:

- left finger;
- right finger;
- closing volume;
- palm/back plate;
- approach swept corridor.

The points are transformed by the exact physical grasp, projected into the image, and used to sample `img_feat_dpt`. Region-wise mean features and projection-valid ratios are retained.

### P2-C: `p2_c`

```text
P2-B + signed ray-depth evidence
```

At the same support points, P2-C computes:

\[
r_m = \hat D(\pi(X_m^c)) - Z_m^c.
\]

For each gripper region it retains mean, mean absolute value, minimum, maximum, near-surface ratio, front ratio, behind ratio, and depth-valid ratio.

## 4. Exactly the same three-layer MLP

All variants instantiate the same fixed-width input layout:

```text
[base block | pose block | projected block | ray-depth block]
```

Unavailable blocks are hard-masked to zero after block normalization. Every variant therefore has the same nominal input width, the same LayerNorm modules, and exactly the same number of trainable parameters.

The predictor is:

```text
Linear(total_input_dim, hidden_dim)
GELU
Linear(hidden_dim, hidden_dim)
GELU
Linear(hidden_dim, D * T)
```

There are exactly three Linear layers. With the same seed, all variants start from identical parameter values; only the active evidence mask differs.

## 5. Only loss

For evaluator friction label \(\mu^\star(g)\) and thresholds

\[
\tau\in\{0.2,0.4,0.6,0.8,1.0,1.2\},
\]

the target is

\[
y_\tau(g)=\mathbf 1[0<\mu^\star(g)\le\tau].
\]

The only objective is

\[
\mathcal L=\operatorname{BCEWithLogits}(\hat{S}_\theta(g),Y^\star(g)).
\]

There is no ranking, hard-negative, class-balancing, collision, empty, utility-regression, width, KD, or residual loss.

## 6. Files

```text
models/p2_gripper_cdf_field.py
p2_gripper_field_common.py
mine_cva_p2_gripper_field_cache.py
p2_gripper_field_cache.py
validate_cva_p2_gripper_field_cache.py
train_cva_p2_cdf_field.py
inference_cva_p2_cdf_field.py
eval_p2_gripper_cdf_field.py
run_p2_gripper_cdf_field.sh
kill_p2_gripper_cdf_field.sh
tools/test_p2_gripper_cdf_field.py
```

The P2 cache schema is versioned separately from the earlier residual-on-P1 prototype. Old P2 enrichment files are rejected and must be regenerated. The expensive CAD/DexNet exact-action labels are reused and do not need to be mined again.

Enrichment validates only the source-cache scenes assigned to each GPU worker rather than rescanning all 100 scenes in every process. It contains no CAD/DexNet worker pool, so it avoids the force-closure deadlock mode encountered during P1 label mining. Each enriched frame is atomically saved and can be resumed with `ENRICH_OVERWRITE=0`.

## 7. Synthetic tests

```bash
cd /home/robotarm/EconomicGrasp
python tools/test_p2_gripper_cdf_field.py
bash -n run_p2_gripper_cdf_field.sh
bash -n kill_p2_gripper_cdf_field.sh
```

The tests verify:

- action-angle periodicity;
- projected field sampling;
- exactly three Linear layers;
- identical parameter count across P2-0/A/B/C;
- hard masking of inactive evidence;
- random scratch training and monotonic CDF output;
- row-level cache semantics `[R,D,T]`;
- checkpoint round-trip;
- paired evaluation protocol.

## 8. Smoke test

Use one training scene and one validation scene from the existing exact-action cache:

```bash
cd /home/robotarm/EconomicGrasp

export REPO_ROOT=/home/robotarm/EconomicGrasp
export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export DATASET_ROOT=/data2/robotarm/dataset/GraspNet1Billion

export BASE_CHECKPOINT=/path/to/original_stage1_checkpoint.tar
export EXACT_ACTION_CACHE_DIR=/path/to/p1_exact_action_cdf/cache

export WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/experiment/p2_scratch_smoke

export ENRICH_GPUS=0
export ENRICH_SCENE_IDS=0,90
export ENRICH_DATA_WORKERS=0
export ENRICH_ACTION_CHUNK=512
export ENRICH_OVERWRITE=0

export REQUIRE_ALL_SCENES=0
export MIN_FRAMES_PER_SCENE=1
export VAL_SCENE_START=90

export TRAIN_GPUS=0
export TRAIN_EPOCHS=2
export TRAIN_BATCH_SIZE=2
export TRAIN_NUM_WORKERS=0
export TRAIN_HIDDEN_DIM=256

export INFER_GPUS=0
export INFER_DATA_WORKERS=0
export INFER_ROW_CHUNK=128
export TEST_SAMPLE_INTERVAL=256
export EVAL_NUM_WORKERS=2

export PHASES=enrich,validate,train,infer,eval
export RESUME_PIPELINE=1

bash run_p2_gripper_cdf_field.sh
```

The smoke run validates interfaces only.

## 9. Formal 1/10 run

```bash
cd /home/robotarm/EconomicGrasp

export REPO_ROOT=/home/robotarm/EconomicGrasp
export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export DATASET_ROOT=/data2/robotarm/dataset/GraspNet1Billion

export BASE_CHECKPOINT=/path/to/original_stage1_checkpoint.tar
export EXACT_ACTION_CACHE_DIR=/path/to/p1_exact_action_cdf/cache

export WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/experiment/p2_scratch_mlp

export ENRICH_GPUS=0,1,2,3,4,5
export ENRICH_DATA_WORKERS=2
export ENRICH_ACTION_CHUNK=2048
export ENRICH_STORE_DTYPE=float32
export ENRICH_OVERWRITE=0

export REQUIRE_ALL_SCENES=1
export MIN_FRAMES_PER_SCENE=26
export VAL_SCENE_START=90

# One variant per GPU; all four use the same seed and architecture.
export TRAIN_GPUS=0,1,2,3
export TRAIN_EPOCHS=20
export TRAIN_BATCH_SIZE=16
export TRAIN_NUM_WORKERS=4
export TRAIN_LR=3e-4
export TRAIN_MIN_LR=1e-6
export TRAIN_HIDDEN_DIM=256
export TRAIN_SEED=0
export TRAIN_AMP=0

export INFER_GPUS=0,1,2,3,4,5
export INFER_DATA_WORKERS=2
export INFER_ROW_CHUNK=512
export TEST_SAMPLE_INTERVAL=10
export EVAL_NUM_WORKERS=10

export PHASES=enrich,validate,train,infer,eval
export RESUME_PIPELINE=1

bash run_p2_gripper_cdf_field.sh
```

## 10. Resume

Enrichment is frame-resumable:

```bash
ENRICH_OVERWRITE=0 PHASES=enrich bash run_p2_gripper_cdf_field.sh
```

Completed training and inference jobs are skipped when:

```bash
RESUME_PIPELINE=1
```

Selective phases:

```bash
PHASES=validate,train bash run_p2_gripper_cdf_field.sh
PHASES=infer,eval bash run_p2_gripper_cdf_field.sh
PHASES=eval bash run_p2_gripper_cdf_field.sh
```

Stop all P2 processes:

```bash
bash kill_p2_gripper_cdf_field.sh
```

## 11. Outputs

```text
WORK_ROOT/
├── cache/
├── train/
│   ├── p2_0/checkpoint_best_p2_scratch.tar
│   ├── p2_a/checkpoint_best_p2_scratch.tar
│   ├── p2_b/checkpoint_best_p2_scratch.tar
│   └── p2_c/checkpoint_best_p2_scratch.tar
├── predictions/
│   ├── base/{test_seen,test_similar,test_novel}
│   ├── p2_0/{...}
│   ├── p2_a/{...}
│   ├── p2_b/{...}
│   └── p2_c/{...}
├── evaluation/
│   ├── P2_REPORT.md
│   ├── p2_official_ap_long.csv
│   ├── p2_incremental_summary.csv
│   ├── p2_delta_vs_base.csv
│   ├── p2_paired_scene_summary.csv
│   └── p2_summary.json
└── logs/
```

## 12. Interpretation

The primary comparisons are:

\[
\Delta_0 = AP(P2\text{-}0)-AP(Base),
\]

\[
\Delta_A = AP(P2\text{-}A)-AP(P2\text{-}0),
\]

\[
\Delta_B = AP(P2\text{-}B)-AP(P2\text{-}A),
\]

\[
\Delta_C = AP(P2\text{-}C)-AP(P2\text{-}B).
\]

- Positive \(\Delta_0\): exact-action supervision plus nonlinear capacity helps even without explicit action evidence.
- Positive \(\Delta_A\): explicit metric action conditioning contributes beyond the common MLP.
- Positive \(\Delta_B\): projected gripper-region appearance contributes.
- Positive \(\Delta_C\): explicit ray-depth occupancy/clearance evidence contributes.

All variants have identical nominal parameter count. They still address only action scoring; proposal generation, view selection, and local pose refinement remain unchanged.
