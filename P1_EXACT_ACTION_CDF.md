# P1: Exact-Action CDF Utility Learnability Probe

## 1. Scientific question

P0-E showed that clean-geometry scoring of the Student's physical grasp actions has a large upper bound. P1 asks a narrower question:

> Can the existing frozen Stage-1 RGB representation predict the evaluator-aligned outcome of an exact physical grasp action?

P1 is intentionally a **head-only linear probe**. It does not introduce a new Transformer or a new grasp-field network. The experiment updates only the weight and bias of the existing `decoder.cdf_head`; all image, metric-depth, proposal, view, CVA, and width modules remain frozen.

A positive result establishes that the current representation already contains decodable action-utility information but the original CDF supervision did not extract it sufficiently. A negative result does not rule out privileged grasp-field learning; it indicates that a richer gripper-conditioned representation is needed.

## 2. Controlled method

### Training actions

For the corrected Stage-1 RGB student, the cache miner retains:

- deterministic image-FPS grasp centers;
- the Student's Top-1 approach view;
- all 12 in-plane angles;
- all four approach-depth anchors;
- the Student's predicted depth-wise width.

Each resulting physical grasp is evaluated against clean GraspNet CAD/table geometry and DexNet force closure. The cache also stores the exact input to the deployed CDF head.

### CDF target

For evaluator friction label \(\mu^\star(g)\) and thresholds

\[
\tau \in \{0.2,0.4,0.6,0.8,1.0,1.2\},
\]

P1 uses

\[
y_\tau(g)=\mathbf 1[0<\mu^\star(g)\le \tau].
\]

Collision, empty, and force-closure failure actions have an all-zero target.

### Only training loss

\[
\mathcal L_{\mathrm{P1}}
=
\operatorname{BCEWithLogits}
\left(\hat{\mathbf s}^{\mu}(g),\mathbf y^{\mu}(g)\right).
\]

The implementation uses ordinary, unbalanced, threshold-wise CDF BCE. It has:

- no ranking loss;
- no hard-negative loss or reweighting;
- no class/threshold balancing;
- no auxiliary utility, collision, empty, or width loss;
- no teacher-output KD;
- no additional scorer network.

Monotonic CDF logits are retained by construction through the current cumulative-positive-increment parameterization.

### Evaluation

The untouched Stage-1 checkpoint (`base`) and the CDF-head-updated checkpoint (`exact`) execute the same RGB-only graph. Test-time model-free collision filtering is fixed to:

```text
collision_thresh = 0.01
collision_voxel_size = 0.01
```

The official GraspNet evaluator compares Base and Exact-CDF on Seen, Similar, and Novel splits.

## 3. Reused repository components

P1 uses the current repository implementations directly:

```text
mine_cva_exact_action_cdf_cache.py
validate_cva_exact_action_cdf_cache.py
exact_action_cdf_cache.py
exact_action_cdf_common.py
exact_action_graspnet_evaluator.py
train_cva_exact_action_cdf_head.py
inference_cva_exact_action_cdf.py
```

The added files only complete the end-to-end protocol:

```text
run_p1_exact_action_cdf.sh
  cache mining → validation → CDF-only training → paired inference → official AP

eval_p1_exact_action_cdf.py
  strict protocol checks, official AP, delta tables, and learnability report

kill_p1_exact_action_cdf.sh
  terminates launcher, miner, trainer, inference, evaluator, and child processes

tools/test_p1_exact_action_cdf_protocol.py
  synthetic protocol/report tests independent of graspnetAPI
```

## 4. Recommended formal configuration

```bash
cd /home/robotarm/EconomicGrasp

export REPO_ROOT=/home/robotarm/EconomicGrasp
export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export DATASET_ROOT=/data2/robotarm/dataset/GraspNet1Billion

export BASE_CHECKPOINT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar

export WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/experiment/p1_exact_action_cdf

# Six GPUs are available. Each miner keeps its own model and scene shard.
export MINE_GPUS=0,1,2,3,4,5
export MINE_EVAL_WORKERS_PER_GPU=2
export MINE_MAX_PENDING_PER_GPU=4
export MINE_SAMPLE_FRACTION=0.1
export MINE_TOP_CENTERS=16
export MINE_RANDOM_CENTERS=4
export REQUIRE_ALL_SCENES=1
export MINE_MIN_FRAMES_PER_SCENE=26

export TRAIN_DEVICE=cuda:0
export TRAIN_EPOCHS=20
export TRAIN_BATCH_SIZE=16
export TRAIN_LR=1e-4
export TRAIN_MIN_LR=1e-6
export TRAIN_AMP=0

# Paired Base/Exact inference can use multiple GPUs.
export INFER_GPUS=0,1,2,3,4,5
export TEST_SAMPLE_INTERVAL=10
export EVAL_NUM_WORKERS=10

export PHASES=mine,validate,train,infer,eval
export RESUME_PIPELINE=1

bash run_p1_exact_action_cdf.sh
```

`TEST_SAMPLE_INTERVAL=10` is the directional 1/10 evaluation. After confirming the mechanism, use:

```bash
export PHASES=infer,eval
export TEST_SAMPLE_INTERVAL=1
export PREDICTION_ROOT=${WORK_ROOT}/predictions_full
export EVAL_OUTPUT_DIR=${WORK_ROOT}/evaluation_full
bash run_p1_exact_action_cdf.sh
```

## 5. Fast smoke test

A smoke cache must contain at least one training scene below `VAL_SCENE_START` and one validation scene at or above it:

```bash
export WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/experiment/p1_exact_action_cdf_smoke
export MINE_GPUS=0
export MINE_SCENE_IDS=0,90
export MINE_ANNO_IDS=0,10
export REQUIRE_ALL_SCENES=0
export MINE_MIN_FRAMES_PER_SCENE=1
export MINE_FC_VERIFY_N=8
export TRAIN_EPOCHS=2
export TRAIN_BATCH_SIZE=2
export TEST_SAMPLE_INTERVAL=256
export INFER_GPUS=0
export PHASES=mine,validate,train,infer,eval

bash run_p1_exact_action_cdf.sh
```

The smoke test validates code paths only; it cannot support a scientific conclusion.

## 6. Resume and selective phases

Cache mining skips an existing `scene_xxxx/ann_xxxx.npz` unless `MINE_OVERWRITE=1`.

The launcher skips completed training and inference outputs when:

```bash
export RESUME_PIPELINE=1
```

Examples:

```bash
# Validate an existing cache and train only.
PHASES=validate,train bash run_p1_exact_action_cdf.sh

# Re-run paired test inference and evaluation only.
PHASES=infer,eval bash run_p1_exact_action_cdf.sh

# Evaluate existing dumps only.
PHASES=eval bash run_p1_exact_action_cdf.sh
```

To stop the complete process tree:

```bash
bash kill_p1_exact_action_cdf.sh
```

Keep an official evaluation process alive while stopping generation/training/inference:

```bash
INCLUDE_OFFICIAL_EVAL=0 bash kill_p1_exact_action_cdf.sh
```

## 7. Outputs

```text
${WORK_ROOT}/
├── cache/
│   ├── scene_0000/ann_0000.npz
│   ├── cache_inventory.json
│   └── cache_inventory_detailed.json
├── train_seed_0/
│   ├── probe_contract.json
│   ├── metrics.jsonl
│   ├── best.json
│   └── checkpoint_best_exact_action.tar
├── predictions/
│   ├── base/test_seen/...
│   ├── base/test_similar/...
│   ├── base/test_novel/...
│   ├── exact/test_seen/...
│   ├── exact/test_similar/...
│   └── exact/test_novel/...
├── evaluation/
│   ├── p1_official_ap_long.csv
│   ├── p1_delta_summary.csv
│   ├── p1_summary.json
│   ├── P1_REPORT.md
│   └── ap_arrays/
└── logs/
```

## 8. Decision rule

The report distinguishes three outcomes without imposing an arbitrary AP threshold:

1. `learnable_and_transfers_to_official_ap`
   Validation CDF loss improves and Mean official AP is higher than Base.

2. `locally_learnable_without_positive_official_ap_transfer`
   Cached validation metrics improve, but official Mean AP does not. This indicates cache/admission mismatch or insufficient ranking transfer.

3. `learnability_not_demonstrated`
   The current frozen feature plus existing linear CDF head cannot extract the privileged signal under this protocol.

The load-bearing result is the paired official AP, not training BCE alone.

## 9. Scope and limitations

P1 deliberately fixes several factors:

- Top-1 view only;
- selected centers are the Base model's top-utility and random centers;
- only the existing linear CDF head is trainable;
- exact labels are cached for fixed Base features and actions;
- no proposal transfer or local action refinement is attempted.

Consequently, P1 tests **linear decodability and downstream transfer** of evaluator-aligned action utility. It is not the final Privileged Grasp-Field method.
