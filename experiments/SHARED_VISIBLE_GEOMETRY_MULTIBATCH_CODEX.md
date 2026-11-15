# Shared Visible Geometry: True Multi-Image Batch

Date: 2026-11-15. Repository: `rcao-hk/EconomicGrasp`.
Branch: `exp/shared-visible-geometry-multibatch`; parent:
`exp/shared-visible-geometry-refinement@8a3a6251c0f5a52cf68a92f64a894df696c0149d`.

This change adds **true batched computation for the newly trained shared visible
geometry field and action-quality heads**, rather than relying only on
per-frame forward/backward with gradient accumulation. It does not alter the
historical frozen MIX50 GVAR source, the offline CAD/Dex-Net label evaluator,
robot actions, quality targets, or the official GraspNet evaluation algorithm.

## 1. Shape contract

| Tensor | Legacy B=1 (retained) | True B>1 |
| --- | --- | --- |
| `feature_map` | `[C,Hf,Wf]` | `[B,C,Hf,Wf]` |
| `pred_depth` | `[H,W]` | `[B,H,W]` |
| camera K | `[3,3]` | `[B,3,3]` |
| source action bank | `[F,K,17]` | `[B,F,K,17]` |
| local action features | `[F,D]` | `[B,F,D]` |
| spatial query points | `[P,3]` | `[B,P,3]` |
| field output | `[P,6]` | `[B,P,6]` |
| utility logits | `[F,K]` | `[B,F,K]` |
| exact quality labels | `[F,K]` | `[B,F,K]` |
| visibility targets | `[P]` | `[B,P]` |

`F` is number of grasp families **per image** (train16/validation32/infer1024).
`K=7` is the actual-action offset family and is **not** a batch dimension.
`P` is number of field supervision queries per image (default4096), while
grasp scoring uses 75 sites per actual action.

**Per-image independence is enforced.** A scene has its own image memory,
metric-depth map, intrinsics, field encoding and action bank. Cross-image
feature pooling, attention, GT-label access and candidate mixing are forbidden.
Only matching tensor shapes can be collated; ragged bank or spatial grids stop
with an explicit error rather than zero-padding physical actions.

## 2. Implementation

- `shared_geometry/model.py`: `normalize_scene`, batched `SharedField.encode`
  and `query` with `torch.nn.functional.grid_sample` operating on `[B,C,H,W]`;
  `FieldGraspModel.action_scores` chunks along F (not B); fresh/stale support
  positions, `LocalControl`, and analytical predicted-geometry control accept
  B>1. For the predicted-geometry control, each scene has its own CPU/KDTree;
  the learned quality reader is batched. Both quality and geometry losses first
  reduce each image, then average across images.
- `shared_geometry/data.py`: explicit `collate_inputs` and `collate_frames`.
  Only feature_map/pred_depth/K/bank/local_features are forwarded to model;
  CAD/teacher targets remain outside the model scene.
- `shared_geometry/experiment.py`: one B>1 forward/backward per microbatch.
  Optional accumulation combines microbatches **only across optimizer steps**.
  The final partial effective group uses its actual count as denominator.
  Validation and diagnostic model forwards can also be batched.
- `shared_geometry/inference.py`: source provider snapshots remain B=1 (the
  historical model itself is not reworked). Independent snapshots can be
  collected for true multi-image NEW-head inference. Official output files and
  changed-action sensor masks remain per scene. The last partial head batch is
  processed, never dropped.
- `shared_geometry/cli.py` / `scripts/shared_geometry/run.sh`: batch flags.

### Two different batch-size concepts

`--batch-size 4 --accumulate 1` means an actual `[4,...]` model input, one
backward and one optimizer step per 4 independent images. Defaults for new
training are B=4, accumulate=1, validation B=4. Explicit legacy settings
`--batch-size 1 --accumulate 4 --val-batch-size 1` retain the old effective
batch of 4 but use four separate forwards.

The effective batch is `batch_size * accumulate` and the complete optimizer
update budget remains 2560/4=640 steps per epoch and 12,800 in 20 epochs.
Unlike the old model, new checkpoints include physical, accumulated, effective
and validation batch sizes in their exact resume protocol; trying to resume
with different batch settings **fails closed**. Previous checkpoint weights
remain strictly loadable for inference through the unchanged state dict, but
old training checkpoint protocols must not be silently resumed as new runs.

### Batching and memory

`point_chunk=16384` limits spatial points **per scene**, not the aggregate
`B*P`. `family_chunk=32` limits grasp families **per scene**. The expensive
source backbone/GVAR is still executed serially. For full infer, the scoring
head evaluates F=1024, K=7, 75 support locations = 537,600 positions **per
image**, so start `INFER_HEAD_BATCH_SIZE=1`; increase to2 or4 only after
measuring GPU memory. Do not confuse a batched new head with a batched native
point-cloud source. This change is not mixed precision or distributed training.

## 3. Commands

```bash
# Separate worktree. Do not overwrite previous run or graph experiments.
git fetch origin exp/shared-visible-geometry-multibatch
git worktree add /home/robotarm/EconomicGrasp-shared-geometry-batch \
  -b codex/shared-geometry-multibatch-exec \
  origin/exp/shared-visible-geometry-multibatch
cd /home/robotarm/EconomicGrasp-shared-geometry-batch

# Offline CPU tests (no CAD, no CUDA required):
python -m unittest tests.test_shared_geometry -v
python -m unittest tests.test_shared_geometry_multibatch -v
bash -n scripts/shared_geometry/run.sh
python -m compileall -q shared_geometry tests/test_shared_geometry_multibatch.py
```

Rebind the *verified* existing cache, visible-depth config, historical GVAR
checkpoint and source reader provider exactly as in
`experiments/SHARED_VISIBLE_GEOMETRY_ACTION_REFINEMENT_CODEX.md`; do **not**
regenerate geometry or exact-action labels merely because batch size changed.
Use a **new RUN_ROOT or model output directory** because protocol includes
new batch controls and the code hash has changed.

```bash
# Assumes the previous run's train/validation indices and frozen source are
# explicitly verified and available in RUN_ROOT.
export MODE=h2
export BATCH_SIZE=4
export ACCUMULATE=1
export VAL_BATCH_SIZE=4
bash scripts/shared_geometry/run.sh train
bash scripts/shared_geometry/run.sh diagnose

# Keep formal inference B=1 for initial GPU acceptance. Only new head batches.
export INFER_HEAD_BATCH_SIZE=1
bash scripts/shared_geometry/run.sh infer

# To test head B=2 or B=4, use a NEW inference output directory and ensure
# predictions and all five official dump policies agree with B=1 per frame.
export INFER_HEAD_BATCH_SIZE=2
```

Do **not** point `run.sh infer` at existing populated test dump directories.
The wrapper intentionally refuses overwriting. For B=1 vs B=2/4 parity runs,
use different `RUN_ROOT` targets or call CLI with a unique `--output` path.

Manual direct training example:

```bash
python -m shared_geometry.cli train \
  --train-index /ABS/train_index.json --val-index /ABS/validation_index.json \
  --mode h2 --hidden 64 --epochs 20 \
  --batch-size 4 --accumulate 1 --val-batch-size 4 \
  --seed 0 --geometry-weight 1 --device cuda:0 \
  --output /ABS/new_models/h2_b4
```

On GPU, run one smoke epoch for B=2 then B=4; monitor `torch.cuda.max_memory_allocated`,
wall time per optimizer update, CUDA OOM, finite geometry losses, and
per-image physical action identity. GPU speedup is not promised before it is
measured; PyTorch operation order can cause tiny B-dependent roundoff.

## 4. Acceptance gates and tests

Local CPU (2026-11-15):

- Original 27 tests pass unchanged.
- Added 14 new tests pass: true B=2/B=4 for all five modes, single vs batched
  output, per-parameter gradient parity, correct per-image loss weighting with
  different valid masks, fresh/stale queries, scene independence, chunk
  invariance, OOD interventions parity, ragged inputs, B=4 training and
  checkpoint resume/guard, B=4 vs B=1×4 training weight comparison, validation
  metric equivalence, 3-scene inference with last partial batch and identical
  official dump rows/masks, CLI and shell checks.
- PyTorch CPU-only. Neither the new runtime nor the old model's CUDA forward
  is claimed to be validated by these tests.

117 acceptance (NOT RUN at code delivery):

1. Original/Trans image-pair strict source parity and rendered-depth alignment.
2. B=1 vs B=2 vs B=4: logits, per-image geometry outputs, losses, and action
   identities (no silent drop/padding); finite gradients and profiler memory.
3. For all five modes, selected-Q, regret, geometry metrics, and output dumps
   match B=1 within declared FP32 tolerance.
4. Same-label, same-budget training B=1×4 vs true B=4×1; check measured speed
   and variance over at least the original seed. Never claim exact same
   floating-point weights because matrix kernels can differ.
5. Full inference: the frozen native 17D actions and reference dumps remain
   matched; changed actions are re-filtered independently. No CAD or GT labels
   are consumed by the test-time head.

**No new official AP or training result has been produced by this code-only
change.** Batch support is an implementation feature, not a research claim.