# DCR-CVA: Decoupled Center Correction and Query Ranking

Branch: `exp/e1-e2-center-hypothesis-cva`.

## Motivation and evidence limits

E1-3 retained E1's center choices but emitted the immutable Stage-1 query score.
It improved the observed Similar/Novel cells over E1's selected CDF ranking.
This supports separating local selection from global ranking. It does **not**
prove that any learned residual ranker will improve over frozen ranking, nor
prove the cause of E2's AP loss. DCR makes these alternatives directly testable.

The existing E1/E2 experiments and default behavior are unchanged. E1-3 remains
a valid no-training baseline; DCR is not a new name for that baseline alone.

## Method

For each fixed-reference query q, retain its exact physical center hypotheses
`g[q,c]`, immutable native rotation/width/insertion depth, and Stage-1 score s0[q].

1. **Correction path:** online RGB -> trainable DPT image adapter -> spatial
   enhancer -> candidate-specific CVA -> six-threshold CDF. Select c* using
   mean CDF utility, with numerical ties staying native. The optional native-
   relative CDF loss affects ONLY this path.
2. **Ranking path:** use a small MLP on **detached** native/candidate CVA latents,
   CDF profiles, offsets and s0. It predicts a bounded log-odds residual r[q,c].
3. **Frozen ranking anchor:** output
   `s[q] = sigmoid(logit(s0[q]) + alpha * r[q,c*])`.
   Implementation uses the equivalent odds form to preserve the exact native
   score when r=0 or alpha=0. s0=0/1 remain endpoints.

Default `|r| <= 0.5`, `alpha=1`; the last MLP layer starts at zero. Furthermore,
`r[q,0]=0` structurally: keeping native also keeps its original ranking score.
This is a conservative ordering update, NOT a calibrated success probability
for the translated action. The numerical residual bound is a design choice,
not an experimentally established optimum or a no-harm guarantee.

## Loss and gradient routing

Let y[q,c] be mean exact six-threshold target utility.

- `selection-loss=cdf`: the original E1 BCE.
- `selection-loss=cdf_relative`: the same BCE plus E2 relative SmoothL1.
  This is an optional ablation; relative supervision has not been established
  as better than E1.
- Ranking supervises **different queries in the same frame**, after center
  selection, using their actual selected-action targets y[q,c*]. For every pair
  with unequal utility, apply gap-weighted logistic pairwise ranking to the
  anchored score logits. Equivalent-utility pairs receive no ordering label.
- Add `RANK_ANCHOR_WEIGHT * mean(r_selected**2)` to penalize needless score drift.

Ranking sees no labels/case identity in its inputs. Exact targets are used only
in the training loss. Hard center selection is detached; the rank loss cannot
move the gripper or optimize an unlabeled pose.

The rank MLP receives detached features/CDF. The image adapter, enhancer,
grouping, action adapter and CDF still receive correction-loss gradients and
are trained jointly. This is **not only training a gate**, and ranking is not
fully end-to-end into the visual backbone. Independent optimizers and gradient
clipping prevent rank gradients from changing the correction update indirectly.
The immutable Stage-1 generator/depth/backbone and native R/w/d remain frozen,
exactly as in E1. No new feature cache or CAD/DexNet mining is needed.

## Checkpoints and validation

Default warm start: the successful E1 `checkpoint_best.pt`. Default fine-tuning
is 6 epochs, correction LR=1e-5, rank LR=1e-4. These are starting settings, not
claimed optimized values. Same 10% frame schedule: train 2600, Seen 780,
Similar 780, Novel 780. Existing cache query budget is unchanged (normally 64).
Full-query inference uses `INFER_QUERIES=0`.

The checkpoint criterion stays **Seen mean selected exact utility**, so choosing
a different ranking mode does not choose a different correction checkpoint.
An initial evaluation and `checkpoint_initial.pt` preserve the warm-start,
zero-residual baseline; `checkpoint_best.pt` remains initial if fine-tuning
never improves this validation criterion. `checkpoint_latest.pt` is resumable.

Per-frame cross-query concordance and top-10/top-50 target utility are also
logged separately for all ranking modes. They use the sampled validation
queries: they are diagnostics, not official AP. There is no held-out tuning
of checkpoint, residual strength or bound. Official AP is still required.

## One inference pass, three identical-action controls

| Dump method | Physical action | Output score |
|---|---|---|
| `native` | immutable Stage-1 native | Stage-1 s0 |
| `local` | DCR-selected center | local CDF mean |
| `stage1` | **same selected center** | **Stage-1 s0** |
| `anchored` | **same selected center** | bounded learned update to s0 |

`anchored - stage1` isolates the ranking residual on identical physical actions.
`stage1 - local` tests fixed versus local-CDF ranking on those same actions.
`stage1 - native` isolates physical correction at fixed query scores. GraspNet
NMS/rank-dependent selection remains part of official evaluation.

The new inference script can also load an old E1/E2 checkpoint; the new ranker
then has exactly zero residual. `local` should reproduce the old model-score
pipeline and `stage1`/`anchored` the old action-only pipeline. Synthetic CPU
integration verifies contracts; real-server numerical equivalence must still
be checked using actual checkpoints before relying on it.

## Run

```bash
cd /home/robotarm/EconomicGrasp
git switch exp/e1-e2-center-hypothesis-cva
git pull --ff-only
python -m pytest -q tests/test_e1e2_cva.py tests/test_dcr_cva.py
```

### Smoke (separate root; no official evaluation of an incomplete schedule)

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/dcr_cva_smoke \
GPUS=0 LOSS_MODES=cdf EPOCHS=1 \
MAX_TRAIN_FRAMES=4 MAX_VAL_FRAMES=4 INFER_MAX_FRAMES=4 \
GRAD_ACCUM_STEPS=1 PHASES=train,infer \
bash scripts/run_dcr_cva.sh
```

### Formal E1-based DCR

```bash
E1_BASE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/dcr_cva_10pct \
GPUS=0,1,2 LOSS_MODES=cdf EPOCHS=6 \
SAMPLE_INTERVAL=0.1 INFER_QUERIES=0 \
LR=0.00001 RANK_LR=0.0001 RANK_BOUND=0.5 RANK_ANCHOR_WEIGHT=0.1 \
OFFICIAL_WORKERS=2 PHASES=train,infer,eval,summary RESUME=1 \
bash scripts/run_dcr_cva.sh
```

Override `DATASET_ROOT`, `STAGE1_CKPT`, `CACHE_ROOT`, `INIT_CHECKPOINT` when paths
differ. Reuse `E1_BASE_ROOT/action_cache` and `E1_BASE_ROOT/train/E1/checkpoint_best.pt`.
Do not re-run `prepare`. Do not reuse a smoke root for formal output.

### Optional relative-selection ablation

```bash
GPUS=0,1 LOSS_MODES=cdf,cdf_relative PHASES=train,infer,eval,summary \
bash scripts/run_dcr_cva.sh
```

Both modes warm-start from the SAME E1 weights by default. This tests relative
fine-tuning under decoupled ranking, not an epoch-matched E2-from-Stage-1 replication.
Training is one GPU per loss mode, not DDP. Inference shards every split by
scene across listed GPUs. Official evaluation uses the GPU list as split-level
concurrency slots (CPU-heavy): 3 active splits x 2 OFFICIAL_WORKERS means up to
about 6 evaluator workers, plus parent processes. The methods within a split
are evaluated serially. Stop any previous launcher before resuming with more
workers. No GPU-holder is required.

Default official methods are `local,stage1,anchored`. `native` dumps are produced
but not reevaluated by default because the native reference is already known.
Use `EVAL_METHODS=native,local,stage1,anchored` for a self-contained comparison.
A new WORK_ROOT is needed for different inference conditions/strength/checkpoints;
changing worker count does not change the scheduled frames. Ctrl-C/TERM cleans
up process groups. Python cache/checkpoint/manifest checks fail on mismatches.

### No-training zero-residual check

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/dcr_zero_check \
INFER_CHECKPOINT=/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct/train/E1/checkpoint_best.pt \
GPUS=0 INFER_MAX_FRAMES=4 PHASES=infer \
bash scripts/run_dcr_cva.sh
```

## Outputs and interpretation

```
WORK_ROOT/train/cdf/{initial.json,metrics.json,best.json,gradient_check.json}
WORK_ROOT/train/cdf/checkpoint_{initial,best,latest}.pt
WORK_ROOT/test/cdf/dump/{native,local,stage1,anchored}/...
WORK_ROOT/test/cdf/traces/<split>/scene_XXXX/ann_XXXX_<case>.npz
WORK_ROOT/test/cdf/official/<method>/<case>/<split>/{summary.json,accuracy.npy}
WORK_ROOT/{comparison.csv,ranking_effects.csv}
```

The small traces contain selections, local utility, base scores and selected
rank residuals, enabling subsequent score-only analysis without another model
forward. No score policy can change selected physical actions.

Keep `stage1` as the established control. Promote `anchored` only if it adds
reproducible held-out AP without losing nominal/robustness performance. Better
training pair loss or sampled validation concordance alone is insufficient.
If residual ranking fails, retaining frozen Stage-1 ranking remains a valid,
simpler factorization; do not infer that a larger rank MLP will solve it.
