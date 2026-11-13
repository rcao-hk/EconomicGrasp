# Quality Decodability → Decision-aware Scoring → Global-budget Selection (P0–P2)

**Created:** 2026-11-13  
**Execution branch:** `exp/quality-decoding-decision-p0-p2`  
**Base commit:** `214ba88e9ada3c7c4dd37bb4ab2c4f0f83119dd5`, from `exp/candidate-conflict-quality-oracle`.  
**Status at authoring:** GitHub implementation available; real 117 CUDA/CAD experiments **not executed in this turn**. No official P0/P1/P2 AP results exist yet.

## 0. Scientific aim and evidence

The accepted full 100% MIX50 Candidate Conflict Graph analysis found that MM Native mean AP is **51.902**, while known-executed-quality greedy NMS + known-quality ranking reaches **83.029**. Q-NMS/native rank is 62.737; native NMS/Q-rank is 73.156. These oracle numbers are **diagnostics using test-set CAD/Dex-Net truth**, not deployable gains, nor globally optimized mathematical upper bounds. Oracle Q never enters model training.

Current independent GVAR reader probes indicate nonzero local evidence value, but GVAR has not demonstrated stable advantage over Point on common physical candidates. RARE-style pairwise improved Mean AP by about +0.064 pp over matched MSE in MIX50, mostly through cross-query ranking; Novel worsened slightly. The NMS pair fine-tune did not improve Novel. Thus two competing explanations remain:

**H1, inadequate readout / supervision:** the frozen GVAR feature already encodes needed action quality, but the scalar scorer and training selection fail to use it.

**H2, missing physical evidence:** frozen GVAR features lack sufficient local contact/clearance/closing geometry to infer exact CAD quality under ambiguous RGB-only observations.

**H3, structured selection:** even if Q were predicted well, local greedy selection loses value due to candidate conflict and global/per-object budgets.

Do not add a large GNN before distinguishing H1/H2/H3. P0 tests feature decodability; P1 tests selection-relevant data; P2 tests global combinatorial selection conditional on known Q.

## 1. Identity, data and leakage contract

- Shared frozen backbone: historical **50% Original + 50% GN-Trans Point RayRoPE + GVAR36 e15**, not a fresh e19 model. Expected historical SHA256: `2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`; verify actual file in the bound config.
- Existing MIX50 RARE **train-index** contains 25,600 frames, each 32 query families ×4 insertion depths with exact physical labels. Per-frame arrays have 64D post-GVAR score-branch features, base CDF expected-utility logit, real 17D executed grasp, family/depth ID and six-μ quality labels.
- **P0 and P1** train probes only on GraspNet train scenes 0–79, select/check validation on 80–89, then perform one-time scorer confirmation on 90–99. Both Original and GN-Trans domains are used. **The underlying frozen base has already seen all 100 train scenes**, so the split is new-scorer-disjoint, *not* unseen-scene backbone generalization.
- Official GraspNet scenes 100–189 are strictly forbidden in P0/P1 fit, validation, threshold/epoch selection and feature pretraining. They are used only for post-freeze test AP and P2 *offline oracle*.
- Freeze original RGB/DPT/RayRoPE/GVAR/CDF, view, angle, camera-frame center and per-depth width. No CAD labels in the network input. GN-Trans training pipeline may still use its legacy virtual depth for workspace/crop; do not claim the entire data preparation is strictly RGB-only.
- S2 scorer is **scalar**, not six independent friction probabilities. Its predicted expected utility is not a calibrated distribution simply because sigmoid is used.
- The existing `candidate_conflict_oracle_20261112_all_splits.zip` contains only all-split summaries (JSON, CSV, Markdown), **not** raw 64D features, full graph caches or action labels. All raw training/test artifacts must be inspected and loaded on 117.

## 2. P0 — What is decodable from frozen GVAR?

### 2.1 Matched-capacity probes

All rows use exactly the same `QualityProbe` architecture (64 hidden, 2-layer scalar residual MLP), optimizer, training frames, seed and 10 epochs. Masked groups keep model parameter count constant.

| Arm | Native expected-Q logit | Depth one-hot | Executed action geometry | 64D post-GVAR feature | Purpose |
|---|---|---|---|---|---|
| `score` | ✓ | ✓ | — | — | Native score/depth recalibration control |
| `score_geometry` | ✓ | ✓ | ✓ | — | What does final action pose/width explain? |
| `score_feature` | ✓ | ✓ | — | ✓ | What did GVAR encode beyond native score? |
| `score_feature_geometry` | ✓ | ✓ | ✓ | ✓ | Combined decodability |
| `score_feature_permuted` | ✓ | ✓ | — | shuffled within frame | Negative feature–action alignment control |

Physical geometry input = actual width, gripper height, insertion, R(9), XYZ center(3), not CAD labels or raw observed depth. Features are the cached **post-reader score branch 64D vector**. P0 does NOT independently assess internal pre-reader tokens or full role-specific contact feature maps; add them only if P0 indicates the need.

Use matched capacity, multiple predetermined training seeds in confirmatory experiments; the default single seed is a pilot. The loss is 10×MSE(sigmoid(logit), Q) by default; `--pair-weight 0.5` is an optional **separate** objective control, not the P0 primary result.

**Primary diagnostics:** scene-balanced confirmation quality MSE, selected-Q/regret within four-depth family, pairwise disagreement where true Q differs; deployment-like full-query AP with all stage settings frozen. Distinguish same-frame rank gains from across-query gains with `both/depth_only/rank_only`.

### 2.2 Run

First confirm the old MIX50 RARE cache index and bound config path. Use a fresh result directory:

```bash
git fetch origin exp/quality-decoding-decision-p0-p2
git worktree add /home/robotarm/EconomicGrasp-quality-p0p2 \
  -b codex/quality-p0p2-exec origin/exp/quality-decoding-decision-p0-p2
cd /home/robotarm/EconomicGrasp-quality-p0p2
git status --short
git worktree list
find .. -name AGENTS.md -print

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/quality_p0p2_20261113
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export BOUND_CONFIG=/ABS/path/to/verified_mix50_gvar_bound.json
export RARE_TRAIN_INDEX=/ABS/path/to/rare_mix50/point_gvar/cache/train_index.json
export DEVICE=cuda:0

bash scripts/quality_decision_p0p2/run.sh check
bash scripts/quality_decision_p0p2/run.sh p0-fit
```

Only after the five modes have independently completed (loss finite, shared frame counts, same optimizer budget), run confirmation **without optimizing on those scenes**:

```bash
for MODE in score score_geometry score_feature score_feature_geometry score_feature_permuted; do
  python -m diagnostics.quality_decision.confirm \
    --weights "$RUN_ROOT/p0/$MODE/best.pt" \
    --index "$RARE_TRAIN_INDEX" --strategy uniform \
    --output "$RUN_ROOT/p0_confirmation/$MODE" --device "$DEVICE"
done
```

Record train/validation/confirmation scene IDs, base checkpoint SHA, cache index hash, each scorer best epoch, mean quality MSE and selected Q on the ten scorer-held-out train scenes. Do not select the best *mode* by official test Novel AP.

**Decision gate:**
- `score_feature` beats matched `score` reliably on confirmation AND improves Novel AP on a frozen scorer: evidence of unexploited post-GVAR information.
- `score_geometry` improves while features do not: evidence for action-parameter conditioning, not necessarily richer RGB evidence.
- Neither improves: might be insufficient features, limited labels/capacity or poor transfer; not proof GVAR contains no action evidence.
- Feature-permuted control matches true features: question whether GVAR features supply action-specific signal.

## 3. P1 — Learn from selection-relevant actions, matched label budget

### 3.1 Why another sampling experiment?

Previous RARE trained on uniformly sampled 32 families per frame and improved full AP only marginally; full test rankings are dominated by top queries after sensor collision/NMS. P1 compares **64 families/arm**, both with four depths and the same images:

- `uniform`: 64 uniformly sampled query IDs.
- `decision`: 32 **shared uniform** IDs + 32 decision-related IDs drawn from high-ranked native NMS-suppressed alternatives, top survivors and fallbacks.

Selection reads native model scores and sensor-filtered geometry **without consulting exact CAD Q labels**. Each candidate's full four-depth executed grasps is then labeled using the official evaluator. Unique physical actions across arms are labeled once.

Within a frame, the two arms have identical query and action budgets (64×4=256 labels/arm); about 32 shared query families give a direct paired comparison. They do **not** necessarily contain exactly the same physical actions; it is a training-distribution test, not a fixed-action ranking comparison.

Mine train scenes 0–99, four frames/scene/domain, **800 input images** total. All 100 scenes are needed for 80/10/10 scorer train/validation/confirmation. The exact-label union per image is bounded by 96×4=384 actions (before physical dedup). Benchmark CAD throughput and disk before all 800 images. Start with a few representative frames for a smoke in an **independent output root**.

### 3.2 Four matched training arms

| Sampler | Pointwise loss | Pairwise loss | New head features |
|---|---|---|---|
| U-MSE | 10×quality MSE | 0 | post-GVAR+action geometry |
| U-RARE | 10×quality MSE | 0.5×non-tie logistic pairs | same |
| D-MSE | 10×quality MSE | 0 | same |
| D-RARE | 10×quality MSE | 0.5×non-tie logistic pairs | same |

All start with the same seed and architecture, 10 epochs, four frames/step and 64 families/frame. Comparing U-MSE vs D-MSE isolates sampled-data distribution at fixed budget; U-RARE vs D-RARE tests the same with pair supervision. Pair counts/loss contributions must be logged since the number of eligible pairs can differ even with equal action counts.

### 3.3 Run

```bash
# One GPU sequentially, or shard on independent GPUs with identical RUN_ROOT.
SHARD_COUNT=3 SHARD_INDEX=0 DEVICE=cuda:0 bash scripts/quality_decision_p0p2/run.sh p1-mine
SHARD_COUNT=3 SHARD_INDEX=1 DEVICE=cuda:1 bash scripts/quality_decision_p0p2/run.sh p1-mine
SHARD_COUNT=3 SHARD_INDEX=2 DEVICE=cuda:2 bash scripts/quality_decision_p0p2/run.sh p1-mine

bash scripts/quality_decision_p0p2/run.sh p1-index
bash scripts/quality_decision_p0p2/run.sh p1-fit
```

Validate combined cache has exactly 800 frame IDs ×2 arms =1600 index entries, 100 train-scene IDs in both modalities, 256 actions/arm, correct SHA/17D identities, finite labels and no incomplete families. The default script refuses overwriting old caches and scores.

One-time confirmation after training:

```bash
for STRATEGY in uniform decision; do
  for LOSS in mse rare; do
    python -m diagnostics.quality_decision.confirm \
      --weights "$RUN_ROOT/p1/heads/\${STRATEGY}_\${LOSS}/best.pt" \
      --index "$RUN_ROOT/p1/cache/index_merged.json" \
      --strategy "$STRATEGY" \
      --output "$RUN_ROOT/p1_confirmation/\${STRATEGY}_\${LOSS}" \
      --device "$DEVICE"
  done
done
```

**Fairness:** use a fixed selection rule for the four checkpoint epochs based solely on validation 80–89; report confirmation scenes 90–99 once. No changing pair weight or selected training scenes after Novel test results.

## 4. P0/P1 official deployment-like test and AP comparisons

The probe inference script returns the same 1024 native queries and native geometry; only its scalar quality scores change insertion choice and/or cross-query priority. Collision prefilter and official eval.py are unchanged. Tests use **Original** GraspNet and must report Seen/Similar/Novel separately.

### Quick 10% protocol (30 scenes ×26 frames/split)

```bash
export PROBE_WEIGHTS="$RUN_ROOT/p1/heads/decision_rare/best.pt"
export LABEL=decision_rare
export INTERVAL=10

bash scripts/quality_decision_p0p2/run.sh infer
bash scripts/quality_decision_p0p2/run.sh eval
```

For full **100%** GraspNet, set `INTERVAL=1` in a **new** `RUN_ROOT` (or use the Python module directly with a new output root), since results may not be overwritten or mixed. Each split must then have 7,680 frames and official AP array `(30,256,50,6)`.

Compare two complete matched runs with:

```bash
python -m diagnostics.quality_decision.compare \
  --left "test_seen=/ABS/LEFT/test_seen" \
  --left "test_similar=/ABS/LEFT/test_similar" \
  --left "test_novel=/ABS/LEFT/test_novel" \
  --right "test_seen=/ABS/RIGHT/test_seen" \
  --right "test_similar=/ABS/RIGHT/test_similar" \
  --right "test_novel=/ABS/RIGHT/test_novel" \
  --policy both --output /ABS/report/paired_ap.json --bootstrap 5000
```

Compare on **same** base checkpoint, frame set, random sampling seed, collision settings and 17D output contract; report 30-scene paired CIs per split. If P1 improves only rank_only but not depth_only or selected Q, do not claim it repaired insertion discrimination.

## 5. P2 — Global-budget selection under known true quality

P2 must use full **test-set oracle labels only for offline analysis**, never for P0/P1 training.

Use the original accepted full conflict graph/label caches for MM, MR, RM, RR, each split separately. The current uploaded aggregate ZIP is insufficient: need `graph_manifest.json`, `label_manifest.json`, per-frame graph `.npz` and exact-label `.npz`. No network re-training required.

Two solver regimes:

**P2a: Full-pool global quality greedy.** Process all candidate nodes, enforce mutual NMS conflict exclusions and global 50/per-object 10 caps during selection, order by exact Q. It is a **feasible heuristic**, not an exact optimizer. Re-score the final chosen set with the existing official AP simulation. Compare against the existing Q-greedy NMS→top-10/object→top-50 pipeline. Difference isolates when applying global budgets changes greedy decisions.

**P2b: Capped-subpool CP-SAT AP optimization.** Truncate explicitly to top 96 candidates by true Q (with native confidence as tie-breaker); impose conflict constraints, max 50 slots and max 10/object. Assign each candidate to at most one rank; optimize the **six-μ mean AP rank weights** derived from the official 1...50 cumulative success definition:

\[
w_r=\frac{1}{50\cdot6}\sum_{k=r}^{50}\frac1k,\qquad
\max \sum_{i,r} w_r\Big(\sum_{\mu=1}^6 b_i^{(\mu)}\Big)y_{ir}.
\]

Coefficients are integer-rounded for CP-SAT. This optimizes rank-weighted mean AP on the **restricted candidate pool**, *not* on the full action space. Check solver status (OPTIMAL vs FEASIBLE), time and objective bound, exact rank reconstruction and official AP match. Compare against **quality greedy on the same 96-node pool**, never compare restricted CP-SAT against full-pool greedy as if only the algorithm changed. CP-SAT may not finish or be installed on 117; first run the full-pool heuristic and the separate capped-pool smoke.

```bash
# Optional dependency; only for the capped CP-SAT subset.
python -c "import ortools.sat.python.cp_model"

export GRAPH_ROOT=/ABS/candidate_conflict/test_novel/graph
export LABELS_ROOT=/ABS/candidate_conflict/test_novel/labels
export SPLIT_NAME=test_novel
export METHOD=MM

# Per-split pilot: two pre-fixed frames per scene, 60 frames.
P2_SOLVER=off FRAMES_PER_SCENE=2 bash scripts/quality_decision_p0p2/run.sh p2
P2_SOLVER=cp_sat FRAMES_PER_SCENE=2 MAX_SOLVER_NODES=96 SOLVER_SECONDS=10 \
  bash scripts/quality_decision_p0p2/run.sh p2

# Full graph heuristic: all 30*256 frames in a separate new output root.
P2_SOLVER=off FRAMES_PER_SCENE=0 bash scripts/quality_decision_p0p2/run.sh p2
```

Repeat for all splits and methods after the first held-out protocol is accepted. If `--frames-per-scene=2`, the result is **not** the official all-split AP; use equal frame weighting/scene bootstrap, report “fixed-frame pilot”.

**Decision gate:** if the capped subpool's CP-SAT result is almost equal to its same-subpool Q-greedy baseline, do not claim learned set-optimization is necessary. If large and stable, next investigate learned candidate interactions; if both are low on Novel, prioritize candidate quality/coverage and contact evidence.

## 6. Stop conditions and acceptance

1. Identify exact GVAR base checkpoint, its e15 training provenance and strict source loader. No `strict=False`, checkpoint metadata fabrication or mixing Point-only as GVAR.
2. P0: 5 probes share identical sample IDs, optimizer budget, parameter counts and predetermined seeds. New scorer train/val/confirm scenes disjoint; no test labels.
3. P1: each sample exactly 64 families×4 depths in both arms; 32 shared uniform anchors; candidate decisions computed **before** CAD labels; union labeled once, exact width/center/R/d/label identity enforced.
4. Cache: source model, frame manifest, physical IDs and label hashes match. Incomplete shards may not be silently treated as complete.
5. P0/P1: infer outputs original 1024 native query geometries and frozen depth-dependent widths, with original collision filter. Full AP refuses smoke or missing frames.
6. P2: graph/label manifest identities match and all loaded graph and label frame hashes validate. Only test-set oracle uses CAD truth.
7. P2: greedy/CP-SAT selected candidates obey no-edge, global K=50 and per-object K=10. CP-SAT result only called exact if status OPTIMAL and only relative to the capped subset.
8. P2: run parity against existing native AP and quality-NMS values on the same frames. If AP mismatches, stop and audit original object assignment, NMS, per-object limits and six friction bits.
9. Do not use test Novel AP to pick P0/P1 hyperparameters or mining policies. The already repeatedly inspected MIX50 test set remains **exploratory**, so independent scenes/seeds or robot trials are required for confirmatory TRO claims.
10. Deliver: `p0/*/protocol.json,history.json,best.pt`, `p0_confirmation/*/summary.json`, `p1/cache/index_merged.json`, `p1/heads/*`, official AP arrays and paired scene reports, `p2/*/per_frame.csv,summary.json`, CPU/CUDA/CAD smoke logs, source/checkpoint SHA, and a Chinese P0–P2 analysis report with explicit positive and negative findings.

**Scope note:** P0 does not independently capture raw pre-GVAR contact roles; P1's decision sampling is a targeted data hypothesis, not an AP-aligned differentiable training objective; P2's solver is not a proof of best possible physical grasp set. These limits must be written into the conclusions.
