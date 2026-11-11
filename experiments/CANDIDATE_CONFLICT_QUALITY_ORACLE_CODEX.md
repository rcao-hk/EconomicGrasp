# Full Candidate Conflict Graph + Quality-aware Selection Oracle — Codex Work Plan

**Date:** 2026-11-11  
**Branch:** `exp/candidate-conflict-quality-oracle`  
**Base:** `exp/rare-ranking-mix50@36b35c0a750930ad98da53f6a525ae15b25681bf`  
**Scope:** full 100% Original GraspNet test split candidate graphs, exact-action quality labels, NMS/selection oracles.  
**No model training, no changes to production inference, no changes to official GraspNet evaluator.**

## 0. Why this diagnostic exists

The source report is the user's uploaded **RARE MIX50 → NMS Pair summary, updated 2026-11-11**. It records:

- MIX50 full evaluation: each Seen/Similar/Novel split has 7,680 frames; official AP arrays are `[30,256,50,6]`.
- Four cross combinations MM/MR/RM/RR were evaluated over all 90 scenes. Full Novel interaction was near zero; the previous “large negative interaction” interpretation was corrected.
- On 20 deliberately selected frames, 21,603 deduplicated executed actions were independently labelled and 1,971 records were comparable to official quality records; these are **not** full-test labels.
- On eight extreme Novel frames, fixed MM final query membership almost restored RR AP (31.9286 vs 31.6835), and examples showed unchanged high-Q actions removed by NMS after score-order changes.
- The small NMS-pair fine-tune did not improve Novel. Its 16-frame validation Q/MSE did not translate to AP.
- The report recommends checking full surviving sets, suppression relations, representative quality and whether the original-MSE mined conflict graph transfers after fine-tuning.

This work plan tests a more basic question before a learned selector is built:

> **Given the exact executed-action quality of every candidate in the current candidate pool, how much headroom remains in greedy GraspNet NMS and its downstream per-object top-10 / global top-50 selection?**

The purpose is to decide whether the bottleneck is primarily:
1. candidate quality (even the best feasible set is weak);
2. greedy NMS representative choice;
3. global ranking after NMS;
4. or insufficient candidate coverage.

## 1. Non-negotiable definitions

### 1.1 What an edge means

The graph is the **GraspNet NMS suppression graph**, not a general physical incompatibility graph.

For two candidates \(g_i,g_j\), create an undirected edge iff:

\[
\|t_i-t_j\|_2\leq 0.03\ \mathrm m
\quad\land\quad
\arccos\left(\frac{\operatorname{tr}(R_i^\top R_j)-1}{2}\right)
\leq 30^\circ.
\]

These are the default official GraspNet NMS thresholds. Width and insertion depth do not enter this NMS edge rule. An edge means “one may suppress the other under score-ordered greedy NMS”; it **does not** mean the two actions are physically impossible to execute together or have the same object identity.

The builder runs the installed `GraspGroup.nms(0.03, radians(30))` on every frame and requires exact row-level parity with the graph-based greedy simulation (max absolute error ≤1e-6). If parity fails, stop; do not silently substitute a different NMS implementation.

### 1.2 Candidate input stage

Inputs must be the full candidate arrays **after the experiment's sensor/model-free collision prefilter but before official GraspNet NMS**:

- one `N x 17` array per scene/frame;
- path layout: `ROOT/scene_XXXX/realsense/NNNN.npy`;
- width clipped to the same `[0,0.10]` m range as official evaluation;
- score, width, height, depth, rotation, translation, object id remain in the standard 17D order.

Do not use:
- already NMS-filtered candidates;
- per-object top-10 or global top-50 outputs;
- final AP-ranked component rows;
- a visualization-only subset.

If only post-NMS dumps exist, the full suppression graph cannot be reconstructed. Report the missing pre-NMS artifacts and stop; do not call a reconstructed graph “full”.

### 1.3 Exact quality labels

Every unique physical action in the union of MM/MR/RM/RR candidates must be evaluated with the repository's `ExactGraspNetActionEvaluator(fc_mode="official", strict=True)`:

- same scene/frame object assignment;
- same CAD collision and table collision;
- same empty-grasp test;
- same six friction thresholds;
- no legacy 5 mm label mask;
- no GT depth or GT geometry is passed into the network (the evaluator is offline only).

Deduplicate labels by a hash of the **executed geometry** `grasp[1:16]` after official width clipping; confidence and object-id columns are excluded. If two methods share the same executed action, label it once. A hash match with different geometry must fail.

Define \(b_i^\mu\in\{0,1\}\) as the official success bit and \(Q_i=\sum_\mu b_i^\mu\in\{0,\ldots,6\}\). Collision/empty actions have all-zero bits.

### 1.4 The oracle policies

The scripts produce five distinct policies. Do not merge their interpretations.

| Policy | NMS order | Per-object top-10 / global top-50 order | Interpretation |
|---|---|---|---|
| Native | model confidence | model confidence | Official baseline; must reproduce official AP |
| Q-NMS / native rank | exact \(Q_i\) | original model confidence | Isolates representative replacement under NMS |
| Q-rank / native NMS | model confidence | exact \(Q_i\) | Isolates downstream ranking, with native NMS set fixed |
| Q-NMS / Q-rank | exact \(Q_i\) | exact \(Q_i\) | Optimistic scalar-quality priority |
| Per-\(\mu\) success oracle | exact \(b_i^\mu\) separately for each \(\mu\) | same \(b_i^\mu\) | Very optimistic threshold-specific ceiling; six different policies, not one deployable ranking |

Additionally, a **fixed-cardinality component set oracle** solves a maximum-\(Q\) independent set inside each conflict component, constrained to retain the same number of candidates as native greedy NMS retained in that component. It is exact only for components up to `--max-exact-component-nodes` (default 20); larger components fall back to the native selection. The report must show exact-node coverage and must not call the hybrid result a globally exact MWIS oracle unless every component was solved exactly.

All oracle policies use true labels and are **diagnostic upper bounds, not deployable methods**. Q is mean success over six friction thresholds, not an exact optimizer of mean AP. The per-\(\mu\) oracle is deliberately more optimistic and must be reported separately.

## 2. Methods and frame protocol

Primary methods are the four full MIX50 cross combinations:

- `MM`: MSE depth + MSE query ranking.
- `MR`: MSE depth + RARE query ranking.
- `RM`: RARE depth + MSE query ranking.
- `RR`: RARE depth + RARE query ranking.

Use the exact 100% Original GraspNet outputs from the 2026-11-11 full cross diagnostic. Each split contains 30 scenes × 256 frames = **7,680 frames**. The three splits cover scenes 100–189.

Do not use the 10% MIX10/MIX50 dumps or the 20-frame extreme-case selection as a substitute for the full 100% candidate set. The 20-frame labels may be reused only if the exact 17D executed geometry hash, width clipping, evaluator mode, scene/frame, and all label fields match; otherwise relabel.

The four methods may share many physical actions. Label the union once per frame, but keep separate method-specific graphs, scores, NMS order and selected sets.

## 3. Code delivered

- `diagnostics/candidate_conflict/core.py`
  - NMS conflict edges, connected components, greedy NMS, official AP replay;
  - exact-quality priorities and fixed-cardinality component independent-set solver;
  - scene-cluster bootstrap and artifact hashing.
- `diagnostics/candidate_conflict/build.py`
  - full per-method, per-frame graph build;
  - input-stage assertion, width clipping, official NMS parity and resumable graph cache.
- `diagnostics/candidate_conflict/label.py`
  - union/deduplicate physical actions across methods;
  - official CAD/Dex-Net exact labels, scene-level multiprocessing, resumable label cache.
- `diagnostics/candidate_conflict/analyze.py`
  - native AP replay parity;
  - NMS/quality/set oracles, per-\(\mu\) ceilings, suppression Q-loss;
  - per-frame/per-scene outputs and scene-cluster bootstrap.
- `tests/test_candidate_conflict.py`
  - CPU-only graph, greedy NMS, component oracle, exact-action bits, official AP formula and CLI tests.

No production model, decoder, training loss, inference path, GraspNet API source or official evaluator is modified.

## 4. Codex execution sequence

### 4.0 Worktree and artifact mapping

Read all applicable `AGENTS.md`, check `git status --short`, `git worktree list`, GPU/CPU/disk resources, and inspect the existing result tree before running anything. Do not kill jobs or delete other experiments.

```bash
git fetch origin exp/candidate-conflict-quality-oracle
git worktree add /home/robotarm/EconomicGrasp-candidate-oracle \
  -b codex/candidate-conflict-oracle origin/exp/candidate-conflict-quality-oracle
cd /home/robotarm/EconomicGrasp-candidate-oracle

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_candidate_conflict.py' -v
python -m compileall -q diagnostics/candidate_conflict tests/test_candidate_conflict.py
```

Inspect the full MIX50 output roots:
- `/data/robotarm/result/grasp/rgbgrasp/rare_mix50_20261110`
- `/data/robotarm/result/grasp/rgbgrasp/rare_mix50_20261110/reports/cross_20261111`
- `/data/robotarm/result/grasp/rgbgrasp/rare_nms_pairs_pilot_20261111`

Build an explicit mapping for each split:

```bash
export MM_DUMP_ROOT=/ABS/PATH/MM/pre_nms_dumps
export MR_DUMP_ROOT=/ABS/PATH/MR/pre_nms_dumps
export RM_DUMP_ROOT=/ABS/PATH/RM/pre_nms_dumps
export RR_DUMP_ROOT=/ABS/PATH/RR/pre_nms_dumps

export MM_AP=/ABS/PATH/MM/ap_test_seen_realsense.npy
export MR_AP=/ABS/PATH/MR/ap_test_seen_realsense.npy
export RM_AP=/ABS/PATH/RM/ap_test_seen_realsense.npy
export RR_AP=/ABS/PATH/RR/ap_test_seen_realsense.npy

export DATASET_ROOT=/data/robotarm/dataset/graspnet
export OUT=/data/robotarm/result/grasp/rgbgrasp/candidate_conflict_oracle_20261111
```

The paths above are placeholders. Verify the exact roots and AP files from execution manifests/logs. Each dump root must contain all 7,680 pre-NMS, post-sensor-filter 17D files for the selected split. If the four strategies use a different directory layout, pass the correct method-specific subdirectory; do not modify candidate arrays to make them match.

### 4.1 CPU tests and one-frame smoke

```bash
python -m unittest tests.test_candidate_conflict -v
python -m compileall -q diagnostics/candidate_conflict
```

Build a one-frame graph first:

```bash
python -m diagnostics.candidate_conflict.build \
  --split test_novel --camera realsense \
  --method "MM=$MM_DUMP_ROOT" --method "MR=$MR_DUMP_ROOT" \
  --method "RM=$RM_DUMP_ROOT" --method "RR=$RR_DUMP_ROOT" \
  --official-ap "MM=$MM_AP" --official-ap "MR=$MR_AP" \
  --official-ap "RM=$RM_AP" --official-ap "RR=$RR_AP" \
  --input-stage sensor_filtered_pre_nms \
  --scenes 167 --anno-start 196 --anno-stop 197 \
  --output "$OUT/smoke/test_novel"
```

Formal graph construction requires official NMS parity. If the smoke fails parity, inspect the installed `grasp_nms` extension and threshold/rotation conventions. Do not relax the tolerance or replace the official implementation without documenting the discrepancy.

Run one-frame exact labels:

```bash
python -m diagnostics.candidate_conflict.label \
  --graph-root "$OUT/smoke/test_novel" \
  --dataset-root "$DATASET_ROOT" --camera realsense \
  --output "$OUT/smoke/test_novel_labels" --workers 1
```

For a smoke analysis, the official AP array is the full split array, but only one frame is present; use `--allow-partial --skip-official-parity`. This is a pipeline smoke only, not a result.

### 4.2 Full graph construction: all 7,680 frames per split

Run one split at a time. Use a new output directory per split; `--resume` only when resuming the exact same protocol and source files.

```bash
for SPLIT in test_seen test_similar test_novel; do
  python -m diagnostics.candidate_conflict.build \
    --split "$SPLIT" --camera realsense \
    --method "MM=$MM_DUMP_ROOT" --method "MR=$MR_DUMP_ROOT" \
    --method "RM=$RM_DUMP_ROOT" --method "RR=$RR_DUMP_ROOT" \
    --official-ap "MM=$MM_AP" --official-ap "MR=$MR_AP" \
    --official-ap "RM=$RM_AP" --official-ap "RR=$RR_AP" \
    --input-stage sensor_filtered_pre_nms \
    --output "$OUT/$SPLIT/graph" --resume
done
```

The builder stores one compressed graph per method/frame, including source dump SHA, physical action IDs, edge list, component IDs, native greedy survivors, suppressor assignment, and official NMS parity. Formal manifest must show:
- 7,680 frames for each method;
- 30 scenes × 256 annotations;
- no missing or duplicate frames;
- all source file hashes verified;
- NMS parity max absolute error ≤1e-6 for every frame.

The edge count is not a count of physical collisions. It is only the official NMS suppression relation.

### 4.3 Exact-action labels for the union of all methods

Before launching the full label stage, use the graph manifest to report:
- candidates per method/frame;
- unique physical actions after cross-method deduplication;
- total exact-action count;
- estimated storage and wall-clock time from a representative 20–50 frame timing run.

The exact label workload may be much larger than the earlier 21,603-action/20-frame audit. Do not extrapolate from the 128-action-per-frame RARE training cache without measuring this candidate workload. If resources are insufficient, report the measured throughput and propose a staged/budgeted oracle; do not silently label only top-ranked candidates and call it full.

```bash
for SPLIT in test_seen test_similar test_novel; do
  python -m diagnostics.candidate_conflict.label \
    --graph-root "$OUT/$SPLIT/graph" \
    --dataset-root "$DATASET_ROOT" --camera realsense \
    --output "$OUT/$SPLIT/labels" \
    --workers 12
done
```

Each worker processes a scene and reuses its CAD/Dex-Net scene model cache. Adjust worker count only after measuring CPU memory and force-closure throughput. Labels are written incrementally with source graph hashes; completed matching frames can be resumed. Never reuse a label cache if the physical action hash, width clip, evaluator source, scene/frame or graph manifest differs.

### 4.4 Analyze each full split and verify official AP parity

```bash
for SPLIT in test_seen test_similar test_novel; do
  python -m diagnostics.candidate_conflict.analyze \
    --graph-root "$OUT/$SPLIT/graph" \
    --labels-root "$OUT/$SPLIT/labels" \
    --official-ap "MM=$MM_AP" --official-ap "MR=$MR_AP" \
    --official-ap "RM=$RM_AP" --official-ap "RR=$RR_AP" \
    --max-exact-component-nodes 20 \
    --bootstrap 5000 --seed 0 \
    --output "$OUT/$SPLIT/analysis"
done
```

**Native AP replay parity is a hard gate.** For every method/frame, the simulated native AP matrix must match the official array entry `[scene_index, anno]` within 1e-5. If any method fails:
- stop oracle interpretation;
- inspect width clipping, candidate input stage, official NMS parity, object assignment, per-object top-10, global top-50 threshold/tie handling, collision/empty and friction labels;
- do not proceed by widening tolerance.

The analyzer reports:
1. native official AP replay;
2. Q-NMS representative oracle while retaining native downstream rank;
3. Q-rank with native NMS fixed;
4. Q-NMS plus Q-rank;
5. fixed-cardinality component set oracle (exact only for components within the size cap);
6. six separate per-friction success-bit oracles;
7. suppressed-candidate Q inversions, component sizes, survivor counts, per-frame/per-scene AP and scene-bootstrap intervals.

The per-\(\mu\) oracle has six different rankings. It is not one scalar policy and must not be presented as a deployable model. The component oracle optimizes sum-Q at fixed survivor count, not official AP; report its AP only as the result of applying official downstream caps to that diagnostic set.

### 4.5 Cross-split aggregation and interpretation

Keep the three split outputs separate first. Then prepare `RESULTS_CN_ALL_SPLITS.md` with:
- per-method and per-split graph size/edge/component statistics;
- native AP parity status;
- Q-NMS representative-only AP gain;
- Q-rank-only AP gain;
- Q-NMS+Q-rank gain;
- component-set oracle gain and exact-node coverage;
- per-\(\mu\) optimistic ceilings;
- scene-cluster 95% CIs;
- candidate counts and exact-label cost.

Do not average all candidates or frames as IID samples. The scene is the bootstrap unit. The 30-scene interval does not include training-seed variance.

## 5. Required scientific interpretation

Use the following decision logic:

1. **Large Q-NMS representative-only gain:** greedy confidence order is choosing poor representatives inside NMS conflicts. A learned conflict-aware selector becomes justified.
2. **Small Q-NMS gain but large Q-rank-only gain:** the main issue is downstream per-object/top-50 priority, not NMS representative choice.
3. **Large component-set gain beyond Q-greedy NMS:** greedy NMS is structurally leaving quality on the table even with true candidate Q; set-level interaction/selection is worth studying.
4. **Small all-oracle gains:** current candidate pool/physical quality is the main limitation; a new selector cannot recover missing high-quality actions.
5. **Per-\(\mu\) ceilings large but mean-Q oracle small:** a single scalar Q objective may not capture threshold-specific AP trade-offs.
6. **Oracle gains concentrated in a few scenes:** do not generalize to all Novel scenes; inspect scene-level variation and object/candidate density.

Do not call an oracle gain a model improvement. Do not call an NMS edge a physical incompatibility. Do not infer that NMS should be removed. The official evaluator's NMS/top-10/top-50 remains unchanged in every simulated AP calculation.

## 6. Expected deliverables and acceptance

- [ ] CPU unit tests pass.
- [ ] Single-frame official NMS parity and exact-label smoke pass.
- [ ] Full graph manifests: all 90 scenes × 256 frames for each of MM/MR/RM/RR.
- [ ] Full deduplicated exact-action labels with SHA-verified frame caches.
- [ ] Native AP replay parity passes for all methods and all frames (max error ≤1e-5).
- [ ] Per-frame/per-scene graph and AP outputs are complete.
- [ ] Component exact-node coverage is explicitly reported; hybrid fallback is never called globally exact.
- [ ] All six friction thresholds are reported.
- [ ] Scene-bootstrap CIs and per-scene contribution tables are generated.
- [ ] `RESULTS_CN_ALL_SPLITS.md` states whether the current candidate pool, greedy NMS, or downstream ranking is the dominant limitation.
- [ ] No training, model changes, evaluator changes, test-set-based hyperparameter tuning or deletion of existing outputs.

## 7. Known limitations

- NMS conflict edges are geometric threshold relations, not ground-truth physical conflicts.
- Exact labels for every pre-NMS candidate may be computationally expensive; measure before launching all 23,040 frames × four strategies. Cross-method action deduplication is expected to reduce repeated labels.
- Q-NMS uses average six-threshold quality; AP is threshold-specific.
- Per-\(\mu\) success oracle is intentionally optimistic and not a single ranking.
- Fixed-cardinality component oracle is exact only for components up to the configured size; report exact-node coverage and fallback count.
- Oracle quality assumes the current candidate pool. It says nothing about grasps the generator never proposed.
- Scene bootstrap is conditional on the fixed model/checkpoint and does not estimate training-seed variability.
- The 20-frame extreme-case audit was deliberately selected and must not be used to estimate full-split incidence.
