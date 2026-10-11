# GVAR P0/P1 follow-up: scene-factorial decision and conditional probe audit

Date: 2026-10-11. Repository: rcao-hk/EconomicGrasp, branch: exp/gripper-volume-action-reader.

**This document defines analysis scripts to execute; it does not claim that the new six-model results or full server analyses have been run.** Historical training/inference results are read-only. Never delete, rename, reset or overwrite old results; do not touch main.

## Scientific contract

The attached GVAR_P0_P1_RESULTS_ZH.md establishes:

- P0-A is complete: volume_rel minus volume_fixed = +0.6965 mean AP pp and +1.4169 Novel AP pp, with scene-paired intervals. This comparison changes *both* support motion and relative geometry and therefore is not a factorial interaction.
- The last supplied update reports volume_fixed_rel training had **started** on 2026-10-10; it does NOT provide e19 official AP. Do not infer completion from its background process.
- P1 diagnosed 20 outcome-selected frames (3 worst/1 best in each of Novel scenes 0160/0165/0167/0174/0189), volume_rel Top-20 post-collision grasps, 36 probes/action, 14,400 probes, 5,207 rendered-reference-valid, 17.789 mm valid-probe MAE. Local predicted-depth STD and gradient are proxies, NOT learned/correctly calibrated uncertainty.

**Two new Python entrypoints:**

- analyze_gvar_p0_decision.py — check completion, then run strict six-model scene-paired statistics and existing 2×2 factorial using official e19 AP.
- analyze_gvar_p1_conditional.py — read saved probe_records.csv, frame_probe_summary.csv, selected_frames.json and novel_frames.csv, then report per-scene/role reliability, same-probe dual-reference errors and scene-conditioned frame-AP associations.
- Tests: tests/test_gvar_p0_p1_followup.py.

They require only Python, NumPy, pytest for testing (and the repository's existing analysis modules), **no CUDA/model inference**.

## 1. Isolated source checkout and tests

First verify applicable AGENTS.md instructions, repo SHA and user modifications. Avoid checking out over the running GPU experiment. Example:

~~~bash
cd /home/robotarm/EconomicGrasp
git status --short
git fetch origin exp/gripper-volume-action-reader
git worktree add --detach ../EconomicGrasp-gvar-p0p1-followup origin/exp/gripper-volume-action-reader
cd ../EconomicGrasp-gvar-p0p1-followup
python -m py_compile analyze_gvar_p0_decision.py analyze_gvar_p1_conditional.py
python -m pytest -q tests/test_gvar.py tests/test_gvar_scene_paired.py \
    tests/test_gvar_p0_p1.py tests/test_gvar_p0_p1_followup.py
~~~

If the worktree exists, inspect and reuse it safely rather than removing it. Never stop another experiment.

## 2. P0: readiness followed by strict factorial decision

Expected results root:

~~~bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007

python analyze_gvar_p0_decision.py \
  --root "$ROOT" \
  --output-dir "$ROOT/analysis/p0_decision_20261011_first" \
  --epoch 19 --bootstrap 50000 --seed 20261009 --ci 0.95
~~~

Three outcomes:

1. **pending**: volume_fixed_rel is missing any required e19/e20 checkpoint, its full 21-epoch training log/manifest, or any of the three Original GraspNet e19 AP files and inference manifests. The script writes P0_STATUS.json and P0_DECISION.md **without making up AP or a factorial effect**. Add --require-ready when the calling workflow should fail on pending.
2. **complete**: all six variants (baseline, slot, volume_fixed, volume, volume_rel, volume_fixed_rel) pass the existing canonical frame-fingerprint, training, inference and official NPY-contract checks; all have e0..e20 finite training logs; the five historical Mean AP values match the archived report within 0.02 AP points. It then writes:
   - P0_DECISION.md, P0_STATUS.json;
   - paired_six/paired_summary.csv (6 variants × 4 splits = 24 data rows), paired_six/paired_metrics.csv, scene deltas and source SHA256 audit;
   - factorial/factorial_effects.csv (7 effects × 12 metrics × 4 splits = 336 data rows), factorial/factorial_scene_effects.csv.
3. **invalid**: any historical control, frame index, inference/manifests, checkpoint identity, invalid AP data or nonempty output conflict raises an error. No silent omission.

After pending, use a **new output directory** for the second run; never append to the first status directory.

The factorial matrix:

| Gripper support | No relative geometry | Relative geometry |
|---|---|---|
| Fixed | volume_fixed | volume_fixed_rel |
| Dynamic | volume | volume_rel |

~~~
Interaction = (volume_rel - volume) - (volume_fixed_rel - volume_fixed)
~~~

Report Seen/Similar/Novel/Mean interaction, and Novel μ=0.4 vs μ=0.8 effects with paired scene-bootstrap confidence intervals.

**Interpretation:** A CI overlapping zero is *not equivalence*; non-overlap does *not* establish training-seed stability. Models generate different grasp candidates, so AP gains must not be attributed solely to ranking. The analysis does not rerun inference, training, or evaluator.

## 3. P1: per-scene / per-role reliability, matched references, AP links

Existing saved-data directories (verify these paths before execution):

~~~bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
CASES="$ROOT/analysis/novel_cases_20261010"
PROBES="$ROOT/analysis/probe_reliability_20261010"
OUT="$ROOT/analysis/p1_conditional_20261011"

python analyze_gvar_p1_conditional.py \
  --probe-dir "$PROBES" \
  --cases-dir "$CASES" \
  --output-dir "$OUT" \
  --variants volume_rel \
  --min-probes 15 --min-frames 3
~~~

The program first checks that P1's probe manifest refers to exactly the selection JSON's SHA256, scene/frame AP deltas agree with novel_frames.csv, each (variant, scene, frame, grasp rank, probe index) is unique, validity masks are logically consistent and each frame has the expected 36-probe/action cardinality. Invalid sources raise an error **before** output creation; no imputation.

### P1.1 Conditional reliability by role and scene

- per_scene_role.csv: per-scene×role counts and coverage for projection, foreground, rendered-reference, sensor-reference and joint-reference valid probes; rendered MAE and error-risk rank correlations.
- per_scene.csv, per_role.csv: descriptive scene- and role-aggregated metrics including an average of within-frame Spearman correlations. Do not treat thousands of correlated probes as independent observations.
- per_frame_role.csv, per_frame.csv: frame-level coverage and role-level/error variation.

### P1.2 Matched rendered vs sensor depth errors

Only on the **same projected probe** where both rendered and sensor depths have valid masks:

~~~
paired_error_difference_mm = 1000 *
 (mean(abs(pred - rendered)) - mean(abs(pred - sensor)))
~~~

This compares two absolute errors on the same set. It does **not** measure abs(rendered − sensor) because only absolute prediction errors were saved in the historical CSV. Do not subtract unpaired MAEs with different validity sets.

### P1.3 Evidence coverage and frame-level AP deltas

Join per-frame overall/closing/contact/approach coverage and MAE with the exact AP-selected frame's ΔAP, ΔAP@μ=0.4 and ΔAP@μ=0.8.

ap_associations.csv contains pooled Spearman (descriptive), mean within-scene Spearman (4 frames/scene, unstable) and **scene-mean-centered Pearson** to reduce between-scene confounding. No p-value or claimed causal relation.

All 20 frames were chosen **based on AP outcomes** (5 scenes × 3 worst + 1 best). This subset is not an unbiased sample of Novel scenes. Correlations are exploratory. Local predicted depth STD/gradient must never be described as calibrated uncertainty; missing external sigma must remain missing.

Expected for the reported P1 data with one volume_rel variant: 5 scenes, 20 frames, 14,400 probe rows, 30 scene-role rows, 120 frame-role rows, 6 role rows and 36 AP-association rows.

Artifacts:

~~~text
P1_REPORT.md
P1_AUDIT.json          # exact input file SHA256
per_scene_role.csv
per_scene.csv
per_role.csv
per_frame_role.csv
per_frame.csv
ap_associations.csv
~~~

## 4. Codex acceptance and reporting

Return the actual Git SHA, Python/NumPy versions, commands and tests; P0 readiness status, and if complete, all four factorial cells, interaction CI and Novel μ friction contrasts. For P1 return per-scene/role coverage, matched rendered-sensor error differences, within-scene reliability and frame-AP associations with their limitations. Include output CSV and markdown paths. Do not start GPU training, re-evaluation, unrelated MGF/RayRoPE experiments or an uncertainty-gated grasp head as part of this task.
