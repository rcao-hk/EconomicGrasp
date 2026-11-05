# RayRoPE P1/P2 diagnostics

Branch: `exp/rayrope-p1-p2-diagnostics`  
Base: `main@7a956d37aaa427a32e598f21fdc6b910ed3c2704`

## Scope and status

This branch contains **offline diagnostics for the existing P1/P2 exports**. It does not retrain the model, change checkpoint weights, or modify the normal inference graph.

Implemented scripts:

| File | Purpose |
|---|---|
| `diagnostics/analyze_rayrope_candidates.py` | Failure attribution, six official friction thresholds, ranking/regret, width-conditioned quality, matched width sweeps, paired interventions, scene-clustered bootstrap |
| `diagnostics/analyze_rayrope_depth_error.py` | Registered predicted-vs-GT depth error and its association with native vs reader-perturbed candidate selection |
| `diagnostics/sample_rayrope_candidates_uniform.py` | Deterministic score-independent sampling from a verified complete pre-score-filter candidate pool |
| `diagnostics/rayrope_p1_p2.py` | Earlier generic candidate-table metrics and metric-depth intervention primitives |
| `tests/test_rayrope_candidate_analysis.py` | Tests for failure classes, six-mu labels, width families, depth metrics, paired effects, and uniform sampling |
| `tests/test_rayrope_p1_p2.py` | Tests for the earlier generic diagnostic primitives |

The uploaded 2026-11-06 result report describes 18 frames / 9 scenes and 64 score-stratified candidates per frame. Its actual `candidates.csv`, depth arrays, and manifest were not included in the repository commit. The scripts therefore accept those server-side files as inputs; **no new analysis result is claimed until the commands below are run on the server**.

## 1. Candidate CSV analysis

### 1.1 Metrics

Run:

```bash
python diagnostics/analyze_rayrope_candidates.py \
  --input_csv "$RUN_ROOT/results/candidates.csv" \
  --output_json "$RUN_ROOT/analysis/candidate_diagnostics.json" \
  --output_md "$RUN_ROOT/analysis/candidate_diagnostics.md" \
  --sampling_protocol score_stratified \
  --high_score_quantile 0.95 \
  --width_unit auto \
  --insertion_bins 0,0.01,0.02,0.03,0.04,0.05,inf \
  --bootstrap_samples 5000 \
  --seed 20261106
```

The analyzer produces:

- Mutually exclusive candidate failure classes: empty grasp, collision, force-closure failure, success, unknown.
- Failure attribution for all candidates, the per-frame high-score subset, and each frame's Top-1 candidate.
- Official friction-threshold decomposition at `mu = 0.2, 0.4, 0.6, 0.8, 1.0, 1.2`: all-candidate success, Top-1 success, Top-5 success, and high-score failure.
- Per-frame score/quality Spearman and Kendall-`tau-b`, Top-1/Top-5 quality, candidate-set oracle quality, Top-1 regret, and high-score failure rate.
- Split-level and overall summaries with **scene-clustered bootstrap** intervals. Frames/candidates are not treated as independent scenes.
- Width bins and insertion-depth bins in metres, width-mode summaries when a width-mode column exists, and matched width-sweep regret when the same physical action family is evaluated at multiple widths.
- Paired intervention score change, rank stability, Top-1 changes, and Top-1 quality changes for candidate IDs shared with the reference condition.

### 1.2 Quality and failure conventions

If all six success-bit columns are detected, quality is their arithmetic mean:

```text
quality = (success_mu_0.2 + ... + success_mu_1.2) / 6
```

If the six bits are absent but the exact evaluator's friction score is present, the script follows the GraspNet convention:

```text
success(mu) = friction_score > 0 and friction_score <= mu
```

A scalar quality field is only a fallback and cannot support six-threshold decomposition by itself. The script does not treat AP as per-candidate quality.

Failure-class priority is: empty grasp → collision → force-closure failure → success → unknown. When both collision and empty flags are present, the explicit empty flag takes precedence; pure collision is separated from empty grasp.

High-score candidates are selected **within each frame**, by the configured score quantile (default top 5%). Ties are resolved deterministically by candidate ID.

### 1.3 Width analysis interpretation

- Width-bin results are observational associations. They do not establish that changing width causes the observed quality difference.
- A matched width sweep is only computed when rows share an explicit parent/action-family ID, or have exactly matched center, rotation, and insertion depth with different widths.
- The score-selected width is compared with the best evaluator-quality width within that same family. This is a within-family oracle, not a global action-space oracle.
- If the CSV only contains one width per geometry family, the matched sweep correctly reports zero eligible families.
- If no width-mode column is exported, fixed-width vs predicted-width cannot be separated from this CSV alone.

### 1.4 Column aliases

The parser auto-detects common aliases for sample/frame, scene, split, candidate/action ID, parent/action-family ID, intervention, score, width, insertion depth, center, pixel center, collision, empty, friction score, and six success bits.

If the server CSV uses nonstandard names, provide a JSON map. Example:

```json
{
  "sample": "frame_key",
  "scene": "scene",
  "candidate": "action_uid",
  "condition": "test_condition",
  "score": "predicted_score",
  "width": "executed_width_m",
  "friction": "dexnet_friction_score",
  "collision_or_empty": "collision_or_empty",
  "empty": "empty_grasp",
  "mu_0.2": "fc_success_02",
  "mu_0.4": "fc_success_04",
  "mu_0.6": "fc_success_06",
  "mu_0.8": "fc_success_08",
  "mu_1.0": "fc_success_10",
  "mu_1.2": "fc_success_12"
}
```

Then pass `--columns_json /path/to/column_map.json`. The six success-bit headers are detected independently by their threshold token and success/pass/force-closure marker. The output JSON records the detected schema so header interpretation is auditable.

### 1.5 Insertion-depth-conditioned analysis

The same candidate analyzer also bins the exact insertion-depth field using `--insertion_bins` (default 0, 10, 20, 30, 40, 50 mm and above). These are observational bins, not a controlled insertion-depth sweep. A causal sweep requires the same center, rotation, width, RGB frame, and exact-action evaluator protocol while varying insertion depth.

## 2. Depth-error association with reader +10 mm

The depth files mentioned in the result report are not, by themselves, ground truth. Create a manifest with registered predicted and GT depth maps:

```csv
sample_id,pred_path,gt_path,mask_path,pred_key,gt_key,mask_key
scene0100_frame0000,/path/pred.npy,/path/gt.npy,,,, 
```

Only `sample_id,pred_path,gt_path` are required. Optional NPZ keys and a validity mask may be supplied. Both depth arrays must be same-resolution, same-camera-Z metric maps in the declared unit.

Run:

```bash
python diagnostics/analyze_rayrope_depth_error.py \
  --depth_manifest_csv "$RUN_ROOT/depth_manifest.csv" \
  --candidate_csv "$RUN_ROOT/results/candidates.csv" \
  --output_json "$RUN_ROOT/analysis/depth_error_reader_plus10.json" \
  --depth_unit m \
  --native_condition native \
  --perturbed_condition reader_plus10mm \
  --local_radius_px 5
```

The script reports signed depth bias, MAE, RMSE, P95 absolute error, valid coverage, optional local error around candidate pixel `u,v`, and frame-level correlations between actual depth error and +10 mm changes in score/Top-1 selection/quality.

This is an **association analysis**, not proof that +10 mm compensates the model's depth error. If GT maps are unavailable or not registered, do not run this analysis; a SHA256 list is insufficient.

## 3. Score-independent candidate sampling

The current 64 candidates per frame were sampled by score range. Their quality distribution and oracle are conditional on that score-stratified selection.

To obtain an independent candidate sample, first export the **complete candidate pool before score-based filtering/truncation**, then run:

```bash
python diagnostics/sample_rayrope_candidates_uniform.py \
  --input_csv "$RUN_ROOT/results/full_candidate_pool.csv" \
  --output_csv "$RUN_ROOT/analysis/candidates_uniform64.csv" \
  --n_per_sample 64 \
  --seed 20261106 \
  --input_is_full_pool
```

The required acknowledgement flag is deliberate. Do not pass the existing 64-row score-stratified table as a full pool. A manifest records the input SHA256, seed, pool size, and selected count per frame.

## 4. Interventions and what can be concluded

The candidate analyzer accepts arbitrary condition names and pairs rows only when the same `sample_id` and physical `candidate_id` occur in both conditions.

| Intervention | Analysis treatment |
|---|---|
| Reader -5/+5/+10 mm | Paired score/rank changes and Top-1 quality under the shared fixed candidate set |
| Placement ±5/±10 mm | Paired action-quality changes; model score should remain unchanged if the intervention is strictly post-decode |
| Coordinates-only / mask-only | Can be analyzed once model-side exports for these conditions exist; they are not generated by the current offline scripts |
| Full-system depth perturbation | Per-condition ranking/quality summary; candidate sets may differ, so do not claim paired fixed-candidate effects without shared physical action IDs |

**Important remaining model-side work:** P1.1 coordinates-only vs mask-only requires an inference-only hook in `moge_rayrope/attention.py` that independently selects perturbed RoPE coordinates and perturbed in-front validity. The offline scripts do not create those counterfactual predictions. Likewise, full-system depth perturbation requires changing predicted depth before proposal/seed selection, not just inside the reader.

## 5. Tests

```bash
python -m unittest tests.test_rayrope_p1_p2 -v
python -m unittest tests.test_rayrope_candidate_analysis -v
```

These are CPU-side synthetic tests. They do not replace a server run against the actual candidate CSV, depth manifest, GraspNet CAD models, or Dex-Net evaluator.

## 6. Interpretation limits

1. Candidate-level quality is the fraction of six official friction thresholds passed; it is not GraspNet AP or robot success rate.
2. Oracle quality is restricted to the exported candidate set.
3. Score-stratified sampling cannot estimate an unbiased proposal-pool quality distribution.
4. Scene bootstrap measures scene sampling uncertainty for the evaluated checkpoint; it does not measure training-seed variance.
5. Width bins are observational. A causal width claim requires matched physical geometry and exact re-evaluation at controlled widths.
6. Placement-only and full-system perturbations answer different questions and must not be combined.
7. The 18-frame / 9-scene result remains a pilot. Expand scene coverage before making dataset-level claims.
