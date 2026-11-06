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


---

## 7. Extended diagnostics: P0/P1/P2

### P0: Width decoder audit (existing CSV, CPU)

The interval `[0.10, +inf)` includes widths **equal to** the decoder clamp
at 0.10 m. Do not interpret that bin as 255 illegal widths. This script
reports below / within tolerance / strictly above the cap.

```bash
python diagnostics/audit_rayrope_width.py \
  --input_csv "$RUN_ROOT/results/candidates.csv" \
  --output_json "$RUN_ROOT/analysis/width_audit.json" \
  --condition native \
  --max_width_m 0.10 --tol_m 0.000001
```

Also inspect the checkpoint's actual `--grasp_max_width` runtime option, any
raw width prediction column and the CSV's width unit. If the executed width
really exceeds the declared cap by more than the tolerance, **stop and
reconcile the decode contract before using width-bin statistics**.

### P0: Old score-stratified64 vs uniform64 paired-scene statistics

This comparison is matched **at the frame/scene level**, not at the physical
candidate level. The two pools also differ in the underlying angle-depth
population, so any difference is descriptive, not a pure sampler effect.

```bash
python diagnostics/compare_rayrope_candidate_pools.py \
  --left_csv /path/to/score_stratified_candidates.csv \
  --right_csv /path/to/uniform_lattice_candidates.csv \
  --left_condition native --right_condition native \
  --output_json "$RUN_ROOT/analysis/paired_pool_comparison.json" \
  --n_boot 5000 --seed 20261106
```

For each of Top-1, Top-5, oracle, regret, Spearman, Kendall and high-score
failure, bootstrap paired scene clusters (stratified by Seen/Similar/Novel).

For 64 candidates, `ceil(64 * 0.05) = 4` means the "top 5% candidate"
measure is operationally **Top-4/64 = 6.25%**, not literally 5%. The
existing analyzer now records the realized fraction.

### P1: Reader coordinates-only / mask-only / both dose response

This uses an *inference-only*, non-persistent switch in
`moge_rayrope/attention.py`. The default forward path is unchanged.

Use a checkpoint **compatible with main's strict grasp_design metadata**
(historical experimental checkpoints are not automatically interchangeable).
The example sets `rayrope_group_chunk=2048` and disables chunk checkpointing,
matching one historical recipe. **Read the actual checkpoint metadata first**:
if its saved settings differ, pass those exact settings instead; changing a
chunk size is not a checkpoint-compatible CLI override in main.

```bash
python diagnostics/run_rayrope_p1_reader.py \
  --checkpoint_path "$CKPT" --dataset_root "$DATASET_ROOT" \
  --test_mode test_seen --camera realsense \
  --multi_modal --use_cdf --enable_point_rayrope --kview_mode A1 \
  --pose_depth_mode global_film --batch_size 1 --num_workers 2 \
  --rayrope_group_chunk 2048 --rayrope_checkpoint_chunks 0 \
  --diag_output_dir "$RUN_ROOT/p1_seen" \
  --diag_frame_manifest_csv "$RUN_ROOT/frames_seen.csv" \
  --diag_fixed_actions 64 \
  --diag_biases_mm=-10,-5,0,5,10,15,20 \
  --diag_modes coords_only,mask_only,both \
  --diag_seed 20261106
```

Frame manifest:

```csv
scene_id,anno_id
100,0
101,128
```

Repeat separately for test_similar and test_novel. P1 samples the 64 native
per-query winner actions by score strata **once** and freezes their physical
position, angle, insertion depth, width, and evaluator labels across all
conditions. The reader may predict a different width but **that new width is
not executed**. Changed scores are read from the same query-angle-depth
index, not by re-running per-query argmax.

```bash
python diagnostics/analyze_rayrope_p1_reader.py \
  --candidate_csv "$RUN_ROOT/p1_seen/p1_fixed_actions_labeled.csv" \
  --interventions_json "$RUN_ROOT/p1_seen/p1_interventions.json" \
  --output_json "$RUN_ROOT/p1_seen/p1_dose_analysis.json"
```

Required mask diagnostics: `token_count`, `valid_native`,
`valid_perturbed`, `mask_flip_count` and `queries_with_mask_flip`.
When mask-only score changes are zero, verify that mask flips were actually
nonzero before claiming mask insensitivity.

A positive bias must be selected on validation scenes and then evaluated
without tuning on held-out test scenes. This experiment is a
**geometry-coordinate sensitivity test**, not an automatically calibrated
depth-correction method.

### P2: Forward full-lattice query winner vs nonwinner, width and insertion sweeps

This is the preferred model-side P2 runner. It extracts the **full**
`Q x A x D` physical action lattice from the existing CDF endpoints and
verifies that per-query winner poses and scores match
`pred_decode_center_view_angle` within 1e-5 before evaluation.

For the same uniformly sampled query anchors:
- `query_winner`: native model argmax angle-depth.
- `query_nonwinner`: one random alternative from the same query (default).
- `lattice_uniform`: score-independent actions from the entire lattice.
- `width_sweep`: native action `(t,R,d)` fixed; executed width swept over
  4/6/8/10 cm, with native predicted width as a reference, while the
  existing score remains width-independent.
- `insertion_sweep_fixed_width`: native `(t,R,w)` fixed; all insertion
  bins evaluated with their corresponding CDF logits.
- Optional `insertion_sweep_predicted_width` changes depth **and** width;
  mark it as a compound intervention, not an insertion-only effect.

```bash
python diagnostics/run_rayrope_p2_lattice.py \
  --checkpoint_path "$CKPT" --dataset_root "$DATASET_ROOT" \
  --test_mode test_seen --camera realsense \
  --multi_modal --use_cdf --enable_point_rayrope --kview_mode A1 \
  --pose_depth_mode global_film --batch_size 1 --num_workers 2 \
  --rayrope_group_chunk 2048 --rayrope_checkpoint_chunks 0 \
  --diag_output_dir "$RUN_ROOT/p2_seen" \
  --diag_frame_manifest_csv "$RUN_ROOT/frames_seen.csv" \
  --diag_query_anchors 16 --diag_nonwinners_per_query 1 \
  --diag_width_grid 0.04,0.06,0.08,0.10 \
  --diag_insertion_width fixed \
  --diag_seed 20261106
```

```bash
python diagnostics/analyze_rayrope_p2_lattice.py \
  --input_csv "$RUN_ROOT/p2_seen/candidate_lattice_labeled.csv" \
  --output_json "$RUN_ROOT/p2_seen/p2_lattice_analysis.json" \
  --n_boot 5000 --seed 20261106
```

Report matched-query winner/nonwinner quality differences and scene-clustered
CI, Top-1/2/3/5/10/20 quality/collision/empty/force-closure curves, and
the within-grid oracle/regret for width/insertion. All physical variants are
separately evaluated by `ExactGraspNetActionEvaluator`. No CAD evaluation
labels enter model inference.

### P2 alternative: exact-action sweeps from existing CSV, without checkpoint

When historical `checkpoint_best.pt` cannot load into the main model, the
old saved physical candidate rows can still support **quality sensitivity**
experiments, provided the CSV contains camera-frame center, rotation matrix
(`r00..r22` or a JSON `rotation_matrix`), metric width, and insertion.

```bash
python diagnostics/eval_rayrope_controlled_sweeps_from_csv.py \
  --input_csv "$RUN_ROOT/results/candidates.csv" \
  --dataset_root "$DATASET_ROOT" --camera realsense \
  --native_condition native \
  --output_csv "$RUN_ROOT/analysis/exact_sweeps_from_csv.csv" \
  --num_anchors_per_frame 16 \
  --width_grid 0.04,0.06,0.08,0.10 \
  --insertion_grid 0.01,0.02,0.03,0.04 \
  --grasp_max_width_m 0.10 --seed 20261106
```

The width sweep is a controlled physical intervention with the **same
native model score** at every width. The CSV-only insertion sweep likewise
preserves the original score because the per-depth CDF logits are not
available. Its condition is explicitly named
`insertion_sweep_quality_only`: **do not compute depth-specific model
ranking/regret from those frozen scores**.

### Test and audit

```bash
python -m unittest tests.test_rayrope_p1_p2 -v
python -m unittest tests.test_rayrope_candidate_analysis -v
python -m unittest tests.test_rayrope_action_lattice -v
```

CPU synthetic tests validate CDF QAD indexing, deterministic sampling,
width clamping, fixed-parameter controls, mask/coords numerical invariants,
and paired bootstrap semantics. Server CUDA/checkpoint/CAD integration
must still pass a smoke test; nothing in this document claims a new model
checkpoint, training run, real GraspNet AP, or completed 18-frame evaluation.

Use distinct output directories for Seen/Similar/Novel and store:
`checkpoint SHA256`, `lattice_manifest.json`, `p1_interventions.json`,
`candidate_lattice_labeled.csv`, `p1_fixed_actions_labeled.csv`,
analysis JSON, evaluator settings, and per-scene frame list. Do not
silently modify the main model or reuse labels after changing a physical
grasp.


### Additional synthetic CPU tests

```bash
python -m unittest tests.test_rayrope_action_lattice -v
python -m unittest tests.test_rayrope_controlled_csv -v
```

A failed smoke or parity check must be investigated before any full evaluation.
Do not silently fall back to another checkpoint or change seed/query width to
make the runner pass. 
