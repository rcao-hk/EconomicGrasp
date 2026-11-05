# RayRoPE P1/P2 diagnostics — implementation contract

Branch: `exp/rayrope-p1-p2-diagnostics`  
Base: `main@7a956d37aaa427a32e598f21fdc6b910ed3c2704`

## What is implemented in this branch

- `diagnostics/rayrope_p1_p2.py`: deterministic metric-depth interventions (bias, affine, local plane), candidate-table validation, per-sample ranking metrics, Top-1/Top-5 quality, oracle regret, high-score failure rate, and paired intervention score/rank stability.
- `tests/test_rayrope_p1_p2.py`: unit tests for invalid-depth preservation, metric interventions, ranking, and regret.
- The code is offline-only and does not alter training, checkpoints, model forward, or normal inference.

This is the analysis layer and data contract. It does **not** claim that model-side intervention capture or DexNet/GraspNet evaluation has already run.

## P1 — geometry-path decomposition

The model-side runner must export the same physical candidate IDs across reader-geometry interventions. The candidate table must contain one row per `(sample_id, candidate_id, intervention)` with the exact evaluator quality for that row.

Required intervention labels:

| Label | Physical grasp / proposal | Reader geometry | Interpretation |
|---|---|---|---|
| `native` | fixed | native predicted depth | reference |
| `evidence_depth_bias_p5mm` | fixed | predicted depth +5 mm | evidence-only |
| `evidence_depth_bias_m5mm` | fixed | predicted depth -5 mm | evidence-only |
| `evidence_depth_bias_p10mm` | fixed | predicted depth +10 mm | evidence-only |
| `placement_p5mm` | translation actually shifted +5 mm along the camera-ray direction | native reader geometry | placement-only |
| `full_depth_bias_p5mm` | proposal/seed/reader/decode follow the perturbed depth | predicted depth +5 mm | full-system |

Important controls:
1. Evidence-only must keep the physical grasp center, rotation, insertion depth, width, RGB feature tensor, candidate ID, and evaluator geometry fixed. Only the geometry passed to the RayRoPE evidence reader changes.
2. Placement-only must change the executed physical translation and evaluate that translated grasp. Do not change the reader depth in this condition.
3. Full-system changes all model paths affected by the predicted depth. It is a deployment stress test, not a controlled reader-only effect.
4. Reuse identical scenes, frames, RGB, proposals, candidate IDs, and collision/evaluator settings. Never mix collision-on and collision-off records.
5. Export both the exact metric depth tensor used by the model and the intervention metadata. Do not use observed/GT depth as RGB-model input.

## P2 — fixed-candidate action identifiability

For each RGB observation, create a pre-registered action set. Each candidate must be a fully specified physical action with at least:
- camera-frame center `center_xyz_m`;
- rotation matrix or approach/closing/vertical axes;
- insertion depth in metres;
- executed width in metres;
- model score;
- exact-action evaluator quality.

The `candidate_id` must identify the physical action, not merely a seed or an angle/depth tensor index. The quality label must be computed for that exact center, rotation, insertion depth, and width. Do not reuse the source annotation's label after moving the center or changing the action.

The P2 analysis reports:
- Spearman and Kendall-τ-b rank agreement between model score and evaluator quality;
- Top-1 and Top-5 quality;
- Top-1 regret relative to the best candidate in the *same fixed candidate set*;
- high-score failure rate (default: top 5% by model score);
- score/rank stability across paired interventions.

The within-set oracle is not a global action-space oracle. State the candidate-generation protocol and candidate budget.

## CSV contract

Required columns:

```csv
sample_id,candidate_id,intervention,score,quality
scene0100_frame0000,g000_native,native,0.82,0.7
scene0100_frame0000,g000_native,evidence_depth_bias_p10mm,0.77,0.7
```

Optional action columns (recommended for auditability):

```text
action_id,center_x,center_y,center_z,approach_x,approach_y,approach_z,
closing_x,closing_y,closing_z,vertical_x,vertical_y,vertical_z,
insertion_depth_m,width_m,depth_source,checkpoint_sha256,collision_mode
```

Quality must use a single documented convention across the table (higher is better). For official GraspNet AP, do not pass AP itself as per-candidate quality. Use the exact-action evaluator's per-candidate force-closure/friction outcome or a documented scalar derived from it. Keep the six friction thresholds available for stratified reporting.

## Run the offline analysis

```bash
python diagnostics/rayrope_p1_p2.py \
  --input_csv /path/to/p1_p2_candidates.csv \
  --output_json /path/to/p1_p2_metrics.json \
  --success_threshold 0.0 \
  --high_score_quantile 0.95

python -m unittest tests.test_rayrope_p1_p2 -v
```

## Main-code integration points (next implementation step)

- P1 evidence-only hook: `moge_rayrope/attention.py:GraspRayRoPEGrouping.forward/_chunk`. The reader depth map can be overridden while keeping `seed_xyz`, `top_view_rot`, RGB feature map, and candidate identity fixed. The override must be inference-only and must not mutate the batch's native depth tensor.
- P1 full-system hook: `models/economicgrasp_bip3d.py:economicgrasp_dpt.forward` before seed selection/grouping, so depth changes propagate through proposal, grouping, and decoding.
- P1 placement-only: modify decoded grasp translations after `pred_decode_center_view_angle`, along the camera ray, and run the exact same evaluator/collision protocol. Do not feed this modified grasp back into the reader.
- P2 candidate extraction: `models/economicgrasp_bip3d.py:pred_decode_center_view_angle` and the CDF endpoints. The current RayRoPE grouping occurs before insertion depth and width are decoded; the GVAR reader is insertion-conditioned but uses a fixed support-envelope width. Therefore the runner must record actual decoded width and must not claim exact-width-conditioned reading.
- Exact quality: reuse the repository's GraspNet/Dex-Net exact-action evaluator, but verify its grasp row convention and evaluator settings before use.

No model-side capture is included in this first commit because the existing inference entry point currently exports only decoded grasp dumps; it does not expose a stable, lossless candidate/query-score contract for the counterfactuals above. Implementing the hooks without first locking that contract risks silently comparing different candidates or reusing labels for changed actions. The next code change should add an explicit opt-in diagnostic mode and keep ordinary inference byte-for-byte unchanged.
