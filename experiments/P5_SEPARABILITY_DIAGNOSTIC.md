# P5 safe-vs-repairable evidence separability diagnostic

## Question

P5-v1.1 shows a characteristic trade-off: under synthetic corrupted geometry it
can move repairable proposals toward the valid set, while on native predicted
geometry the same policy still damages too many already-safe proposals.  This
diagnostic asks whether the frozen P5 evidence already contains enough
information to decide **when** repair should be activated.

The diagnostic does not retrain P5 and does not call the GraspNet/Dex-Net
analytic evaluator during extraction.

## Query semantics

For the frozen Stage-1 `(view, angle, insertion-depth)` operation and its nearest
coherent positive annotation:

- **safe**: distance <= 5 mm;
- **repairable**: 5 mm < distance <= the P5 repair radius and the same operation
  has a positive cached CDF label;
- **unknown**: no coherent positive repair target.

Unknown queries are excluded from the binary probe rather than treated as
negatives.

Two binary tasks are exported:

1. `need_repair`: repairable vs safe;
2. `beneficial`: whether the *current frozen P5 repair vector* decreases the
   set violation.  This second task separates "should ideally be repaired" from
   "the current repair vector is actually useful here".

## Feature levels

All feature levels are frozen and exported from the same P5 forward pass.

- **F0**: low-dimensional scalar baseline: native score, depth, width,
  insertion depth, angle/spatial coordinates, keypoint visibility, and summary
  statistics of gripper-keypoint depth residuals.
- **F1**: F0 + pooled RGB/pre-geometry DPT evidence at gripper keypoints.
- **F2**: F0 + pooled geometry-conditioned evidence.
- **F3**: F0 + output of `P5GripperEvidenceEncoder`.
- **F4**: F0 + output of `P5CrossRayContext`.

Comparing F3 vs F2 tests whether gripper-centric encoding creates a useful
repair-needed representation.  Comparing F4 vs F3 tests whether cross-ray
context helps or dilutes that information.

## Causal gate controls

The exporter computes three mechanism baselines with the same P5 repair vector:

- `native`: never apply the repair vector;
- `p5`: always apply the repair vector (current P5 behavior);
- `oracle_gate`: apply the current P5 repair vector only to repairable queries.

The analysis also reports `oracle_benefit`, which applies the vector only where
it is known post hoc to reduce set violation.  This is a stricter upper bound on
what activation alone can recover without changing the vector itself.

If `oracle_gate` is weak, a binary repair-needed gate is not the shortest next
step because the repair vector remains a bottleneck.  If the oracle is strong,
probe separability becomes actionable.

## Probe protocol

The launcher extracts four conditions:

1. train / native;
2. train / deterministic corrupt;
3. test_seen / native;
4. test_seen / deterministic corrupt.

Queries are sampled uniformly from coherent known queries within each frame; no
class balancing is done during export.  Linear logistic probes are trained on
three regimes (`native`, `corrupt`, `mixed`) and evaluated cross-scene on both
Seen conditions.

Reported metrics include:

- AUROC;
- AUPRC (repairable/beneficial is positive);
- precision among the top 5/10/20% intervention scores;
- recall at 1/2/5% negative-class false-positive rate;
- simulated set violation, within-safe ratio, safe no-harm ratio, and
  repairable capture after learned gating.

## Run

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/path/to/p5_v11/checkpoint_epoch_XXX.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/p5_separability \
TRAIN_SAMPLE_INTERVAL=0.1 \
EVAL_SAMPLE_INTERVAL=0.1 \
QUERY_SAMPLE_PER_FRAME=128 \
bash run_p5_separability.sh
```

For a fast extraction smoke:

```bash
DIAG_MAX_BATCHES=2 bash run_p5_separability.sh
```

Smoke mode intentionally skips probe fitting.

Main outputs:

```text
analysis/RESULTS.md
analysis/probe_metrics.csv
analysis/gate_metrics.csv
analysis/report.json
analysis/linear_probes.json
```

## Interpretation

| Oracle activation | Frozen-feature separability | Interpretation |
|---|---|---|
| strong | strong | Current evidence is sufficient; add a selective repair gate. |
| strong | weak | Repair vectors can help, but current evidence does not reveal when to use them. |
| weak | strong | The model knows which proposals are wrong, but its repair vector is inadequate. |
| weak | weak | Both representation and repair formulation need revision. |

Additional domain interpretation:

- high corrupt AUROC but low native AUROC: synthetic-error cues do not align
  with native predicted-depth errors;
- F3 >> F2: gripper-centric encoding adds repair-needed information;
- F4 >> F3: cross-ray consistency is useful;
- F4 < F3: cross-ray aggregation likely dilutes the local signal.
