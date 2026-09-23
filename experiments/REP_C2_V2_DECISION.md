# Rep-C2-v2 Decision Diagnostic

This diagnostic operates on already-trained Rep-C2-v2 checkpoints. It does not
retrain the verifier architecture and does not call CAD/DexNet.

Its purpose is to answer two questions exposed by the full-query C2-v2 result:

1. Did the auxiliary predicted utility difference contain a better decision
   signal than the original global `P(beneficial)` threshold?
2. Does frame-level correction context help identify whether the current frame
   is in a "native mostly trustworthy" or "anchor likely wrong" regime?

## Decision rules

For every C2-v2 variant (`score`, `rgb`, `rgb_only`) the diagnostic compares:

| Rule | Decision score |
|---|---|
| `p_beneficial` | original C2-v2 (P(beneficial)) |
| `pred_delta` | auxiliary predicted (widehat{\Delta U}) |
| `class_margin` | (P(beneficial)-P(harmful)) |
| `fused_delta` | confidence-weighted signed predicted delta |
| `query_ridge` | lightweight ridge calibration from query-local verifier/A1 signals to true (Delta U) |
| `context_ridge` | query-ridge features plus frame-level A1 correction context |

The frame context is fully inference-available and contains no injected-error
label:

```text
proposal rate
mean signed proposed offset
mean absolute proposed offset
positive / negative offset fractions
mean / std / mean-absolute A1 proposal margin
```

The key comparison for the regime hypothesis is:

[
\text{context\_ridge} - \text{query\_ridge}.
]

## Seen calibration split

Because a learned calibration layer is introduced, Seen scenes are
deterministically split in half:

```text
Seen calibration-train:
    fit query_ridge and context_ridge to exact delta utility

Seen calibration-val:
    select all decision thresholds using full-query verifier increment
```

The split is scene-level, deterministic from `SPLIT_SEED`. Similar and Novel
are never used to fit ridge weights or choose thresholds.

The underlying C2-v2 checkpoint was itself selected on Seen, so Seen numbers are
still validation diagnostics rather than an independent test. Similar/Novel
remain the held-out evidence.

## Threshold policies

Two policies are reported.

### Global

One threshold per verifier variant / decision rule, chosen on Seen
calibration-val by maximizing:

[
\Delta_V = G_V-G_{A1}.
]

This is the deployable-style result. No error-regime identity is required.

### Case-specific privileged

A separate threshold is chosen for nominal / -20 / +20 on Seen calibration-val
and reused for the corresponding synthetic case on Similar/Novel.

This is **not deployable** because it assumes knowledge of the synthetic error
case. It is an upper-bound diagnostic for the "regime-dependent threshold"
hypothesis.

A large gap:

[
\text{case-specific} \gg \text{global}
]

means calibration/regime identification is an important part of the remaining
failure. If both are poor, representation/discrimination is the stronger
limitation.

## Metrics

Every rule is evaluated with the full Stage-1 query count in the denominator:

```text
verified_gain
a1_fixed0_gain
oracle_accept_gain
verifier_increment
oracle_gap
oracle_gap_recovery
beneficial_retention
harmful_rejection
accept_precision
benefit_harm_auc
pearson_with_delta
```

The main quantities are:

[
\text{verifier increment}=G_V-G_{A1},
]

[
\text{oracle-gap recovery}
=
\frac{G_V-G_{A1}}{G_O-G_{A1}}.
]

## Run

Use the trained objective-v2 verifier directory:

```bash
cd /home/robotarm/EconomicGrasp
git switch exp/rep-a-depth-robust-reader
git pull --ff-only

python -m pytest -q tests/test_rep_c2v2.py
```

Then:

```bash
SOURCE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_formal_joint_ap \
C2V2_DIR=/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_fullpath_verifier_objective_v2/train \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_decision_diagnostic \
GPUS=0,1,2 \
SPLITS=test_seen,test_similar,test_novel \
CASES=nominal,bias:-20,bias:20 \
QUERY_CHUNK=128 \
SEEN_TRAIN_FRACTION=0.5 \
PHASES=fit,eval \
bash scripts/run_rep_c2v2_decision.sh
```

`fit` uses GPU0. After calibration is complete, `eval` distributes splits
across the listed GPUs, so with three GPUs Seen/Similar/Novel run in parallel.

No new Stage-1 inference, training-cache mining, or CAD/DexNet evaluation is
performed.

## Outputs

```text
WORK_ROOT/
  calibration/
    calibration.json
    seen_calibration.csv
    threshold_sweep_<variant>_<rule>.json
  eval/
    test_seen/
      comparison.csv
      per_frame.csv
      summary.json
    test_similar/
      ...
    test_novel/
      ...
```

Interpretation priority:

1. compare `pred_delta` vs `p_beneficial`;
2. compare `query_ridge` vs `pred_delta`;
3. compare `context_ridge` vs `query_ridge`;
4. compare global vs case-specific privileged thresholds;
5. only if a rule has positive held-out verifier increment consider generating
   official-AP dumps in a follow-up experiment.
