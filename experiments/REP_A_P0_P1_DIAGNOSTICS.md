# Rep-A-P0 / Rep-A-P1 Diagnostics

These diagnostics reuse trained Rep-A checkpoints. They do **not** retrain any
network, regenerate candidates, or call CAD/DexNet.

## P0: selection-policy control

Purpose: separate score/representation robustness from conservative native
fallback. For each A0-A3 checkpoint, evaluate:

- `val_selected`: its original Seen-validation-selected margin;
- `fixed_0`: common margin 0;
- `fixed_0.1`: common margin 0.1;
- `matched_move_rate`: a margin chosen on nominal Seen validation only so the
  variant matches the nominal Seen-validation move rate of A0 under A0 own
  validation-selected margin.

The matched policy never uses Similar/Novel outcomes or corruption severity.
All policies are applied to the same logits from the same checkpoint, so
probability drift, representation drift, BCE/Brier, and within-ray pairwise
accuracy are margin-independent controls.

Run:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness \
GPUS=0,1 bash scripts/run_rep_a_p0_selection.sh
```

Primary outputs:

```text
WORK_ROOT/diagnostics/rep_a_p0_selection/test_similar/summary.json
WORK_ROOT/diagnostics/rep_a_p0_selection/test_similar/per_frame.csv
WORK_ROOT/diagnostics/rep_a_p0_selection/test_novel/...
```

Interpretation: if A3 remains robust at `fixed_0` and matched move rate, the
robustness is not explained only by its large validation-selected margin. If
the advantage disappears after policy matching while pairwise/distribution
metrics remain unchanged, conservatism explains much of the selection gain.

## P1: trained A3 branch intervention

Purpose: test whether the depth-independent RGB branch is actually responsible
for the extra robustness seen in A3 under large metric-depth errors.

Using the exact same trained A3 weights:

```text
full     = depth-conditioned representation + 0.1*RGB + action embedding
no_rgb   = depth-conditioned representation + action embedding
rgb_only = 0.1*RGB + action embedding
```

The learned 0.1 RGB fusion scale is retained. No branch is renormalized or
recalibrated. All A3 interventions share the original A3 validation-selected
margin. This makes the branch removal a strict checkpoint intervention rather
than a new tuned model.

A separately trained A1 model is also reported as `a1_reference` using A1 own
validation-selected margin. `A3-no_rgb` is **not** mathematically the same as
A1: A3 shared weights were trained jointly with an RGB branch. Similarity to
A1 is therefore evidence about pathway dependence, not model identity.

Default stress cases are:

```text
nominal, bias +/-20 mm, bias +/-40 mm, smooth 20 mm RMS
```

Run:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness \
GPUS=0,1 bash scripts/run_rep_a_p1_branches.sh
```

Besides grasp metrics, P1 reports:

- depth/RGB/action component norms;
- drift of each component from its own nominal value;
- probability and representation drift for each intervention;
- distance of `no_rgb`/`rgb_only` from `full` under the same corrupted input;
- selection turnover and within-ray pairwise discrimination.

The decisive pattern would be: A3-full is robust, A3-no_rgb moves toward A1
or loses robustness, while A3-rgb_only remains stable but may lose metric
discrimination. That would support the interpretation that depth-error
training teaches the model to use the independent image evidence as a fallback.
If `no_rgb` stays almost as robust as full, the extra robustness is primarily
encoded in the jointly trained depth-conditioned path rather than the bypass.

## Smoke protocol

Run both diagnostics on two frames before the formal pass:

```bash
MAX_FRAMES=2 MAX_VAL_FRAMES=2 GPUS=0 \
  bash scripts/run_rep_a_p0_selection.sh

MAX_FRAMES=2 GPUS=0 \
  bash scripts/run_rep_a_p1_branches.sh
```

Then run `python -m pytest -q tests/test_rep_a.py`; the added integration test
checks that `intervention=full` exactly reproduces the pre-intervention A3
forward and that the component sums match all three P1 intervention formulas.
