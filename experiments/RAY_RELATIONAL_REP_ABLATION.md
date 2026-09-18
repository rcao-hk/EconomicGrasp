# Fully Trainable Cross-Center Representation Ablation

Branch: `exp/ray-bestofk-exact-action-diagnostic`

## Goal

The previous relational audit showed that both opportunity recognition (Should I
move?) and conditional center ranking (Where should I move?) lose accuracy from
train to validation. This experiment tests whether that failure comes from the
evidence exposed to the relational selector rather than from the gate alone.

**Every variant trains the complete selector from scratch**:

- input projection;
- Transformer encoder;
- move gate;
- conditional center selector;
- auxiliary delta head.

The architecture, exact-action objectives, K=7 candidates, train/validation
split, checkpoint selection, and threshold selection are fixed. Only the
relational token representation changes.

## Variants

### G0_current_full

Exact reproduction of the current relational token:

[
[F^{sel}_k,
 F^{sel}_k-F^{sel}_0,
 F^{mean}_k,
 F^{mean}_k-F^{mean}_0,
 s_k,
 s_k-s_0,
 Delta z_k,
 mathbb{1}(k=0)].
]

This is the controlled baseline.

### G1_no_abs_raw

Remove absolute CDF confidence (s_k), but retain its cross-center residual:

[
[F^{sel}_k,
 Delta F^{sel}_k,
 F^{mean}_k,
 Delta F^{mean}_k,
 Delta s_k,
 Delta z_k,
 mathbb{1}(k=0)].
]

Question: is the poor generalization caused by reliance on absolute grasp
confidence?

### G2_residual_only

Remove all absolute local features and absolute CDF confidence:

[
[Delta F^{sel}_k,
 Delta F^{mean}_k,
 Delta s_k,
 Delta z_k,
 mathbb{1}(k=0)].
]

Question: are cross-center changes more invariant than absolute evidence?

### G3_residual_profile

Use G2 plus compact ray-global profile statistics. The profile contains:

- mean/std/min/max of (Delta s_k);
- mean/std/max of (|Delta F^{sel}_k|_2);
- mean/std/max of (|Delta F^{mean}_k|_2);
- positive-vs-negative ray-side asymmetry for the above three signals.

The 13-D profile is broadcast to every K token before the shared Transformer.

Question: does explicitly describing how evidence changes along the complete ray
improve opportunity/center reasoning?

### G4_mean_profile

Use G2 plus channel-wise K-profile statistics of the angle-mean residual:

[
[operatorname{Mean}_k(Delta F^{mean}),
 operatorname{Std}_k(Delta F^{mean})].
]

These (2C) statistics are broadcast to every token.

Question: does the more conservative angle-mean evidence provide a more
transferable ray-level context?

## Fixed training objective

For every mode:

[
L=L_{gate}+L_{selector}+0.5L_{delta}.
]

The move target remains strict:

[
y_{move}=1
iff
max_{k
e0}U_k>U_0.
]

The selector is trained on move-positive queries using exact-action utility, and
the delta head regresses (U_k-U_0).

Scenes 0--99 are used for training. Scenes 100--129 are validation only.
Similar/Novel are held out from model/threshold selection.

## Unit tests

```bash
pytest -q \
  tests/test_ray_relational_rep_ablation.py \
  tests/test_ray_relational_selective.py
```

The tests include an exact G0 equality check against the original relational
token composer and invariance checks for residual-only variants.

## Train all variants

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector \
GPUS=0,1,2,3,4 \
GRAD_ACCUM_STEPS=1 \
bash scripts/run_ray_relational_rep_ablation_train.sh
```

Outputs:

```text
WORK_ROOT/relational_rep_ablation/
  G0_current_full/
  G1_no_abs_raw/
  G2_residual_only/
  G3_residual_profile/
  G4_mean_profile/
```

Each directory contains `checkpoint_best.tar`, `checkpoint_latest.tar`,
`best.json`, `metrics.jsonl`, and the threshold sweep.

`GRAD_ACCUM_STEPS` controls the effective number of cache frames per optimizer
update without increasing the per-forward ray batch.

## Held-out Similar / Novel

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector \
GPUS=0,1,2,3 \
bash scripts/run_ray_relational_rep_ablation_heldout.sh
```

The launcher uses each variant's validation-selected best checkpoint and move
threshold. The held-out protocol is unchanged:

- K=7 offsets: -40,-20,-10,0,+10,+20,+40 mm;
- sample interval 0.1;
- 128 topk_uniform queries/frame;
- same frozen Stage-1 checkpoint;
- same exact-action evaluator.

A compact `comparison.csv` is written under
`WORK_ROOT/relational_rep_ablation_heldout/`.

## Interpretation

The primary comparison is not only validation utility. A useful representation
should improve or preserve held-out Similar/Novel utility while reducing the
train-to-validation and Seen-to-Novel degradation in:

- move/opportunity discrimination;
- rescue versus harm;
- center selection;
- oracle-headroom recovery.

If residual/profile representations improve training or validation but not
Similar/Novel, then removing absolute confidence is insufficient: the frozen
center-conditioned local features themselves remain the likely domain-specific
bottleneck.
