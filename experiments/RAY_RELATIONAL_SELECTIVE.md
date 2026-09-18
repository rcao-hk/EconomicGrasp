# Cross-Center Relational Selective Correction

Branch: `exp/ray-bestofk-exact-action-diagnostic`

## Motivation

The completed independent-selector diagnosis established:

1. K=7 ray expansion contains large exact-action oracle headroom.
2. Raw CDF cross-center ranking is unreliable.
3. Independent per-center MLP selectors recover only a small fraction of oracle
   headroom, especially on Novel.
4. A native fallback / confidence mechanism reduces harm substantially.
5. Exact-utility ties are common, so forcing a K-way center decision on every
   ray creates unnecessary intervention.

Therefore the next question is not whether to add more MLP capacity. The
decision itself is re-factorized:

[
P(\mathrm{move}\mid\mathcal F_{1:K})
]

followed by

[
P(k\mid\mathrm{move},\mathcal F_{1:K}).
]

All K hypotheses on the same camera ray are encoded jointly.

## Representation

For each center k, the relational token is

[
[F^{sel}_k,
 F^{sel}_k-F^{sel}_0,
 F^{mean}_k,
 F^{mean}_k-F^{mean}_0,
 s_k,
 s_k-s_0,
 \Delta z_k,
 \mathbb{1}(k=0)].
]

The frozen K-center Stage-1 reread is unchanged. A small Transformer encoder
contextualizes the ordered K-set.

Default encoder:
- d_model = 128
- heads = 4
- layers = 2
- FFN = 256
- dropout = 0.1

## Heads

The contextual K tokens feed three heads:

- **move gate**: native contextual token + valid-set pooled context -> one logit;
- **conditional center selector**: one ranking logit per center;
- **delta head**: auxiliary contextual prediction of exact utility difference.

The gate target is deliberately strict:

[
y_{move}=1
\iff
\max_{k\ne0} U_k > U_0.
]

Exact-utility ties do **not** teach the model to move.

The conditional selector loss is applied only to move-positive queries and
soft-ranks non-native centers by their exact decoded-grasp utility. The delta
head regresses (U_k-U_0) on valid alternatives.

Default objective:

[
L=L_{gate}+L_{selector}+0.5L_{delta}.
]

## Deployment

At inference:
1. jointly encode K centers;
2. choose the highest-ranked valid non-native center;
3. switch only if gate probability exceeds the validation-selected move
   threshold; otherwise keep native.

The threshold is selected on scenes 100--129 using validation exact utility,
with lower harm and then fewer changes as tie-breaks. Similar/Novel are not used
for threshold tuning.

## Training

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector \
TRAIN_GPU=0 \
bash scripts/run_ray_relational_selective_train.sh
```

Default caches:
- `WORK_ROOT/cache`: training scenes; only scene id < 100 is consumed;
- `WORK_ROOT/cache_val`: validation scenes; only scene id >= 100 is consumed.

## Held-out Similar / Novel

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector \
GPUS=0,3 \
bash scripts/run_ray_relational_selective_heldout.sh
```

This uses the same K=7, 128-query topk_uniform, exact-action protocol as the
previous independent-selector held-out diagnosis.

## Interpretation gate

The experiment is successful only if joint K-context improves the held-out
trade-off, not merely training fit.

Primary comparisons:
- relational vs independent `full` selector on Similar/Novel;
- learned-minus-native utility and Success@0.8;
- rescue vs harm;
- intervention rate;
- oracle-headroom recovery.

A validation gain without Similar/Novel gain is evidence that relational
capacity still fits scene-specific structure rather than solving physical-center
identifiability.
