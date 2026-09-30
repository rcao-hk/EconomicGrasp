# P1-1 / P1-2 / P1-3: frozen-source online experiments

Branch: `exp/dav2-metric-grasp-field`.

These experiments are designed to run without a new feature/action/label cache.
They reuse the complete matched20 Metric-Grasp-Field checkpoint as a **frozen
strong source**. Candidate centres, views, widths, metric depth, relative
decoder, ray head, and source Base-CVA scorer remain deterministic and frozen
unless a documented intervention explicitly replaces an input to the frozen ray
head.

The experiments are independent of P0 results and can run concurrently with
P0 on disjoint GPUs. With the current six-GPU layout, a practical schedule is:

```text
GPUs 0,1,2 -> P0
GPUs 3,5,6 -> one P1 family at a time
```

P1-2 uses the exact CAD/DexNet evaluator and is CPU-heavy. Do not run P1-2
concurrently with P0-2 on the same host unless CPU/RAM contention has been
measured. P1-1 or P1-3 training is the preferred concurrent job while P0 is
running.

To queue all P1 families on GPUs 3,5,6 while P0 occupies another GPU pool:

```bash
GPUS=3,5,6 bash scripts/run_mgf_p1_queue.sh
```

The queue runs P1-1 -> P1-3 -> P1-2 sequentially, so the P1 jobs do not
oversubscribe the same GPUs. Override `STAGES` to run a subset.

---

## P1-1: what does the ray field actually contribute?

P1-1 has two subquestions.

### A. Point estimate versus distribution shape

The following three controls use the **same learned ray profile and therefore
the same profile mean**. Only the downstream evidence operator changes:

| Variant | Evidence used at each support point |
|---|---|
| `profile_hard` | hard first-surface relation from the profile mean |
| `profile_fixed` | analytic Gaussian around the same profile mean |
| `profile_learned` | learned profile CDF shape |

All variants use the same frozen Base, candidate bank, task adapter/readout
initialization, BCE, ranking loss, training exposure, and optimizer. Their
residual output starts at zero, so every control initially reconstructs Base.

This comparison answers whether the learned **shape** of the ray distribution
adds value beyond its point estimate. It does not claim calibrated uncertainty.

### B. Where does the frozen DAV2 relative prior matter?

`profile_learned` is also the full relative-prior reference. Three additional
controls recompute the already-trained, frozen ray head online:

| Variant | relative scalar | relative decoder feature |
|---|---:|---:|
| `relative_metric_only` | no | no |
| `relative_scalar` | yes | no |
| `relative_feature` | no | yes |
| `profile_learned` | yes | yes |

When relative features are removed, the same ablation is applied to the
trainable TaskFeatureAdapter. Metric depth/features remain unchanged.

Important limitation: this is an **input intervention on the trained frozen ray
head**. It isolates dependence of the current representation on relative cues;
it is not equivalent to retraining four ray heads from scratch.

Run:

```bash
GPUS=3,5,6 \
bash scripts/run_mgf_p1_1_online.sh
```

Defaults: five additional scorer-only epochs, batch/GPU 3, AdamW 3e-4, WD
1e-3, cosine LR, BCE + 0.1 query-ranking, 20% train exposure, 10% Seen
monitoring, latest checkpoint only. Collision off/on dumps are produced from
the same inference forward; **collision-off is the primary mechanism result**.

Smoke:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/mgf_p1_1_smoke \
PHASES=train EPOCHS=1 MAX_TRAIN_FRAMES=12 MAX_VAL_FRAMES=6 MAX_STEPS=2 \
GPUS=3,5,6 bash scripts/run_mgf_p1_1_online.sh
```

---

## P1-2: are canonical transferred labels faithful to the queried action?

The cleaned CDF trainer transfers canonical CDF/width annotations from the
nearest grasp point when the predicted centre is within 5 mm. P1-2 audits that
assumption directly.

For each sampled frame:

1. run the frozen source with existing online canonical matching;
2. select a mixture of high-Base and random valid queries;
3. per query, select high-Base and random valid angle-depth candidates;
4. evaluate **those actual predicted 17D grasps** with
   `ExactGraspNetActionEvaluator`;
5. compare transferred threshold labels with exact collision + force-closure
   labels.

No exact label is fed back into training and no reusable cache is written.
Outputs are result JSON/CSV only.

The audit reports:

- threshold-level label agreement;
- transferred versus exact utility MAE;
- any-success false-positive/false-negative rates;
- strata by centre-to-nearest-canonical-point distance;
- strata by predicted-width versus canonical-width mismatch;
- exact `clear / pure_collision / empty` strata.

The stock exact evaluator does not expose a continuous collision margin, so
P1-2 reports exact collision states rather than inventing a "collision
proximity" quantity.

Run:

```bash
GPUS=3,5,6 \
bash scripts/run_mgf_p1_2_online.sh
```

Default workload is one stride-10 frame per test scene, eight queries/frame,
and up to eight candidates/query (four Base-top + four seeded random). Start
smaller if needed:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/mgf_p1_2_smoke \
SPLITS=test_seen FRAMES_PER_SCENE=1 QUERIES=2 \
TOP_CANDIDATES=2 RANDOM_CANDIDATES=1 GPUS=3 \
bash scripts/run_mgf_p1_2_online.sh
```

---

## P1-3: what form should the physical correction take?

All P1-3 variants use the same frozen strong Base, full learned ray evidence,
same support points, and same BCE + ranking objective.

| Variant | Correction |
|---|---|
| `residual6` | current six-parameter monotone CDF residual |
| `base_conditioned` | residual reader explicitly receives six Base logits + Base utility |
| `evidence_gated` | residual is multiplied by deterministic evidence sufficiency |
| `shared_shift` | one action-dependent scalar shifts all six Base logits equally |

### Base-conditioned residual

The physical reader sees what it is correcting. Candidate Base logits and their
mean success utility are projected into the residual hidden state. Base remains
detached and frozen.

### Evidence-gated fallback

The gate is deliberately **not learned**. For each physical action it is:

```text
mean over support points [
    valid_projection * (1 - normalized ray-profile entropy)
]
```

Thus invalid or diffuse evidence moves the correction toward zero and falls
back to Base. A learned gate could trivially saturate to one and would not test
the intended fallback hypothesis.

### Shared shift

A single scalar is added to every threshold logit. It preserves all Base CDF
increments exactly and tests whether the more expressive six-dimensional
monotone residual is actually necessary.

Run:

```bash
GPUS=3,5,6 \
bash scripts/run_mgf_p1_3_online.sh
```

The `residual6` run intentionally duplicates the P0 full frozen-source
residual as an internal P1-3 reference. If P0 full is already complete, it can
be used as a cross-check; do not merge checkpoints from different protocols.

---

## Interpretation order

1. **P1-1:** `profile_learned > profile_fixed/hard` supports value from profile
   shape beyond the mean. Relative variants then indicate which frozen DAV2
   cue the current ray head depends on.
2. **P1-2:** large disagreement concentrated near 5-mm matching or large width
   mismatch would identify supervision transfer as a material limitation.
3. **P1-3:** compare correction forms only after confirming identical source
   hashes and budgets. A simpler shared shift that matches residual6 should
   replace the more complex parameterization.
4. Collision-off is primary for representation/scoring claims. Collision-on is
   a system-level secondary result because it uses sensor depth outside the RGB
   network.
5. These experiments use existing official test splits for diagnosis. Do not
   repeatedly select architectures/hyperparameters from Similar/Novel and then
   report the same numbers as untouched final-test evidence.
