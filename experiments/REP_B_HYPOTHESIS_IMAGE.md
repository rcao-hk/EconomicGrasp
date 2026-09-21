# Rep-B: Hypothesis-conditioned Image Grasp Representation

Branch: `exp/rep-a-depth-robust-reader`.

## Goal

Rep-B tests whether fixed physical depth hypotheses can be discriminated from
image evidence without letting predicted depth determine where RGB evidence is
stored/read. It reuses the existing Rep-A cache; no new mining is required.
Formal sampling therefore remains the Rep-P0 protocol: sample interval 0.1.

## Representation

For every K-ray action hypothesis, a shared gripper-projected image reader
samples 40 points from closing/finger/palm/approach regions in the pre-enhancer
image feature map. The K tokens of the same ray then pass through a relational
Transformer. Candidate translation is a physical action property and may affect
the projection. Predicted depth, when enabled, enters only as a soft prior:

```text
delta = (candidate_camera_z - predicted_depth_at_seed) / sigma
prior = MLP([delta, exp(-0.5*delta^2), depth_valid])
```

Default sigma is 30 mm. Predicted depth never gates image projection, feature
membership, or candidate validity.

## Variants

| Variant | candidate-conditioned RGB | K-way relation | soft depth prior | depth-error augmentation |
|---|---:|---:|---:|---:|
| B0 | yes | yes | no | no |
| B1 | yes | yes | yes | no |
| B2 | yes | yes | yes | yes |

B0 is intentionally depth independent. B1 tests whether a weak late metric
anchor restores clean discrimination. B2 uses the same structured depth-error
training as Rep-A (25% nominal; otherwise bias +/-20 mm, scale +/-3%, or
smooth error <=10 mm RMS). Only the soft prior sees perturbed depth.

The formal first-round loss is six-threshold CDF BCE. An optional within-ray
pairwise utility loss exists but is disabled by default (`PAIRWISE_WEIGHT=0`)
to avoid changing representation and objective simultaneously.

## Validation / test

Checkpoint and native-fallback margin are selected only on nominal test_seen.
Similar/Novel never tune the model. Default test corruptions are identical to
Rep-A: signed 5/10/20/40-mm bias, signed 2/5% scale, smooth 5/10/20-mm RMS,
and edge shift 2 px.

Metrics include selected utility, Success@0.8, rescue/harm, oracle headroom,
within-ray pair accuracy, probability/representation drift, selection turnover,
and image/prior/pre-relation component drift. B0 also reports an explicit
depth-invariance check.

## Smoke

```bash
git switch exp/rep-a-depth-robust-reader
git pull --ff-only
python -m pytest -q tests/test_rep_a.py tests/test_rep_b.py

WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_b_smoke \
REP_A_WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness \
GPUS=0 MAX_TRAIN_FRAMES=8 MAX_VAL_FRAMES=4 EPOCHS=2 PHASES=train \
bash scripts/run_rep_b.sh

WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_b_smoke \
REP_A_WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness \
GPUS=0 MAX_TEST_FRAMES=4 TEST_CASES=nominal,bias:-20,bias:20 \
PHASES=test,summary bash scripts/run_rep_b.sh
```

## Formal run

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_b_hypothesis_image \
REP_A_WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness \
GPUS=0,3,5 EPOCHS=20 PAIRWISE_WEIGHT=0 PHASES=train \
bash scripts/run_rep_b.sh

WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_b_hypothesis_image \
REP_A_WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness \
GPUS=0,3,5 PHASES=test,summary bash scripts/run_rep_b.sh
```

Outputs are under `WORK_ROOT/train/B*` and `WORK_ROOT/test/B*/<split>`.
`comparison.csv` and `paired_effects.csv` are written under `WORK_ROOT/test`.

## Interpretation gate

Rep-B is useful only if it improves both axes: (1) B0/B1 must improve nominal
within-ray discrimination over the old static RGB-only representation; and
(2) B2 must reduce corruption degradation relative to B1 without losing clean
Novel discrimination. If B0 remains near random, increasing Transformer size
is not the next step; the single-view hypothesis projection itself is lacking
metric evidence.
