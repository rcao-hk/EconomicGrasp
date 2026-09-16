# Robust grasp representation after GN-Trans mixed training

## Scope

This stage keeps the successful data setting fixed:

- 10% original GraspNet training frames;
- 10% GN-Trans material-varied frames;
- pose-aware RGB metric depth (`global_film`);
- CVA Transformer, A1 view protocol;
- CDF score head + depth-wise width;
- identical supervision and decoder.

The intervention is restricted to **local grasp evidence support**. It does not
add repair, distillation, confidence gating, multi-depth center hypotheses, or a
cross-domain consistency loss.

## Question

The current CVA local grouping reads image features from a gripper-aligned patch
whose metric extent is projected using the predicted center depth. The next
experiment asks whether grasp prediction benefits from a representation that is
less exclusively tied to that single metric support.

This is an AP-driven representation experiment. Metric-depth accuracy can be
diagnosed directly from the learned checkpoints and is not used as a gate for
running the grasp evaluation.

## Variants

All variants keep `patch_size^2` local tokens per center-view query.

### `metric` — controlled baseline rerun

Exact existing `ViewConditionedAttentionGrouping` coordinates and computation.
The robust wrapper returns the baseline grid without recomputation.

### `wide` — receptive-field control

The same metric-conditioned, gripper-aligned patch is expanded around the image
query by `WIDE_SCALE` (default 1.5). Token count and all heads remain unchanged.

Purpose: determine whether a possible gain from `dual` is simply due to reading
a wider image neighborhood.

### `dual` — dual-support representation

A checkerboard partition splits the unchanged token budget approximately 50/50:

- metric half: original gripper-aligned metric-conditioned coordinates;
- image half: fixed-radius image-space coordinates centered on the same image
  query (default radius 32 px).

The fixed image-support radius does not use predicted metric depth. The proposal
center, predicted depth, view, labels, CDF head, width head, and decoder are not
changed.

No new trainable parameters are introduced. Therefore model state-dict keys and
shapes stay compatible with the baseline CVA-CDF architecture; the support mode
is stored as checkpoint protocol metadata and must match at inference.

## Training

Fresh training is preferred for the controlled comparison.

```bash
# Baseline rerun
DATASET_ROOT=/data/robotarm/dataset/graspnet \
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans \
GPUS=0,1,2 \
VARIANT=metric \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/robust_repr_metric \
bash run_gntrans_robust_repr_train.sh

# Wider metric support control
DATASET_ROOT=/data/robotarm/dataset/graspnet \
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans \
GPUS=0,1,2 \
VARIANT=wide \
WIDE_SCALE=1.5 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/robust_repr_wide \
bash run_gntrans_robust_repr_train.sh

# Main dual-support variant
DATASET_ROOT=/data/robotarm/dataset/graspnet \
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans \
GPUS=0,1,2 \
VARIANT=dual \
IMAGE_RADIUS_PX=32 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/robust_repr_dual \
bash run_gntrans_robust_repr_train.sh
```

The launcher defaults to the same 10% + 10% data protocol, 20 epochs, A1,
`global_film`, CDF, and fused-background supervision as the existing mixed
baseline.

## Evaluation

Use the same checkpoint epoch and 10% evaluation frames for each variant.

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans \
CKPT=/path/to/checkpoint_14.tar \
GPUS=0,1,2,3,4,5 \
VARIANT=dual \
IMAGE_RADIUS_PX=32 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/robust_eval/dual_e14 \
SAMPLE_INTERVAL=0.1 \
bash run_gntrans_robust_repr_eval.sh
```

The evaluation launcher runs both original GraspNet RGB and GN-Trans RGB for
Seen / Similar / Novel and then uses the standard evaluator. Collision settings
remain identical across variants.

## Primary comparison

First compare:

1. `metric` vs `wide`: does simply enlarging the current support help?
2. `metric` vs `dual`: does a depth-radius-independent image support add value?
3. `wide` vs `dual`: is any dual gain more than a receptive-field effect?

Primary endpoint: GraspNet AP on Seen / Similar / Novel and mean AP. GN-Trans AP
is a secondary robustness endpoint.

Do not interpret a gain as proof of a specific invariance mechanism. At this
stage the claim is intentionally narrower: a changed grasp-evidence support
improves or does not improve the learned RGB-only grasp representation under the
fixed mixed-data training setting.

## Sanity tests

```bash
pytest -q tests/test_robust_grasp_support.py
```

The tests verify:

- `metric` reproduces baseline grid coordinates exactly;
- `wide` keeps token count and expands support radius;
- `dual` keeps token count;
- the image-support half is invariant to changes in the query metric depth when
  the selected image location is fixed.
