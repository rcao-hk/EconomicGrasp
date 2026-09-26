# DAV2 Metric Grasp Field — online training with a fixed detach boundary

Branch: `exp/dav2-metric-grasp-field`.
Created directly from `main` at `52d09f925059bec3643610ecf1f1722894627ee5`.
No Rep-P0/P1 branches are merged. Existing main source files are not modified.

## What is implemented

A shared frozen DAV2 encoder feeds two decoders:

1. The ORIGINAL DAV2 relative-depth decoder, restored strictly from
   `checkpoints/depth_anything_v2_<encoder>.pth`, frozen and kept in eval mode.
2. The existing pose-aware **metric DPT** decoder, still trainable with online
   metric supervision. Its output is NOT the original DAV2 relative depth.

The original decoder's dense feature and relative inverse-depth output, together
with the metric decoder's feature/depth and crop-aware camera rays, feed a new
`MetricRayEvidenceHead`.
This predicts a first-surface depth distribution over fixed metric camera-Z bins.
The initial profile is a 30-mm Gaussian-shaped prior around existing metric depth,
plus a trainable, zero-initialized logit correction. The profile is supervised by
metric labels; it is not described as calibrated epistemic uncertainty.

A separate trainable feature adapter combines proposal features, frozen relative
geometry features and **detached** metric features. The `GraspFieldReadout` queries
27 support points in five physical regions: closing/contact bands, left finger,
right finger, palm and approach corridor. It reads visual features plus signed
metric residual, front/surface/behind probabilities and valid/unknown flags.
It predicts the six-threshold grasp CDF **directly**, not a bounded score residual.

The depth CDF is monotone over camera Z. The grasp CDF is monotone over friction
threshold. Grasp utility is NOT constrained to be monotone over translation or
insertion depth. Surface-behind is unknown, NOT occupied.

The wrapper executes the main encoder once per frame and uses its returned raw
features for the relative decoder. Before sharing features, it verifies the
encoder state equals the official DAV2 encoder. A silently fine-tuned/different
encoder is rejected. No second encoder or external metric foundation model is used.

## Detach contract (fixed for this experiment)

| Module / quantity | Geometry loss | Grasp loss |
|---|---|---|
| DAV2 encoder | frozen | frozen |
| Original relative-depth decoder | frozen | frozen |
| Existing metric DPT + pose adapter | trainable | blocked |
| New metric ray/profile head | trainable | blocked |
| Proposal / spatial enhancer / view / CVA / width | no new geometry supervision | trainable |
| Task feature adapter + field reader | no geometry supervision | trainable |

Every numeric metric depth/profile passed to grasp prediction is detached.
The new metric-feature path is detached too, preventing a latent bypass into
metric DPT. The original undetached depth is retained separately for depth loss.
This is ONLINE joint training with an intentional gradient partition; it does
not mean the metric head is frozen, and it does not study removing detach.

`train/gradient_contract.json` is produced by an actual server forward **before**
training. It checks grasp->metric parameters/depth/profile = 0, geometry gradient
is finite/nonzero, and the field reader has a gradient when valid CDF labels exist.
A zero-valid-label first batch is reported, not misrepresented as a reader test.
The trainer fails if validation has no valid CDF labels.

## Actions and supervision: important boundary

This is NOT the previous fixed-action Rep-P1 diagnostic.

- Main's proposal, image-FPS (configurable), view sampling and width prediction
  remain online and trainable.
- Candidate centres are main's internally predicted-depth centres. This version
  does not introduce +/-40mm centre expansion or label reuse across shifts.
- All angle x insertion-depth candidates are scored. Camera depth and gripper
  insertion depth are different variables.
- Physical readout exactly uses main's decode convention: angle `a*pi/A`,
  insertion depth `(d+1)*.01m`, height `.02m`, and
  width `clamp(1.2*width_prediction/10, 0, grasp_max_width)`.
- Width used by the scorer is detached and is trained with main's width loss;
  the final width is not silently changed after evidence sampling.
- Existing GraspNet CDF/width annotation files ARE still required:
  `economic_grasp_label_300views_extend_angle_cdf_depth/scene_XXXX_labels.npz`.
  No Rep-P0/P1 feature/action cache, feature mining stage or online DexNet call
  is required.

Main's label processor transfers canonical-angle CDF labels from the nearest
annotated centre within 5mm and uses stored annotated widths. These are ordinary
online detector labels, NOT exact evaluator labels for arbitrary predicted
translations/widths. This distinction must remain explicit in comparisons with
Rep-P0/P1. It is impossible to promise arbitrary-action exact labels while neither
reading an action-label cache nor evaluating those actions online.

Default objective:

```
L_task = 1*objectness + 10*graspness + 100*view + 10*width
         + 1*field_CDF_BCE + .25*base_CDF_BCE
L_geometry = 10*main_depth_L1 + 1*profile_CE + 10*profile_mean_L1
```

The existing depth L1 retains main's full-image masked normalization. The new
profile losses average valid geometry labels. Base CDF is a configurable auxiliary
objective, retained to supervise the reused coarse CVA path; it is NOT used as the
final field score. Predicted depth remains supervised even though task gradients
are detached. Invalid GT does not enter losses or prediction-derived masks.

## Data and checkpoint protocol

Default sampling is **all scenes, every tenth annotation**:

- train: 100 scenes x annotations `0,10,...,250` = 2,600 unique frames;
- Seen validation: 30 x 26 = 780;
- Similar/Novel inference: 780 frames each.

This is approximately 10% (26/256), not a random 10-scene subset. Unique schedules
and hashes are saved to `sampling.json`. Standard DDP padding may repeat a few
training frames to equalize ranks (4 repeats for 2,600 frames on 6 GPUs); its count
is recorded. Validation is complete on rank 0, without padded duplicate frames.

Train for 20 epochs by default; no policy-based early stopping. Best checkpoint
is selected using Seen field-CDF BCE. Coverage is logged. This is a practical
training criterion, not proof of final AP superiority. Similar/Novel never choose
checkpoints or thresholds. `checkpoint_latest.pt` and `checkpoint_best.pt` are
both saved. Resume is epoch-boundary only and verifies code, checkpoint, data
schedule, seed, architecture, optimizer and world-size contracts.

## Prerequisites and start

Run from the repository root in the SAME CUDA/MinkowskiEngine/PointNet2/GraspNet
Python environment that already runs main. Standalone CPU contract tests do not
require these extensions. Do not assume the old Python main trainer's flags apply
here; new scripts have their own parsers and set main's global cfg explicitly.

Required files:

- official `checkpoints/depth_anything_v2_vitb.pth` (or matching vits/vitl);
- normal GraspNet RGB/depth-label/metadata resources already required by main;
- the existing CDF/width annotation folder;
- optional trusted main Stage-1 RGB-CDF checkpoint for warm start.

Default launcher uses the previously used Stage-1 path. Set `INIT_CHECKPOINT` to
the real server path. Set it explicitly empty for new task/metric heads:
`INIT_CHECKPOINT=''`. Warm start is recommended for the first integration run to
avoid coupling initial depth/proposal learning with a new field reader.
The 10% restriction applies to THIS run's online frames. It does not erase data
previously used by an initialization checkpoint. For a strict 10%-frame training
budget beyond DAV2 pretraining, use `INIT_CHECKPOINT=''` for every compared model.

```
git fetch origin
git switch --track origin/exp/dav2-metric-grasp-field   # first checkout only
# For an existing local branch: git switch exp/dav2-metric-grasp-field
python -m pytest -q tests/test_metric_grasp_field.py
```

### Separate smoke run

Use the full old checkpoints but a NEW output directory. `M_POINT=64` reduces
only the smoke query count and must not be carried into formal 1,024-query results.

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_smoke \
GPUS=0 INFER_GPUS=0 \
EPOCHS=1 M_POINT=64 BATCH_SIZE=1 GRAD_ACCUM=1 \
MAX_TRAIN_FRAMES=4 MAX_VAL_FRAMES=4 INFER_MAX_FRAMES=4 \
PHASES=train,infer \
bash scripts/run_metric_grasp_field.sh
```

Check `train/gradient_contract.json`, `train/metrics.json`, saved grasp arrays,
and absence of runtime errors. The evaluator refuses smoke/partial checkpoints
and truncated inference schedules as formal AP runs.

### Formal online training + inference + evaluation

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/dav2_metric_grasp_field_10pct \
GPUS=0,1,2,3,4,5 BATCH_SIZE=1 GRAD_ACCUM=4 EPOCHS=20 \
PHASES=train,infer,eval \
bash scripts/run_metric_grasp_field.sh
```

This uses DDP for ONE model, not one variant per GPU. Default effective batch is
24 frames/optimizer update except the final accumulation group. Each inference
split is sharded over the available inference GPUs. Heavy frozen features are
computed online every batch; there is no disk feature cache.

FP32 is default. `AMP=1` is opt-in. Action readout uses recomputation checkpointing
and `ACTION_CHUNK=256`; reduce the chunk before changing the candidate budget
if GPU memory is insufficient. Hardware memory/time have not been benchmarked here.

Phase/config examples:

```bash
RESUME=1 PHASES=train bash scripts/run_metric_grasp_field.sh
CHECKPOINT_KIND=latest PHASES=infer,eval bash scripts/run_metric_grasp_field.sh
TOP4=1 TEST_ROOT=/path/to/separate_top4_dump PHASES=infer,eval bash scripts/run_metric_grasp_field.sh
```

Use a separate root when changing top4 or evidence mode. `RESUME=1` requires an
existing latest training checkpoint in a train phase, and verified inference
markers in an infer phase. Initial runs intentionally default to `RESUME=0`.

## Same-source representation controls

`EVIDENCE_MODE=learned|fixed|hard` controls ONLY task-side ray evidence. All three
use the SAME trained profile head's mean; geometry training still uses the full
profile. `fixed` uses an analytic fixed-sigma Gaussian at that mean; `hard` uses a
point surface at that mean. The hard/fixed/learned comparison therefore must use
separate matched-training runs, or be explicitly labelled an inference-only
perturbation. The launcher stores and verifies this choice in the checkpoint.
The old deterministic depth output is still the proposal-centre source.

Restoring DAV2 relative features and retaining a profile are separate possible
sources of benefit. A first field run establishes integration, not either
mechanism's novelty. Do not claim calibrated probabilities or guaranteed metric
precision merely because 160 bins span .2--1m (5mm bin width).

## Outputs

```
WORK_ROOT/
  train/
    protocol.json, sampling.json, gradient_contract.json
    metrics.json, best.json
    checkpoint_latest.pt, checkpoint_best.pt
  test_best/                         # test_latest for CHECKPOINT_KIND=latest
    <split>/protocol.json
    <split>/completed/*.json
    dump/scene_XXXX/realsense/XXXX.npy
    official/<split>/summary.json, accuracy.npy
  logs/
```

Inference does NOT apply observed-depth collision filtering. The network consumes
RGB and camera metadata; all sensor/GT depth/label tensors are stripped before
prediction. The underlying main dataset crop/workspace preprocessing is retained,
so do not claim the ENTIRE benchmark preprocessing pipeline is annotation-free.
Official evaluation necessarily uses benchmark geometry/labels, not network inputs.

## Validation status at implementation

Local checks exercise pure tensor operations, detach/no-bypass gradients, metric
updates, field chunking/permutation invariance, physical action conventions, parser
entry points, and wrapper/loss/checkpoint integration with an explicitly MOCKED
main frontend. Tests do not replace CUDA/GraspNet server smoke or a trained-model
AP run. Full training/inference was not executed in the development container.
