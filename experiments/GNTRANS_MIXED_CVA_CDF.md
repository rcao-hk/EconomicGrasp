# GN-Trans + GraspNet mixed CVA-CDF training

## Purpose

This branch tests one question before adding repair or distillation:

> Does exposing the unchanged RGB-only EconomicGrasp-DPT CVA-CDF model to
> material-shifted GraspNet renders improve grasp synthesis under transparent /
> specular appearance while preserving the original GraspNet domain?

The experiment is intentionally **data-distribution only**.  It does not add a
repair head, KD loss, confidence gate, ray hypotheses, or a new decoder.

Base branch / commit:

```text
main @ ffb73e2cb4d8a05069907dcec53a40fd365a515a
```

Model / supervision remain those of `train_cva_ddp.py`:

- `economicgrasp_dpt`;
- CVA Transformer with `kview_mode=A1`;
- CDF score target + depth-wise width;
- predicted metric depth for network geometry;
- no collision head;
- no repair / no distillation.

## Training data contract

Training scenes are `scene_0000`--`scene_0099` in both domains.

For `mix_train_fraction=0.1`, sampling is performed **independently inside each
source dataset, per scene**.  With 256 frames per scene, stride 10 keeps

```text
0000, 0010, 0020, ..., 0250
```

which is 26 frames / scene:

```text
Original GraspNet: 100 scenes * 26 = 2600 samples
GN-Trans:          100 scenes * 26 = 2600 samples
Mixed epoch:                         5200 samples
```

The selected scene/frame keys are asserted to be identical between domains.
The two subsets are concatenated, so they have equal epoch weight but are
shuffled normally by the DDP sampler; batches are not forced to contain one
sample from each domain.

`test_seen` validation uses the same rule independently for both domains:

```text
Original Seen: 30 * 26 = 780
GN-Trans Seen: 30 * 26 = 780
Mixed Seen validation: 1560
```

The mixed validation loss is only a training monitor / checkpoint heuristic.
Final conclusions should use official AP reported separately on original
GraspNet and GN-Trans.

## GN-Trans supervision

`GraspNetTransDataset` supplies:

- GN-Trans rendered RGB as the network image;
- rendered metric GT depth for depth supervision;
- virtual scene graspness;
- original GraspNet camera metadata / poses;
- original extended CVA-CDF grasp cache through `CVAExtendedLabelAdapter`.

The network does **not** consume observed or GT depth as its grasp geometry.
The GN-Trans inference script asserts that

```text
depth_map_used_for_geometry == depth_net_pred
```

at runtime.

The default launcher uses fused TSDF background supervision in both domains:
original GraspNet keeps the existing `--use_fuse_depth` behavior; GN-Trans uses
rendered GT on object pixels and the paired original GraspNet TSDF depth on
background pixels.  Set `USE_FUSE_DEPTH=0` only if intentionally running the
corresponding non-fused protocol in both domains.

## Important interpretation constraint

`10% + 10%` doubles the number of training images relative to an original-only
10% run (5200 vs 2600 samples/epoch).  Therefore a gain over an old 10%-only
checkpoint is a **feasibility result for mixed material data**, not yet a clean
causal attribution to material augmentation.

If this run is positive, the next controlled data ablation should equalize
training exposure / optimization steps (for example original-only versus
original+GN-Trans under a fixed number of updates).

## Data check

Run before training:

```bash
python check_gntrans_mix_data.py \
  --dataset_root /data/robotarm/dataset/graspnet \
  --gntrans_rgb_root /data/robotarm/dataset/GN-Trans \
  --fraction 0.1 \
  --check_all_selected \
  --load_one \
  --output /tmp/gntrans_mix_check.json
```

Expected selected counts:

```text
train:        2600 original + 2600 GN-Trans
seen:          780 original +  780 GN-Trans
similar:       780 original +  780 GN-Trans
novel:         780 original +  780 GN-Trans
```

## Training

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/cva_cdf_gntrans_mix_10pct \
TRAIN_FRACTION=0.1 \
EVAL_FRACTION=0.1 \
BATCH_SIZE=1 \
MAX_EPOCH=20 \
POSE_DEPTH_MODE=global_film \
USE_FUSE_DEPTH=1 \
bash run_gntrans_mix_cva_cdf_train.sh
```

Fresh training deliberately does not pass `--checkpoint_path`.  The normal
pretrained DINO/DPT initialization inside the architecture remains, but no
trained Stage-1, repair, or distillation checkpoint is loaded.

The trainer writes `gntrans_mix_protocol.json` with selected-domain counts and
index fingerprints.  Full interval checkpoints also store this protocol.

## Evaluation

For the current controlled EconomicGrasp AP convention, use the same original
GraspNet sensor cloud as the optional collision post-filter for **both** RGB
domains.  This isolates the RGB/material intervention:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans \
CKPT=/path/to/checkpoint_XX.tar \
GPUS=0,1,2,3,4,5 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/gntrans_mix_eval/eXX \
SAMPLE_INTERVAL=0.1 \
COLLISION_THRESH=0.01 \
GNTRANS_COLLISION_SOURCE=original_sensor \
POSE_DEPTH_MODE=global_film \
bash run_gntrans_mix_cva_cdf_eval.sh
```

This collision-filtered protocol is not strictly network-only RGB inference;
the post-filter uses a depth-derived cloud.  For a strict RGB-only dump, run

```bash
COLLISION_THRESH=0 GNTRANS_COLLISION_SOURCE=none ...
```

`virtual_gt` is available in GN-Trans inference only as an explicit privileged
collision diagnostic and should not be presented as RGB-only inference.

Both original and GN-Trans predictions are evaluated with the standard
GraspNet evaluator because GN-Trans keeps the same scene geometry, camera, and
grasp annotations.

## First decision

Evaluate fixed epochs rather than choosing only from mixed validation loss.
The first run should answer two separate questions:

1. Original GraspNet AP: does material augmentation preserve / improve the
   ordinary RGB domain?
2. GN-Trans AP: does training on material-shifted RGB improve transparent /
   specular appearance robustness?

Only after this data-level result is known should repair, distillation, or
additional geometry-consistency modules be reconsidered.
