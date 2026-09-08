# P1 — Grasp-Query Center Recentring

Branch: `exp/p0-pose-space-cdf-transport`

## Question

P0 tests whether the Stage-1/teacher physical-center mismatch is geometrically
transportable. P1 performs the next deployable intervention:

> Can a small RGB-only, grasp-query-specific metric correction improve the
> physical centers actually consumed by the CVA grasp pipeline, without trying
> to improve the full dense depth map and without calling Dex-Net during
> training?

The Stage-1 model selects image-FPS seeds exactly as before. For each selected
pixel with predicted z-depth `z`, P1 predicts a bounded residual

\[
\Delta z = r(F_{\rm seed}, z, (u-c_x)/f_x, (v-c_y)/f_y),
\]

and backprojects

\[
c' = \pi^{-1}(u,v,z+\Delta z).
\]

Only `c'` is substituted into downstream view/CVA reasoning and decoding. The
dense DPT depth map is unchanged.

During training, `gt_depth_m` at the selected pixel provides the direct residual
target. During inference, `gt_depth_m` is explicitly removed from the batch
before model forward.

## Files

- `models/p1_grasp_center_recenter.py`
- `train_cva_p1_center_recenter_ddp.py`
- `inference_cva_p1_center_recenter.py`
- `run_p1_center_recenter_train.sh`
- `run_p1_center_recenter_inference.sh`

No existing Stage-0/1/2 or P0 file is modified.

## Model contract

The correction head receives:

- selected DPT/CVA seed feature;
- normalized current z-depth;
- normalized camera-ray x/y coordinates.

It outputs one scalar residual through `max_residual * tanh(raw)`. The final
linear layer is initialized to zero, so attaching P1 initially reproduces the
Stage-1 centers exactly.

Default maximum correction: `±0.08 m`.

The model exports both base and corrected centers plus diagnostics:

- `p1_center_xyz_base`
- `p1_center_xyz_corrected`
- `p1_center_depth_base`
- `p1_center_depth_corrected`
- `p1_center_residual_pred`
- `p1_center_residual_applied`
- `p1_center_residual_target` (training/GT diagnostics only)
- `p1_center_residual_valid_mask` (training/GT diagnostics only)

Key scalar diagnostics include base/corrected z MAE, base/corrected XYZ MAE,
correction magnitude, clipping ratio, target coverage, and residual sign
agreement.

## Training modes

### 1. `center_only` — primary causal experiment

- initialize from controlled Stage-1 checkpoint;
- freeze every Stage-1 parameter;
- train only the P1 head;
- stop existing grasp-loss gradients at the corrected center;
- direct GT center loss is the only gradient updating the head.

This is the first experiment to run. If AP improves, the result directly
supports the claim that sparse physical-center correction alone repairs part of
the RGB-only mismatch.

### 2. `head_grasp`

- Stage-1 remains frozen;
- direct center loss remains active;
- existing grasp losses may also update the P1 head through the corrected
  physical center.

Run this only after the `center_only` mechanism is established.

### 3. `joint`

- jointly fine-tune Stage-1 and P1;
- center loss and ordinary CVA-CDF supervised losses are both active.

This is an optimization follow-up rather than the primary P1 causal control.

## Training loss

The direct P1 objective is Smooth-L1 in metres:

\[
L_{\rm center}
=
\operatorname{SmoothL1}
(\Delta z_{\rm applied}, z^* - z).
\]

Default beta is `0.01 m`. The ordinary CVA-CDF supervised objective is retained
for logging and for `head_grasp`/`joint`. No teacher, force-closure evaluator,
CAD evaluator, or Dex-Net call is introduced.

## Fast scene-stratified subsampling

P1 training adds two experiment-local arguments:

```text
--p1_train_sample_interval FLOAT
--p1_eval_sample_interval FLOAT
```

Both default to `1.0`, preserving the previous full-dataset behavior. A value of
`0.1` keeps about one tenth of the frames **inside every scene**. Sampling is:

- deterministic;
- scene-stratified;
- evenly spread through each scene's frame sequence;
- based on the dataset's actual `scenename` mapping rather than assuming a fixed
  number of frames per scene.

This is deliberately different from selecting only 10% of scenes. All train or
validation scenes remain represented, so object/clutter/viewpoint diversity is
preserved as much as possible during the fast P1 mechanism test.

The trainer reconstructs the DDP samplers and DataLoaders after subsetting.
Training still shuffles the selected subset epoch by epoch; validation remains
deterministic. Checkpoints record:

```text
p1_sampling_protocol
p1_train_sample_interval
p1_eval_sample_interval
p1_train_full_size
p1_train_selected_size
p1_train_scene_count
p1_eval_full_size
p1_eval_selected_size
p1_eval_scene_count
```

A resumed P1 run must use the same train/eval sample intervals. Older P1
checkpoints without these fields are interpreted as `1.0 / 1.0`.

## Recommended 1/10 mechanism run

```bash
git checkout exp/p0-pose-space-cdf-transport

DATASET_ROOT=/path/to/graspnet \
INIT_CKPT=/path/to/stage1_checkpoint_20.tar \
GPUS=0,1,2,3,4,5 \
P1_TRAIN_MODE=center_only \
P1_TRAIN_SAMPLE_INTERVAL=0.1 \
P1_EVAL_SAMPLE_INTERVAL=0.1 \
OUTPUT_ROOT=/path/to/results/p1_center_only_10pct \
POSE_DEPTH_MODE=global_film \
USE_FUSE_DEPTH=1 \
MAX_EPOCH=5 \
bash run_p1_center_recenter_train.sh
```

This is the recommended first mechanism-validation configuration. If center MAE
falls substantially but grasp metrics do not move on the 1/10 validation set,
do not immediately spend compute on a full-data `center_only` run; first inspect
whether sparse center accuracy is actually the causal bottleneck.

## Full-data run

Omit the two interval variables or set both to `1.0`:

```bash
DATASET_ROOT=/path/to/graspnet \
INIT_CKPT=/path/to/stage1_checkpoint_20.tar \
GPUS=0,1,2,3,4,5 \
P1_TRAIN_MODE=center_only \
P1_TRAIN_SAMPLE_INTERVAL=1.0 \
P1_EVAL_SAMPLE_INTERVAL=1.0 \
OUTPUT_ROOT=/path/to/results/p1_center_only_full \
POSE_DEPTH_MODE=global_film \
USE_FUSE_DEPTH=1 \
MAX_EPOCH=5 \
bash run_p1_center_recenter_train.sh
```

`INIT_CKPT` must be the controlled Stage-1 (`distill_stage=1`,
`seed_selection_mode=image_fps`, predicted geometry) checkpoint. The launcher
may also pass `CDF_LABEL_FOLDER` and `GRASPNESS_MODE` if required by the local
dataset setup.

## Inference

```bash
DATASET_ROOT=/path/to/graspnet \
P1_CKPT=/path/to/results/p1_center_only/checkpoint.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/path/to/results/p1_center_only_preds \
POSE_DEPTH_MODE=global_film \
USE_FUSE_DEPTH=1 \
COLLISION_THRESH=0.01 \
bash run_p1_center_recenter_inference.sh
```

The inference entry point removes `gt_depth_m` before forward and records
`p1_inference_summary.json` inside each split output.

For a frozen-base checkpoint, the zero-residual diagnostic is:

```bash
P1_FORCE_ZERO_RECENTER=1 \
... \
bash run_p1_center_recenter_inference.sh
```

This should recover the Stage-1 physical centers and is a useful implementation
sanity check.

## What to compare

At minimum report:

1. Stage-1 checkpoint-20 AP on Seen/Similar/Novel.
2. P1 `center_only` AP under the identical Top-1 + collision protocol.
3. Training/validation:
   - base center z/XYZ MAE;
   - corrected center z/XYZ MAE;
   - correction magnitude and clipping ratio.
4. Re-run the P0 paired diagnostic with the trained P1 checkpoint only after a
   P1-aware P0 adapter is added; the current P0 script instantiates the original
   Stage-1/2 model and therefore should not silently be used on a P1 checkpoint.

## Decision rule

- **Center MAE improves and AP improves:** proceed to `head_grasp`, then consider
  continuous privileged grasp-field supervision.
- **Center MAE improves strongly but AP does not:** the remaining bottleneck is
  downstream representation/ranking, not raw center accuracy alone.
- **Center MAE does not improve:** revisit the residual representation/features
  before adding more KD.
- **Seen improves but Similar/Novel do not:** the recenter head is overfitting
  the Stage-1 depth-error distribution; this argues for uncertainty/multi-depth
  representation rather than a deterministic residual.
