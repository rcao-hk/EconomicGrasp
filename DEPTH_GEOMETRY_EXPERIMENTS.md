# CVA depth-collapse diagnosis and local geometry training

These independent entry points start from a healthy **Stage-1/2 RGB student**
checkpoint on main. They retain frozen DINO and all existing geometry detach
paths. There is no teacher, mesh evaluation, DexNet call, or new grasp cache.
Python entry points and helpers live at repository root. The four supplied
launchers are tracked explicitly even though the existing `scripts` ignore rule
remains in place; unrelated ignored scripts are unaffected.

## Start here

Activate the same Linux/CUDA environment that already runs
`train_cva_distill_ddp.py`, including its compiled point/attention dependencies.
The dataset needs the existing extended CDF/width cache, GT depth, graspness and
camera metadata. Set the paths once:

```bash
export DATASET_ROOT=/path/to/graspnet
export CHECKPOINT=/path/to/healthy_stage1_student.tar
export CUDA_VISIBLE_DEVICES=0
export OUTPUT_ROOT=/path/to/depth_geometry_results

python test_cva_depth_geometry.py
bash scripts/diagnose_cva_depth_contrast.sh
```

`pose_depth_mode` and `use_fuse_depth` are read from checkpoint metadata.
Explicit overrides (`POSE_DEPTH_MODE`, `USE_FUSE_DEPTH=0|1`) must agree with it.
`use_fuse_depth` controls **training GT construction**, not observed-depth input
to the RGB student. Supply the same `MIN_DEPTH`/`MAX_DEPTH` used by your original
checkpoint if it did not use 0.2/1.0 m. Old checkpoints do not reliably record
these bounds. Nonempty output directories are rejected unless resuming training.

## Contrast diagnosis

The default scan selects 32 frames evenly across `test_seen`, rather than taking
the first scene. It uses deterministic Top-1 view selection regardless of your
normal Top-4 AP inference protocol. Change split or enable the optional input
gradient probe with:

```bash
DIAG_SPLIT=test_novel DIAG_MAX_FRAMES=64 PROBE_GEOMETRY_GRADIENT=1 \
  bash scripts/diagnose_cva_depth_contrast.sh
```

For each image the frozen DINO/DPT outputs are computed once and reused. Only
the geometry depth is replaced by `mean(D) + beta * (D - mean(D))`, with beta in
0, 0.25, 0.5, 1, 1.25. The mean is fixed; intervened depths are **not clamped**.
Out-of-range fraction is logged so clipping/range effects cannot masquerade as
shape effects.

Two protocols are evaluated with the **same student weights**:

* `dynamic`: rerun image-FPS, view selection, label matching and support grids.
* `fixed_gt`: project cached anchors into the camera, reject occluded/off-image
  anchors against visible GT depth, choose GT view-graspness argmax, then record
  and replay the actual physical positions, views, both label passes and all
  local sampling grids. These are fixed values, not merely detached values.

The fixed protocol uses at most 256 unique visible anchor pixels per frame.
Unsupported views retain the existing CDF validity masks. Frames with no usable
fixed supervision are explicitly reported, not silently treated as zero loss.
The retained RGB appearance/GSE features can still respond to the intervened
depth; the fixed protocol isolates them from changes in query geometry and
supervision. It is a diagnostic intervention, not a deployable GT-input model.

Outputs:

* `contract.json`: exact arguments, selected frame indices and protocol.
* `frames.jsonl`: losses, valid/positive labels, CDF denominator, anchor matching
  distance, depth/foreground errors, spatial std and paired local errors.
* `summary.json` / `summary.csv`: frame-paired beta-minus-one differences within
  each protocol; skipped fixed frames are listed in JSON.
* `depth_*.png`: identical-colour-scale depth visualizations for the first frames.
* `gradient_probe.jsonl` (optional): fixed-query input derivative through GSE and
  sampled support depth, its contrast-direction finite-difference check, and
  alignment with GT depth error. No parameter update is performed. This is a
  model input sensitivity, not a physical grasp-success derivative or a claim
  about a DPT optimizer step. Clamp/ReLU boundaries may affect finite differences.

Compare beta effects **within** each protocol. Their absolute losses describe
different query sets and must not be directly subtracted. A loss decrease with
flattening supports a shortcut only when checked against geometry degradation
and the label counts. These outputs are not GraspNet AP; use the normal evaluator
separately for final checkpoint comparison.

## Three training controls

First run a short smoke test in the real training environment:

```bash
MAX_TRAIN_FRAMES=8 MAX_STEPS_PER_EPOCH=2 EVAL_MAX_FRAMES=4 EPOCHS=1 \
  bash scripts/train_cva_depth_geometry.sh
```

Then run matched controls, all initialized independently from `CHECKPOINT`:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 EPOCHS=5 \
  bash scripts/run_cva_depth_controls.sh
```

DDP uses `device_ids=None`, matching the existing distillation trainer. Each
process explicitly moves its model and dense inputs to its local GPU, while
variable-length CDF/width/view caches stay on CPU for row selection. Setting
`device_ids=[local_rank]` would make DDP recursively move those full caches to
CUDA and violate the label matcher's CPU-residency requirement. The trainer
checks this requirement before calling the model.

| Variant | Objective in addition to existing grasp supervision |
|---|---|
| `none` | Original full-map masked metric L1 |
| `foreground` | Metric L1 + relative-depth loss on matched foreground pairs |
| `anchor` | Metric L1 + relative-depth loss around visible GT anchor centers |

`TRAIN_SCOPE=joint` fine-tunes the existing grasp and depth heads; all geometry
detach paths remain on. `TRAIN_SCOPE=depth_only` freezes the grasp network and
trains only metric DPT/Pose-FiLM, useful to isolate geometry improvement before
downstream adaptation. Keep the scope identical across compared arms.

For each pair `(i,j)` the added error is
`(D_pred[i]-D_pred[j]) - (D_gt[i]-D_gt[j])`, with SmoothL1 beta 5 mm. Pixel pairs
and weights never depend on student depth or score. Anchors are sampled in an
object-balanced way. Foreground controls use the same integer offsets and GT
center depth within 10 mm, so accepted pair counts and pixel-distance
distributions match exactly; physical scale is approximate and logged. Both
arms share the acceptance mask. Failed control matching rejects the pair in
both arms. Both endpoints must have valid visible GT depth and a GT foreground
label; missing/occluded geometry is not invented. Inter-object foreground pairs
are allowed: this experiment supervises local visible depth structure, not
object-specific surface normals or guaranteed contact pairs.

The original metric L1 anchors absolute depth; relative depth alone cannot
constrain global shifts. Losses are in metres. Defaults: metric weight 10,
relative weight 10, depth/grasp LR 1e-5, cosine schedule, five epochs. These are
initial experiment settings, not tuned claims. `CLIP_MODE=global` preserves the
original joint norm clipping; `separate` is an explicit optimizer ablation.
Do not change clipping, normalization or learning rates between the three arms
when attributing a gain to anchor conditioning.
The native grasp weights are objectness 1, graspness 10, view 100, CDF score 1,
width 10. Both entry points expose `--objectness_loss_weight`,
`--graspness_loss_weight`, `--view_loss_weight`, `--score_loss_weight`, and
`--width_loss_weight` and record them in the run contract. If the initializing
checkpoint used other coefficients, supply those explicitly to every arm.

Check `train_steps.jsonl` for pre-clip depth/grasp gradient norms, spatial std,
relative errors and effective pair counts. Training fails if a non-baseline arm
has no valid relative supervision for the entire epoch. Validation samples the
same GT pairs every epoch and shards frames without DDP padding duplicates.
By default, training scenes 90..99 are held out for validation and excluded from
training entirely. The validation set covers 64 frames across these scenes.
This holdout applies to the new fine-tuning run; the initializing checkpoint may
already have seen these training scenes.
`EVAL_SCENE_IDS` changes the held-out scenes independently of `SCENE_IDS` (the
training filter). This avoids selecting a checkpoint on benchmark test labels.
Validation runs only the depth branch and cached-anchor sampler, saving the
cost of the grasp decoder. `EVAL_SPLIT=test_seen|test_similar|test_novel` is an
explicit exploratory override; do not use those results for final model selection
and then report the same test split as an untouched evaluation.

Checkpoints include `checkpoint.tar`, `epoch_*.tar`, and `best_geometry.tar`.
The latter is selected by mean depth MAE + mean anchor-relative MAE, **not AP**.
They use the original unprefixed `model_state_dict` and Stage-1 input metadata,
so `inference_cva_distill.py --distill_stage 1` can evaluate them with your normal
Top-1/Top-4 and collision-filter protocol.

Resume at a completed epoch by setting `CHECKPOINT` to the new `checkpoint.tar`,
`OUTPUT_DIR` to the same run directory, and `RESUME=1`. Preserve the original
variant, scope, epochs/schedule, batch size, world size, pair settings and frame
selection. Resume validates this contract. To start a different fine-tuning
experiment from a checkpoint, omit `RESUME` and use a new output directory.

All launchers accept additional Python arguments after the script name.
`DRY_RUN=1` prints the assembled command without running Python.

## Verification scope

`test_cva_depth_geometry.py` runs CPU tensor, gradient, replay-contract and CLI
tests, including the real CVA selector/grouping/CDF decoder with synthetic
ViewNet and label fixtures and equivalence to main's supervised loss. It also
uses two CPU/Gloo processes to verify that the production DDP wrapper preserves
nested CPU labels and synchronizes gradients to the combined-batch reference.
This distributed test is skipped when Gloo is unavailable. The suite also
checks Bash syntax, paths containing spaces, DDP/resume launch arguments and
independent control initialization when Bash is available. Bash tests are skipped
if unavailable; Windows can set `DEPTH_TEST_BASH` to a GNU Bash executable.
The tests do not import the full CUDA model/dataset stack. Full dataset loading,
CUDA kernels, multi-GPU training and AP still require the real training machine.
Use the smoke command above before launching full controls.
