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

## Inference and GraspNet AP for the controls

The repository already contains the following relevant Python entry points:

| Entry point | Use |
|---|---|
| `inference_cva_distill.py` | Strict Stage-0/1/2 CVA-CDF inference. Use its Stage-1/2 RGB student path for these checkpoints. |
| `inference_cva.py` | General legacy/CDF CVA inference; lacks the distillation checkpoint/input-contract checks. |
| `inference_cva_distill_p0b.py` | P0-B student/teacher/oracle-hybrid diagnostic inference. |
| `eval.py` | Existing GraspNet AP evaluation for a single dump/split; saves the raw accuracy array. |
| `eval_cva_distill_p0b.py` | P0-B-specific AP summary with completeness checks. |

`scripts/` is ignored in this repository. The tracked version did not include
inference/AP Bash launchers for these entries; ignored scripts on the training
machine may exist independently. The new launchers below are explicitly tracked
without changing that ignore rule.

The new root-level `inference_cva_depth_controls.py` orchestrates the existing
`inference_cva_distill.py`, preserving the real model, selector and CDF decoder.
`eval_cva_depth_controls.py` calls `GraspNetEval.eval_seen/eval_similar/eval_novel`.
It does not introduce a new grasp metric or any training-time geometry evaluation.
The existing inference entry now applies `--seed` to Python/NumPy/PyTorch and
seeds data-loader workers from PyTorch. This controls random sampling, but does
not promise bitwise deterministic CUDA kernels.

Keep the original initialization checkpoint as `BASE_CHECKPOINT`. Compare it
with the final `checkpoint.tar` from all three arms after the same training
budget. `best_geometry.tar` is an optional secondary comparison, selected by
depth metrics, not grasp AP; selected epochs can differ and are recorded.

**Edit the user-settings block at the top of
`scripts/run_cva_depth_controls_eval.sh`, then run it without any environment
prefixes:**

```bash
bash scripts/run_cva_depth_controls_eval.sh
```

The file contains explicit settings for the GPU list, per-GPU inference batch,
CPU worker counts, dataset/checkpoint paths, actual training `RUN_TAG`, output
directory, variants, splits and decoding protocol. These values override
inherited environment variables. The configured defaults include:

```bash
GPU_IDS="1,2"
INFER_BATCH_SIZE=3
INFER_NUM_WORKERS=2
EVAL_NUM_WORKERS=4
RUN_TAG="stage1_e15_seed0"  # change to your actual training tag, including any suffix
VARIANTS="base,none,foreground,anchor"
SPLITS="test_seen,test_similar,test_novel"
TOPK_VIEWS=1
FRAME_STRIDE=1
COLLISION_THRESH=0
RUN_MODE="all"
DRY_RUN=0
```

Check `DATASET_ROOT`, `BASE_CHECKPOINT`, `CONTROLS_DIR` and `PREDICTION_ROOT` in
the same block. The default output is
`$OUTPUT_ROOT/ap_${RUN_TAG}_top${TOPK_VIEWS}_stride${FRAME_STRIDE}_bs${INFER_BATCH_SIZE}_mgpu`.
Change `DRY_RUN` to `1` in the file to print the configured commands. Set
`RUN_MODE="inference"` or `RUN_MODE="ap"` there to run only one stage.

Multi-GPU inference schedules complete **variant x split** jobs. Each listed
GPU runs at most one independent model process; a free GPU takes the next job.
Four checkpoints x three splits produce 12 jobs (three controls alone produce
nine). Every job retains the original frame order, seed, batch size and output
layout. No DDP or `NPROC_PER_NODE` is involved. A single variant/split remains
one job, so at most `min(number of GPUs, number of pending jobs)` GPUs are used.

`GPU_IDS="1,2"` assigns one job to device 1 and one to device 2. Each child
receives exactly one `CUDA_VISIBLE_DEVICES` value and uses its local `cuda:0`.
GPU UUIDs are also accepted; unavailable devices raise an error instead of
silently falling back to CPU. `INFER_BATCH_SIZE=3` means three frames **per GPU**,
and does not need to match the training batch. Changing the GPU list alone does
not invalidate completed matching predictions.

All inference jobs must succeed before AP begins. AP runs one variant/split at
a time with `EVAL_NUM_WORKERS` CPU processes; it does not multiply that worker
count by the number of GPUs. It requires the GraspNetAPI/dataset assets, but no
checkpoint or GPU. This entry evaluates grasp AP; it does not yet export depth
error metrics on the inference frames.

| Setting in the combined launcher | Default / meaning |
|---|---|
| `GPU_IDS` | `1,2`; use `1` for one GPU or e.g. `0,1,2,3` for four. |
| `BASE_CHECKPOINT` | Original Stage-1 initialization checkpoint, explicitly configured in the file. Required when `VARIANTS` includes `base`. |
| `CONTROLS_DIR` | Parent of `none/`, `foreground/`, `anchor/`; defaults to `$OUTPUT_ROOT/controls_${RUN_TAG}`. |
| `PREDICTION_ROOT` | Separate directory for predictions and AP results, derived in the settings block or set explicitly. |
| `CHECKPOINT_NAME` | `checkpoint.tar`; can be `epoch_04.tar` or `best_geometry.tar`. |
| `VARIANTS` | `base,none,foreground,anchor`; select a comma-separated subset to evaluate completed arms. |
| `SPLITS` | `test_seen,test_similar,test_novel`; comma-separated subset permitted. |
| `TOPK_VIEWS` | `1`; set `4` for Top-4 decoding, with a new prediction directory. |
| `FRAME_STRIDE` | `1` (full); `10` selects annotation IDs 0,10,...,250 in each scene. |
| `COLLISION_THRESH` | `0`; positive values enable the legacy captured-cloud collision postprocessor. |
| `COLLISION_VOXEL_SIZE` | `0.01` metres. |
| `INFER_BATCH_SIZE` / `INFER_NUM_WORKERS` | `3` / `2` per GPU. |
| `EVAL_NUM_WORKERS` | `4` CPU evaluation processes. |
| `RUN_MODE` | `all`; `inference` only generates predictions, `ap` only evaluates existing predictions. |
| `CHECK_ONLY` | `1` checks all manifests/dumps without computing AP. |
| `FORCE_EVAL` | `1` recomputes already cached AP results. |
| `DRY_RUN` | `1` prints Bash-assembled commands without requiring files/CUDA. |

The standalone inference/AP Bash launchers still accept environment settings
and extra Python arguments; standalone inference accepts `GPU_IDS` as well.
The combined launcher is configured by editing its own settings block and does
not accept command-line arguments. Python inference's
`--dry_run` also reads/validates the actual checkpoints and prints each underlying
model command, but does not launch inference. Pose-FiLM dimensions and
`use_fuse_depth` are read from checkpoint metadata; no manual fuse-depth flag is
needed. These input contracts must agree across the compared checkpoints.

For an initial cheaper pass, use a new `PREDICTION_ROOT`, `SPLITS=test_seen` and
`FRAME_STRIDE=10`. **Sampled AP requires the existing GraspNetAPI fork supporting
`anno_sample_ratio`.** The unmodified upstream API supports the full protocol
(`FRAME_STRIDE=1`); the new evaluator fails clearly if sampling is unavailable.
It verifies the returned accuracy tensor has exactly 30 scenes x selected frames
x 50 ranks x 6 friction thresholds. Sampled results must be labelled as sampled,
not reported as the complete benchmark.

Do not pass the same literal `sample_interval` to the old Python entries:
legacy inference takes a fraction (`0.1`), whereas `eval.py` takes an integer
stride (`10`). The new entries use `FRAME_STRIDE` once at inference time and read
the exact sampling protocol from manifests during AP evaluation.

Set Top-1/Top-4 and collision filtering identically for every arm. The default
`COLLISION_THRESH=0` disables the extra captured-cloud postprocessor; official
AP still performs its own geometric evaluation. To reproduce older experiments
that used collision threshold `0.01`, set it explicitly for all four checkpoints.
Such postprocessing uses captured depth outside the RGB model, so record that
protocol when interpreting RGB-only results. Additional pre-evaluation NMS is
not applied; the installed GraspNetAPI supplies its normal evaluation behavior.

Outputs:

* `PREDICTION_ROOT/<variant>/<split>/scene_XXXX/<camera>/AAAA.npy`: predicted grasps.
* Each variant/split also has `inference.log` and `inference_manifest.json`,
  recording checkpoint SHA256, input contract, command, seed, sampling and coverage.
* `ap_<split>_<camera>.npy` / `.json`: raw official accuracy tensor and AP summary,
  including per-scene AP and evaluator source fingerprints.
* `PREDICTION_ROOT/ap_summary.csv` / `.json`: AP, AP0.4 and AP0.8 in percent.
* `PREDICTION_ROOT/ap_deltas.csv`: differences in percentage points for
  `none - base`, `foreground - none`, `anchor - foreground`, and `anchor - none`.

AP is the mean precision over ranks 1..50, all evaluated frames and friction
coefficients 0.2,0.4,0.6,0.8,1.0,1.2; AP0.4/AP0.8 fix the friction coefficient.
This follows the [official GraspNet evaluator](https://github.com/graspnet/graspnetAPI/blob/master/graspnetAPI/graspnet_eval.py).
The table describes only the variants/splits requested in the latest evaluation
command; rerun evaluation with the full selection to consolidate cached results.

Repeated identical inference commands skip completed matching variant/split
dumps. An interrupted split is recomputed in full; there is no per-frame resume.
Each concurrent job writes its model output to its own `inference.log`. The
console reports GPU assignments, completions and log paths. If a job fails, the
launcher reports its log tail, stops outstanding jobs and prevents AP from
starting. Ctrl-C and SIGTERM stop/reap child processes; Linux cleanup includes
their DataLoader workers. Completed jobs are retained for the next run.
Different checkpoints, inference settings or recorded code require a new
`PREDICTION_ROOT`. Checkpoint hashes are checked before/after each inference job;
if training overwrites `checkpoint.tar`, that job cannot be marked complete.
Wait until training finishes or select an immutable saved epoch file.
AP refuses missing, extra, corrupt or changed dump files;
valid empty `(0,17)` grasp arrays remain valid evaluation inputs. The AP cache
uses input/evaluator identities, output-array SHA256 and prediction-file size/
mtime fingerprints; prediction fingerprints are not full content hashes.
Prediction files are retained. Matching AP jobs are reused unless `FORCE_EVAL=1`.

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

`test_cva_depth_evaluation.py` adds CPU tests for the real inference frame
selector, checkpoint/command contracts, missing/corrupt dumps, interrupted-run
recovery, AP aggregation/differences/cache, upstream vs sampled API dispatch,
CLI help and real Bash quoting/launching. Real CPU child processes verify GPU
environment isolation, all 12 variant/split jobs, concurrency limits, dynamic
queue refill, cache reuse after GPU-list changes, and failure/interrupt cleanup.
Bash tests edit a temporary copy of the configuration block, verify that those
settings override inherited variables, and check that failed inference prevents
AP. These fixtures test orchestration and metric arithmetic, not physical grasp
correctness or actual CUDA memory/performance. POSIX process-group cleanup of
DataLoader descendants still needs confirmation on the Linux training host.
