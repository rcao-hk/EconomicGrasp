# Rep-A: reader structure x depth-error training

Created from EconomicGrasp `main` at `52d09f925059bec3643610ecf1f1722894627ee5`.
Branch: `exp/rep-a-depth-robust-reader`.

## Question and scope

Does changing how a grasp representation reads image evidence reduce its
sensitivity to an imperfect monocular depth estimate? Is structured depth-error
training sufficient without changing the reader?

This is the **fixed-action representation experiment A**, not the later
end-to-end pose-recovery experiment C. A candidate's translation, rotation,
width, height, insertion depth and exact-action labels are identical across
all four cells and all test perturbations. We do NOT perturb the scene or
physically move a labelled action. We do NOT tune the depth estimator.

| Cell | Reader | Training depth input |
|---|---|---|
| A0 | Original spatial enhancer + original CVA attention grouping | Nominal predicted depth |
| A1 | Same as A0 | Nominal/structured-perturbed predicted depth |
| A2 | A0 + independent pre-enhancer image-reading branch | Nominal predicted depth |
| A3 | Same as A2 | Same depth-augmentation distribution as A1 |

**Nominal means the original RGB-predicted depth, not clean/GT depth.**

## What the code actually changes

The main-branch CVA reader already samples 2D features. It is NOT replaced by
an invented hard point-cloud crop. A0/A1 import and train these real modules:

- `models.grasp_spatial_enhancer.GraspSpatialEnhancer`
- `models.kview_query_transformer.ViewConditionedAttentionGrouping`

The perturbed depth is passed to BOTH spatial enhancement and grouping's local
depth-delta input. The cached seed pixels and physical action locations do not
change. Enhanced features and seed features are recomputed with gradients on
every training step; old pooled `F_selected/F_mean` are not training inputs.

A2/A3 add a small image reader that samples 40 locations (8 per region) in the
projected closing region, fingers, palm and approach region of the same grasp.
It reads the PRE-enhancer DPT proposal feature, encodes region identity and
local coordinates, and uses cross-attention. It receives **no predicted depth**.
Invalid projections are masked; an all-outside-image read returns zero.
Its output is added to the original reader output with a fixed 0.1 residual
scale. The original depth-conditioned path remains present.

All cells train the original enhancer, original grouping, a full-action
embedding and a common six-output quality scorer. A2/A3 also train the image
reader. The image backbone, proposal DPT, metric depth predictor and candidate
pose/width generators are frozen. This is full reader/scorer training, **not
only a gate or a new MLP on frozen pooled grasp features**.

The shared six-threshold BCE predicts success at friction thresholds
0.2,0.4,0.6,0.8,1.0,1.2. Predicted utility is their mean. There is no auxiliary KD,
geometry reconstruction, consistency, gate or class-balancing loss. These six
sigmoids are independently trained; the code does not impose monotonicity.

A0 is an original-reader baseline under this NEW fixed-action objective. It is
not a claim to reproduce the old full Stage-1 decoder's AP. A2/A3 have additional
parameters; record parameter counts and do not attribute a gain exclusively to
routing without a subsequent capacity-matched control.

## Existing data and the only export needed

Reuse the COMPLETED Rep-P0 cache:

```
P0_CACHE_ROOT/
  train/scene_0000/ann_0000.npz
  test_seen/scene_0100/ann_0000.npz
  test_similar/...
  test_novel/...
```

No DexNet rerun or new label mining is needed. P0's geometry descriptors alone
are insufficient to train a spatial image reader, so `prepare_rep_a_cache.py`
runs the frozen Stage-1 once for each existing frame and adds dense image/depth
memory. This exports the proposal DPT feature at its native spatial resolution,
BEFORE spatial enhancement; it does not export the upsampled 448x448 map.
Default storage dtype is float16 for that feature, float32 for depth and labels.
All cells share the same precision. Export prints actual MiB/frame and a storage
estimate; no claim of negligible storage is made.

For each frame, the cached native grasp is replay-checked with `max_abs <= 5e-5`
against Stage-1 before ANY labels are reused. Wrong checkpoints/configurations
fail, rather than silently attaching old labels to different grasps. Complete
R/w/h/d equality across P0 offsets is also checked. The earlier re-decoding
ray-selector cache is intentionally rejected.

The model receives only RGB/calibration metadata; sensor/rendered depth,
segmentation and CAD features are NOT reader inputs. Dataset preprocessing uses
the existing P0/GraspNet workspace crop convention unchanged. This experiment
does not establish an independent GT-free deployment crop pipeline.

Exported frame data includes `image_feature`, `depth`, `K`, predicted
`objectness/graspness`, seed `token_ids`, unchanged `actions/valid/friction/utility`,
original query IDs, replay error and SHA256 fingerprints. Files are separate
from P0 and atomic. No source label files are modified.

## Reproducibility and memory

- Train scenes 0-99, validation 100-129; enforced ranges, not arbitrary paths.
- Similar 130-159 / Novel 160-189 never tune checkpoints or selection margins.
- Every variant uses the same seed/order/budget and initial shared weights.
- A1/A3 corruption RNG is keyed by seed/epoch/scene/frame, independently of
  dropout/architecture RNG. Test corruption is keyed by seed/split/frame/type,
  shared across variants and severities.
- Checkpoint and margin are selected on nominal validation utility; ties prefer
  lower harm, then fewer changes. No per-severity test tuning.
- Twenty epochs by default for every cell, no unequal early-stopping budgets.
- Gradient accumulation handles the final short chunk correctly.
- Checkpoints include optimizer/RNG/history; resume restarts at the last fully
  completed epoch, not halfway through an optimizer step.
- One frame of dense evidence in memory. No dataset-wide dense-feature preload,
  no CAD evaluator, no DataLoader workers. Preparation shards by scene.
- CPU Stage-1 checkpoint is released after loading. Host-available-memory and
  disk-space guards fail safely. Defaults disable NumPy huge-page advice and
  cap CPU thread counts.
- Per-output locks and owned process groups prevent duplicate writes and leave
  no detached jobs after Ctrl+C. No broad process-name kill is used.
- Atomic writes/resume validate source P0 SHA, Stage-1 SHA, reader source hashes
  and export contract. Corrupt data may be repaired; protocol mismatch requires
  a fresh output root. Resume does not silently accept a changed action set.

## Depth interventions

Training uses 25% nominal frames; otherwise uniform choice among:

- global bias, uniform in [-20,+20] mm;
- global scale error, uniform in [-3%,+3%];
- smooth 5x5 low-frequency error field, bilinearly resized, RMS in [0,10] mm.

These are pre-registered first-round controls, not a fitted model of all real
monocular errors. Invalid input pixels stay invalid. Only a 1-cm positivity
floor is applied; perturbations are NOT silently clamped back into the depth
predictor's [0.2,1.0] training interval. Actual RMS/bias/clamping are recorded.

Default test cases include nominal, signed 5/10/20/40-mm bias, signed 2%/5%
scale error, 5/10/20-mm smooth error, and a 2-pixel depth-only edge shift.
40 mm, 5% and edge shift test beyond parts of the training support.

## Run

Use a fresh WORK_ROOT for smoke. Set P0_CACHE_ROOT to the cache directory you
actually used; the launcher default contains the `_5mm` suffix.

```bash
# First: local code tests (no dataset, Stage-1 weights or evaluator required).
python -m pytest -q tests/test_rep_a.py

# One GPU, 2 frames in each split; prepare only.
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_smoke \
P0_CACHE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources_5mm/cache \
GPUS=0 MAX_FRAMES=2 PHASES=prepare bash scripts/run_rep_a.sh

# Train all cells on the tiny export, then test a short stress suite.
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_smoke \
GPUS=0 EPOCHS=2 TEST_CASES=nominal,bias:-10,bias:10,smooth:10 \
PHASES=train,test,summary bash scripts/run_rep_a.sh
```

Formal run (keep the smoke outputs separate):

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness \
P0_CACHE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources_5mm/cache \
GPUS=0,3,5,6 PHASES=prepare bash scripts/run_rep_a.sh

WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness \
GPUS=0,3,5,6 EPOCHS=20 GRAD_ACCUM_STEPS=1 \
PHASES=train,test,summary bash scripts/run_rep_a.sh
```

`run_rep_a_prepare.sh`, `run_rep_a_train.sh`, `run_rep_a_test.sh` are convenience
wrappers. `PHASES=summary` requires all four variants' requested test splits.
A smaller GPU list runs jobs in waves, not DDP. Default GPU list is only `0`.
`RESUME=1` is default for train/test; preparation always validates and resumes.
Use `REPAIR_CORRUPT=1` only to rewrite corrupt export files.
`MAX_FRAMES` limits preparation per shard and test frames per job; zero means
all. Limited exports are pilots, not the formal complete protocol.

## Outputs and interpretation

```
WORK_ROOT/cache/manifest.json, reader_init.pt, <split>/scene_*/ann_*.npz
WORK_ROOT/train/A*/protocol.json, gradient_check.json, metrics.json, best.json
WORK_ROOT/train/A*/checkpoint_best.pt, checkpoint_latest.pt
WORK_ROOT/test/A*/<split>/summary.json, per_frame.csv, [per_query.csv.gz]
WORK_ROOT/test/comparison.csv, factorial_effects.csv
```

Performance: exact-action utility, Success@0.8, rescue/harm, intervention rate,
fixed candidate oracle-headroom recovery, and the native-score top half.
Sensitivity: probability drift, representation cosine/relative-L2 drift,
selection turnover, performance drop from nominal.
Discrimination: within-ray pairwise accuracy ignoring exact-utility ties.
A representation becoming constant is NOT a positive result just because its
drift is small.

The summarizer checks action hashes and paired frames, then reports A2-A0,
A1-A0, A3-A1, A3-A2 and interaction (A3-A2)-(A1-A0) with scene-level bootstrap
intervals. Intervals are not training-seed uncertainty and not multiplicity
corrected. Per-severity values are diagnostics, not a way to select a test
checkpoint/margin. Similar/Novel have already informed method development;
reserve a new external/robot test for a final claim.

## Testing status

The authoring environment runs the pure-PyTorch/NumPy tests. The four actual
upstream reader backward tests run inside the complete EconomicGrasp checkout;
if those modules are unavailable they are explicitly skipped. Real Stage-1
replay/export and GraspNet training require the user's CUDA/data environment.
No real training or robustness results are claimed by this implementation.
