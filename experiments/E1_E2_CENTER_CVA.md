# E1/E2: online-image center-hypothesis CVA, 10% GraspNet schedule

Base: `main` commit `52d09f925059bec3643610ecf1f1722894627ee5`.
Branch: `exp/e1-e2-center-hypothesis-cva`.

## Scope and the meaning of joint training

This is the first, action-aligned stage of the proposed end-to-end experiments.
Raw RGB is processed online on every training iteration. Grasp losses update the
DPT **image adapter**, spatial enhancer, candidate-specific CVA grouping, action
context adapter, and the existing monotonic CVA CDF decoder together. It is NOT
training on cached image features, and it is NOT another frozen-feature verifier.

The reference Stage-1 is immutable: foundation backbone, metric depth/global-FiLM,
seed/view/native angle/width/insertion-depth generation all remain frozen and in
eval mode. A separate trainable copy of the image DPT path prevents image updates
from silently moving the depth estimate or native proposals. This is intentional:
otherwise a cached exact-action label could refer to a different physical grasp.
It is not an all-parameters-unfrozen RGB-to-pose experiment.

No previous experimental branch is required. All new code is additive; main's
models, old training/inference scripts, and default exact evaluator are unchanged.

## E1 versus E2

Both variants instantiate the same modules with the same initialization and use
the same actions, labels, train/validation frames and inference candidate budget.

- **E1:** six-threshold CDF BCE for every evaluated, valid center hypothesis.
- **E2:** E1 plus native-relative utility SmoothL1, default weight 1 and beta 0.1.

```
p(c) = sigmoid(CDF_logits(c))
U_pred(c) = mean_t p_t(c)
L_relative = SmoothL1(U_pred(c)-U_pred(0), U_gt(c)-U_gt(0))
```

The relative loss excludes the zero-offset cell and invalid hypotheses. There is
no independent delta head, H/E/B gate, RGB bypass, context token, or consistency
loss in this round. The prediction remains monotonic over the six friction
thresholds. Both variants select the highest predicted utility and prefer native
on exact numerical ties; there is no validation-tuned acceptance threshold.

## Physical-action contract

Offsets `[-40,-20,-10,0,10,20,40]` are **camera-z millimeters**, not Euclidean ray
length and not the gripper's insertion-depth dimension. Translating on a ray
changes x, y, and z consistently. Invalid/out-of-range hypotheses are masked.

For each native action the center branch rereads all angle-conditioned patches
at the shifted xyz using the shared main CVA grouping. It runs the existing
angle/CDF decoder and gathers the **native angle and insertion-depth indices**.
The original native rotation, width and insertion depth are the executed action
parameters for every center offset. The width output head is frozen and never
substitutes its evolving prediction into an exact-labelled action. A zero-init
residual action adapter exposes physical width/insertion depth to the reader.

Internally the shared decoder sees `Q*C` center-view queries with its usual A/D
axes; gathered action CDF logits are `[C,Q,6]`. This initial protocol does not
jointly search new rotations/widths/insertion depths and then reuse stale labels.

Offline labels are freshly evaluated at **all C physical actions**, not copied
from a nearest CAD point, from the native label, or from the old A1-selected pair.
The cache stores actual action tuples, K, input fingerprints and nominal reference
depth; training verifies their identity. No CAD/DexNet runs during training.

## Data and corruption schedule

`SAMPLE_INTERVAL=0.1` means integer frame stride 10: `0,10,...,250` in every scene.
This inherited '10%' schedule is 26/256 frames (about 10.16%), not a random 10%
subset of scenes and not a limit on inference queries.

| Split | Scenes | Frames |
|---|---|---:|
| train | 0--99 | 2600 |
| validation_seen | 100--129 | 780 |
| test_similar | 130--159 | 780 |
| test_novel | 160--189 | 780 |

Cache preparation uses 64 uniformly spaced decoded query indices per frame by
default. Training cache: nominal plus one deterministic sampled bias (+/-20 mm),
scale (+/-3%), or smooth (1--10 mm RMS) joint-error condition per frame. Training
selects one of the two cached conditions per frame/epoch (approximately 50/50).
Validation: nominal and joint +/-20 mm, same sampled query budget, all validation
frames. Checkpoints are selected only by Seen macro selected exact utility.

Inference defaults to **all native queries** (`INFER_QUERIES=0`) on the same 10%
frame schedule, then official evaluation consumes precisely those frames.
Query chunking is recorded in the training protocol and reused at inference;
it must not be confused with frame subsampling. CVA includes query normalization,
so changing the outer query chunk is a new model protocol, not just a free speed
knob. `GROUP_CHUNK` controls the main attention's internal chunk size.

Model inputs are RGB, intrinsics, camera-pose metadata and the existing benchmark
validity mask. Dataset crop/workspace preprocessing is inherited from main; this
experiment does not independently remove its sensor/segmentation preprocessing.

Exact-label voxel/table resolution is explicit, default **5 mm**. Official AP
uses main's GraspNet API evaluation settings, not a custom '5-mm official AP'.
Training exact utility and official AP remain different metrics.

## Update / checks

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin
git switch --track origin/exp/e1-e2-center-hypothesis-cva
# On subsequent updates: git pull --ff-only
python -m pytest -q tests/test_e1e2_cva.py
```

Use the existing Stage-1 RGB-only CVA-CDF checkpoint (pose mode `global_film`).
No Rep-A/B/C checkpoint is required. Point `STAGE1_CKPT` at the correct local file.
All Python CLIs isolate their own argparse flags before importing legacy modules.

## Smoke (new directory, one GPU)

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_smoke \
STAGE1_CKPT=/your/stage1_checkpoint.tar \
DATASET_ROOT=/data/robotarm/dataset/graspnet \
GPUS=0 CACHE_QUERIES=8 PREP_MAX_FRAMES=4 \
MAX_TRAIN_FRAMES=4 MAX_VAL_FRAMES=4 INFER_MAX_FRAMES=4 \
EPOCHS=1 GRAD_ACCUM_STEPS=1 \
PHASES=prepare,train,infer \
bash scripts/run_e1e2_cva.sh
```

Preparation checks nominal replay against main's original Stage-1 decode on the
first frame of each worker. Training checks nonzero gradients in the online DPT,
enhancer, grouping and CDF decoder, and no gradients in the immutable reference.
Official evaluation rejects incomplete smoke dumps.

## Formal 10% experiment

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct \
STAGE1_CKPT=/your/stage1_checkpoint.tar \
DATASET_ROOT=/data/robotarm/dataset/graspnet \
SAMPLE_INTERVAL=0.1 GPUS=0,1 CACHE_QUERIES=64 \
VOXEL_SIZE=0.005 PHASES=prepare \
bash scripts/run_e1e2_cva.sh

WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct \
STAGE1_CKPT=/your/stage1_checkpoint.tar \
GPUS=0,1 EPOCHS=12 GRAD_ACCUM_STEPS=4 \
PHASES=train \
bash scripts/run_e1e2_cva.sh

WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct \
STAGE1_CKPT=/your/stage1_checkpoint.tar \
GPUS=0,1 INFER_QUERIES=0 SAMPLE_INTERVAL=0.1 \
SPLITS=test_seen,test_similar,test_novel \
TEST_CASES=nominal,bias:-20,bias:20 \
PHASES=infer,eval,summary OFFICIAL_WORKERS=2 \
bash scripts/run_e1e2_cva.sh
```

Specify `DATASET_ROOT` in every invocation when it differs from the default.
E1/E2 training uses one GPU per variant, not DDP; `GRAD_ACCUM_STEPS` is the
effective frame-batch size (one online RGB frame per forward). Preparation and
inference shard scenes across all listed GPUs, including within a single split.
Official evaluation now uses the same GPU list as **split-level concurrency
slots**: with `GPUS=0,1,2`, Seen/Similar/Novel for one variant run
concurrently; with two GPUs they run in waves of two. GraspNetEval itself is
CPU-heavy rather than GPU-accelerated, so the approximate CPU evaluator
parallelism per wave is `num_active_splits * OFFICIAL_WORKERS`. Avoid setting
both values too high for host RAM/CPU. Preparation owns one CAD/DexNet evaluator
per GPU worker; start with 1--2 workers and check RAM before increasing
concurrency. No GPU-holder is needed.

## Same-budget no-error-training control

Reuse the exact-action cache, but train on nominal conditions only in a new root:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/e1_cva_no_error_training \
CACHE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct/action_cache \
STAGE1_CKPT=/your/stage1_checkpoint.tar \
VARIANTS=E1 ERROR_TRAINING=0 GPUS=0 \
PHASES=train,infer,eval,summary \
bash scripts/run_e1e2_cva.sh
```

This control has the same center hypotheses/CDF representation as E1. Do not
attribute candidate-budget improvements to error-aware training without it.

## Outputs / resume

```
WORK_ROOT/action_cache/{train,test_seen}/scene_*/ann_*.npz
WORK_ROOT/train/E{1,2}/{protocol.json,metrics.json,best.json,gradient_check.json}
WORK_ROOT/train/E{1,2}/checkpoint_{best,latest}.pt
WORK_ROOT/test/E{1,2}/dump/{model,native}/<case>/scene_*/<camera>/*.npy
WORK_ROOT/test/E{1,2}/official/{model,native}/<case>/<split>/summary.json
WORK_ROOT/comparison.csv
```

Default `RESUME=1`. Preparation resumes atomic per-frame cache files; inference
resumes completed frame/all-case outputs; training resumes at epoch boundaries
with optimizer/RNG state. Official AP resumes whole method/case/split calls.
Changing worker count does not change the scene/frame schedule. Do not change
candidate grid, label voxel resolution, model/data protocol or smoke limits in
an existing output root. Checkpoint changes also require a new inference root.
Existing Rep-C2-v2 caches contain only native/selected pairs and cannot replace
this all-center cache. Native official AP is evaluated only under the first
listed variant to avoid duplicate expensive evaluation.

For the physical-action/score control, run inference in a separate output root
with Python `inference_e1e2_cva.py --score-source stage1`; this keeps original
query ranking while changing only the physical center. The default is model CDF
ranking. Never silently compare utility and official AP as the same quantity.

## Test boundary

CPU tests exercise schedule, sharding, CDF targets, E1/E2 loss differences,
translation/insertion-depth separation, native tie handling, online gradient
flow and immutable reference parameters with explicit lightweight module mocks.
No claim is made that real Stage-1 checkpoints, CUDA kernels or GraspNet CAD data
were executed in the development container. Run the server smoke above first.
