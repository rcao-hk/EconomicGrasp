# DCR-AIR: Action-aligned Image Readout after DCR-E1-4

Branch: `exp/e1-e2-center-hypothesis-cva`.

## Why this experiment

DCR-E1-4 shows that further DCR fine-tuning improves global bias/scale
corruptions, especially on Seen/Similar, but does not consistently improve
Novel smooth/local geometry errors. The next controlled question is therefore
not "can the same center corrector be made larger?", but:

> Can an independent, action-aligned image evidence path improve center
> selection when the depth-conditioned CVA evidence is locally unreliable?

AIR is intentionally a **minimal structural experiment**, not a new full grasp
generator.

## Controlled contract

Everything below remains fixed:

- the trained DCR checkpoint;
- Stage-1 proposal/query generation;
- the center grid `[-40,-20,-10,0,10,20,40]` mm;
- native rotation, width, height, and insertion depth;
- the E1 exact-action cache and labels;
- the final Stage-1 query score used by official ranking.

AIR adds only one trainable path:

```text
pre-enhancer 2D DPT feature map
+ explicit physical grasp action g=(t,R,w,h,d)
-> project 13 gripper/contact/closing/approach keypoints into the image
-> bilinear image-feature readout
-> action-conditioned attention pooling
-> bounded scalar logit residual
-> add the same scalar to all six DCR CDF logits
```

The scalar shift preserves the ordering of the six friction-threshold CDF
logits. The last AIR layer is zero-initialized, so the initial model exactly
reproduces the frozen DCR center selector.

The AIR reader does **not** consume:

- the active/corrupted depth map;
- the corruption case identity;
- CAD geometry;
- exact-action utility at inference.

Depth still affects the upstream Stage-1/DCR candidate set. AIR is independent
only as an *evidence readout path for an already explicit physical action*.

## Physical alignment

The 13 image sampling points are generated from the same GraspNet action row
that would be executed. They cover:

- grasp origin;
- closing-region center;
- approach corridor;
- two inner contact sides;
- two fingertips;
- two finger roots;
- upper/lower closing boundaries;
- upper/lower approach boundaries.

The rotation convention follows `utils/collision_detector.py`:
`local = (point - translation) @ R`. Therefore evidence is never read at one
center and executed at another.

## Training

AIR reuses the existing E1 exact-action cache; there is no CAD/DexNet call in
training. The entire DCR and Stage-1 models are frozen. Only the AIR reader is
optimized with six-threshold CDF BCE plus a small residual L2 anchor.

Checkpoint selection remains a mechanism metric on Seen validation: macro mean
exact utility after AIR center selection. The initial zero-residual checkpoint
is retained, so a worse trained AIR does not silently replace DCR.

Formal run:

```bash
cd /home/robotarm/EconomicGrasp
git switch exp/e1-e2-center-hypothesis-cva
git pull --ff-only

E1_BASE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct \
DCR_BASE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/dcr_cva_10pct \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/dcr_air_10pct \
GPUS=0,1,2 \
EPOCHS=6 \
PHASES=train,infer,eval,summary \
bash scripts/run_dcr_air.sh
```

Smoke:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/dcr_air_smoke \
GPUS=0 \
EPOCHS=1 \
MAX_TRAIN_FRAMES=4 MAX_VAL_FRAMES=4 INFER_MAX_FRAMES=4 \
GRAD_ACCUM_STEPS=1 PHASES=train,infer \
bash scripts/run_dcr_air.sh
```

The default test suite closes the same DCR-E1-4 corruption conditions:

```text
nominal
bias:-15, bias:+15
bias:-25, bias:+25
scale:-0.03, scale:+0.03
smooth:5, smooth:10
```

Official methods:

| Method | Physical action | Final ranking score |
|---|---|---|
| `native` | Stage-1 native | Stage-1 |
| `dcr_stage1` | frozen DCR-selected center | Stage-1 |
| `air_stage1` | AIR-selected center | Stage-1 |

Thus `air_stage1 - dcr_stage1` isolates the new evidence path while keeping
global query ranking fixed.

## Outputs

```text
WORK_ROOT/train/
  initial.json
  metrics.json
  best.json
  gradient_check.json
  checkpoint_initial.pt
  checkpoint_best.pt
  checkpoint_latest.pt

WORK_ROOT/test/
  dump/{native,dcr_stage1,air_stage1}/...
  traces/<split>/scene_XXXX/ann_XXXX_<case>.npz
  official/<method>/<case>/<split>/{summary.json,accuracy.npy}

WORK_ROOT/comparison.csv
WORK_ROOT/air_effects.csv
```

AIR traces store both DCR and AIR center decisions, the base/fused local
utilities, AIR residuals, keypoint visibility, and Stage-1 scores.

## Go / no-go interpretation

Inspect these cells first:

1. Novel `smooth:5` and `smooth:10`: AIR should improve both directions
   consistently over DCR if independent image evidence addresses the local
   geometry failure.
2. Novel nominal: AIR should not introduce a new no-harm regression.
3. Novel positive bias/scale: check whether AIR reduces the directional
   asymmetry observed in DCR-E1-4.
4. Similar smooth: AIR should retain the modest DCR gains rather than trade them
   away.

If AIR fails these checks, do not respond by making the reader larger. That
would indicate that the bottleneck is more likely candidate/action support or
the need for richer geometry evidence rather than another local selector.

## DCR-E1-4 selection behavior audit

The existing DCR inference traces are sufficient to determine whether the DCR
fine-tune changed offset directionality, without retraining:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/dcr_e1_4_10pct \
bash scripts/run_dcr_selection_audit.sh
```

Outputs:

```text
WORK_ROOT/selection_audit/selection_audit.csv
WORK_ROOT/selection_audit/offset_histogram.csv
WORK_ROOT/selection_audit/transition_matrix.csv
WORK_ROOT/selection_audit/audit_meta.json
```

The audit reports all queries plus per-frame Stage-1 Top-10/Top-50 subsets. It
is deliberately label-free. Offset transitions must not be described as
exact-action rescue/harm unless matched evaluator labels are added separately.
