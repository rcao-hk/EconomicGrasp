# P0-1 / P0-2: online frozen-source experiments

Branch: `exp/dav2-metric-grasp-field`. The existing matched20, residual, sweep,
model, inference, and training files are unchanged. These are separate experiment
entry points, not a new implementation of the original baseline.

## Scope and no-cache contract

P0-1 runs the source encoder, metric decoder, proposal, view/CVA grouping, width
and ray head online for each RGB batch, in `eval()` + `no_grad()`. All source
parameters have `requires_grad=False`. Explicit `is_training` flags are false.
Frozen parameters **and persistent buffers** are hashed before/after training.
The optimizer and DDP contain only the selected private scoring control.

No new feature, action, or label cache is generated. P0-1 still reads the existing
canonical GraspNet CDF annotations through `CVAExtendedLabelAdapter`, preserving
uint16 millimetre widths on CPU and doing live nearest-annotation matching. This
is native supervision, NOT a claim of exact labels for predicted widths/centres.
P0-2 calls the existing `ExactGraspNetActionEvaluator` on actual physical actions
at runtime; its JSON/CSV reports are results, never training inputs. CAD assets
are held in RAM for at most one scene. No DexNet evaluator runs in P0-1 training.

## P0-1 controls

| Variant | Trainable | Fixed |
|---|---|---|
| `base` | nothing; inference reference | complete source |
| `feature_only` | private task adapter + zero-output residual readout | source, metric depth, ray profile, centres/views/candidate widths |
| `full` | identical private adapter/readout, with 4 ray-evidence channels | same as feature_only |
| `cva` | private copy of the existing CVA scoring decoder | source, grouping features, all executed candidate widths |

Feature-only removes **only the four explicit ray-evidence channels**. It retains
identical support positions, roles, projection validity, local coordinates and
proposal/relative/metric latents. It is not a pure-RGB-appearance control. Both
residual controls load identical source adapter/readout hidden weights and zero
the final readout layer. Their parameter counts and initial predictions match.

The CVA control is a complete copied scoring decoder, not just the final linear
layer. Its private width feature tokens can contribute to score attention, but
its output width is discarded. Physical candidate widths ALWAYS come from the
frozen original source. This is a same-objective strong scorer control, not a
parameter-count-matched control. Trainable names/counts are recorded.

All three train with unbalanced threshold BCE + query-listwise ranking (0.1,
temperature 0.1). There is no constant auxiliary Base loss or geometry loss in
this fine-tuning objective. Masked losses use globally correct DDP denominators,
and metrics pool counts/histograms instead of averaging batch AUROCs. Single-
class ranking metrics are undefined (`null`), not reported as zero.

Default: **5 additional** epochs, 20% train (5200 frames), 10% Seen monitoring
(780 frames), 3 GPUs x batch 3, AdamW 3e-4 / WD 1e-3, cosine, clip 1, seed 0.
The source is the complete absolute matched20 checkpoint, not a freshly trained
Base. Monitor test_seen but save/use **latest only**: do not select a model or
hyperparameter using official Similar/Novel results. These are warm-start probes,
not a same-total-budget comparison against the original 20-epoch baseline.

```bash
python -m pytest -q tests/test_mgf_p0_online.py

SOURCE_CHECKPOINT=/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_mixed_matched/train/checkpoint_latest.pt \
GPUS=0,1,2 BATCH_SIZE=3 EPOCHS=5 \
bash scripts/run_mgf_p0_1_online.sh
```

Variants run sequentially on the same GPUs; each training run uses DDP. Inference
is sharded. Default `PHASES=train,infer,eval`, `COLLISION=both`. Off/on dumps are derived
from the same network forward. **Collision-on is the primary benchmark result**
to match the historical EconomicGrasp/GraspNet reporting protocol; collision-off
is retained as a secondary network/scoring diagnostic. Sensor depth is used only
for the collision-on post-filter; the source network never receives it.
The inherited GraspNet crop/workspace preprocessing is **not** claimed to be
annotation-independent. This experiment does not resolve that deployment issue.

Useful controls:

```bash
PHASES=train VARIANTS=feature_only,full,cva EPOCHS=20 bash scripts/run_mgf_p0_1_online.sh
PHASES=infer,eval COLLISION=both INFER_BATCH_SIZE=1 bash scripts/run_mgf_p0_1_online.sh
```

Use a new WORK_ROOT for changed training settings/code. `RESUME=1` resumes only
an identical protocol and source hash; latest checkpoints include control weights,
optimizer and per-rank RNG/loader states. They require the original source file.
Do not feed these control-only checkpoints to the old MGF inference entry point.

Default output:

```text
mgf_p0_1_frozen_online/
  feature_only/train/{protocol,gradient_contract,metrics}.json
  full/train/checkpoint_latest.pt
  cva/train/checkpoint_latest.pt
  base/test_collision_on/official/<split>/{summary.json,accuracy.npy}   # primary
  base/test_collision_off/official/<split>/{summary.json,accuracy.npy}  # diagnostic
  <variant>/test_collision_on/official/<split>/...
  <variant>/test_collision_off/official/<split>/...
  comparison/{comparison.md,comparison.json}
```

The comparator rejects mismatched protocols and reads `accuracy.npy` to verify
summary values. A smoke run can use PHASES=train, EPOCHS=1, MAX_TRAIN_FRAMES=30,
MAX_VAL_FRAMES=6, MAX_STEPS=2 in a separate root. Formal AP rejects partial runs.

## P0-2A: fixed actions, observation changes

Load the P0-1 controls and identical frozen source. For each audit frame, choose
8 fixed source-Base queries (half highest-score, half random), retaining all
12x4 angle/depth actions and **their original widths**. Force the SAME image
seed pixels, metric centres and view IDs under every observation condition.
The selector's eval path ignores forced_view_inds; the code explicitly forces
its argmax without turning on training mode. Identity checks run every forward.

GN-Trans replaces only `colorpath` with `scenes/SSSSS/FFFF_color.png`. Original
sensor-space depth/segmentation crop, dimensions and camera metadata are kept.
File dimensions and crop intrinsics must match. Rendering registration and
unchanged object poses are prerequisites of the paired dataset; metadata checks
alone cannot prove registration.

| Condition | Visual latent features | Numeric depth + ray profile |
|---|---|---|
| original | original RGB | original prediction |
| visual_only | material RGB | original prediction |
| geometry_only | original RGB | material prediction |
| both | material RGB | material prediction |

The feature bundle includes proposal, relative and metric latents. Numeric depth
is replaced before spatial enhancement/ViewNet/CVA, and the corresponding ray
profile is replaced separately. Geometry comes from another RGB forward, never
sensor/GT depth. CDF scores are recomputed on the same physical actions. Their
exact labels are evaluated once in the current frame because scene geometry and
physical friction assumptions are unchanged by visual material augmentation.

## P0-2B: fixed image, physical-action changes

Each fixed query uses two anchors: frozen-Base's best angle/depth and a random
nonwinner. Probe native plus:

- same-image-ray camera-Z translation: -20, -10, +10, +20 mm;
- local approach-axis roll: -15, +15 degrees (native angle-bin multiples);
- width: -10, +10 mm.

All modified physical actions get NEW exact CAD/table-collision + force-closure
labels online. Translation reruns CVA grouping at the changed centre. Roll uses
the corresponding native angle-bin prediction (not arbitrary off-grid tilt).
CVA has no explicit hypothetical-width input: its width-intervention scores are
invariant by construction, and this limitation is recorded, not hidden. The
residual readout queries each action's actual width. Invalid widths are excluded
and counted, never silently clamped or assigned fabricated negative labels.

Outputs include per-condition selection regret/utility and observation score
shift, plus action delta-sign agreement excluding target ties (prediction ties
count 0.5). Each action family also reports best-action selection regret. These
are exact-action diagnostic metrics, **not** official GraspNet AP. No external
sensor collision prefilter is applied; collisions are part of exact labels.

```bash
GNTRANS_RGB_ROOT=/path/to/paired_gntrans_rgb \
CONTROLS_ROOT=/data2/robotarm/result/grasp/rgbgrasp/mgf_p0_1_frozen_online \
GPUS=0,1,2 FRAMES_PER_SCENE=1 QUERIES=8 \
bash scripts/run_mgf_p0_2_online.sh
```

Action-only needs no GN-Trans data:

```bash
MODE=action bash scripts/run_mgf_p0_2_online.sh
```

Defaults cover one frame in every scene of Seen/Similar/Novel (30 per split),
sharded across GPUs. Exact force closure is CPU-heavy even when network forwards
use CUDA; start with QUERIES=2 for a small run in a different WORK_ROOT. Increase
FRAMES_PER_SCENE/QUERIES only after the first run succeeds. `FC_MODE=official`
is the default. Optional reuse_contacts checks VERIFY_N actions per evaluator
call against official force closure and fails on any mismatch.

Reports: `mgf_p0_2_online_audit/<split>/frames/*.json`, `summary.json`,
`per_action.csv`. Merge verifies every requested frame and signature. RESUME=1
skips matching completed reports; no reported labels are loaded by the trainer.

## Interpretation boundaries

1. Full > feature-only supports additional value of explicit ray evidence on
   top of the retained latent and support geometry, not all geometry vs none.
2. Fixed-action observation robustness and action-change sensitivity are distinct.
   Do not demand monotonic score decrease for every perturbation: some improve
   actual force closure.
3. Do not select alpha/hyperparameters repeatedly on official test splits.
4. Frozen eval view selection is intentionally deterministic in P0-1; this is a
   scoring attribution probe, not reproduction of original stochastic-view training.
5. New CPU numerical/CLI tests do not substitute for a real CUDA/GraspNet smoke.
