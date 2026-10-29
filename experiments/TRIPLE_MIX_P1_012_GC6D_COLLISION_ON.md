# P1-0 / P1-1 / P1-2 and GC6D collision-on evaluation

## Purpose and provenance

This branch builds on codex/triple-mix-p0-abcd-20261027 without changing
training weights, checkpoint fingerprints, previous collision-off AP, or the
2026-10-27 historical Mixed-v2 source. All activities are **diagnostics**.

- P1-0: evaluate the original two-source Point RayRoPE checkpoint on the
  full 235-scene GC6D test set, compare with three-source Point/GVAR and
  derive native GC6D scene-paired confidence intervals.
- GC6D collision-on: perform fresh full test inference on three-source
  Point and Point+GVAR with model-free sensor-depth collision threshold
  0.01, voxel size 0.01 m, approach distance 0.05 m, followed by the
  GraspClutter6DEval.eval_all native benchmark.
- P1-1: decompose raw GC6D physical candidate friction/width/collision
  versus point-filter retention and predicted-versus-target geometry match.
- P1-2: compute per-head source-specific gradient norms and exact cosines
  for Depth, Objectness, Graspness, View, CDF, Width and Total, single GPU,
  without optimization.

**This is an implementation/acceptance document, not a report of newly
measured AP or gradients.**

## Historical checkpoint constraint

The 2026-10-27 Point and GVAR models are saved with keys
(version, protocol, model), **not** main-branch model_state_dict/grasp_design.
They MUST be loaded by their own historical rayrope_gvar.runtime.load_model
with verify=True, checking strict model weights, source fingerprint, DAV2
weights, and depth/data protocol. Do not disable verification or set
strict=False.

Three-source historical model root:

    /data/robotarm/EconomicGrasp-triple-historical-20261027

Three-source experiment root:

    /data/robotarm/result/grasp/rgbgrasp/triple_historical_20261027

Three-source Point checkpoint:

    rayrope_point/train/checkpoint_latest.pt

Three-source GVAR checkpoint:

    rayrope_point_gvar/train/checkpoint_latest.pt

Historical SHA256 Point:

    17db3697376ecea1221b816b4a2589330f172fa47e740ce9af6747848068e9de

Historical SHA256 GVAR:

    0fc7fc6b0762aebff22b08f67156ff3ee60b933f8b24fe9a2c8e597ce38a0d4a

**The exact two-source Mixed-v2 Point checkpoint pathname is not in the
available P0 report.** Never substitute the three-source Point checkpoint.
Locate candidates read-only using:

    python tools/mixed_p1_find_two_source_point.py \
      --search-root /data2/robotarm/result/grasp/rgbgrasp/rayrope_point_gvar_mixed_v2 \
      --max-candidates 80 --output-json /tmp/two_source_point_candidates.json

Protocol discovery only produces candidates. Explicitly provide
TWO_POINT_CKPT and TWO_HIST_ROOT, the original source tree matching
the saved code SHA, before running P1-0 two-source evaluation.

## Execution checkout / safety

Do not reset or switch any active training worktree. Use an independent
checkout from this branch (e.g. a worktree):

    cd /home/robotarm/EconomicGrasp
    git fetch origin
    git worktree add /data/robotarm/EconomicGrasp-p1-012 \
      codex/triple-mix-p1-012-gc6d-collision-on-20261029
    cd /data/robotarm/EconomicGrasp-p1-012

Select an idle, permitted GPU after checking activity. Do not stop existing
jobs, delete data, overwrite checkpoint outputs or modify historical sources.

    export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
    $PYTHON_BIN -m unittest discover -s tests -p 'test_mixed_p1_gc6d.py' -v
    $PYTHON_BIN -m compileall -q tools/mixed_p1_*.py
    bash -n run_triple_mix_p1.sh
    DRY_RUN=1 STAGES=collision,baseline,p11_raw,p11_geometry,p12,compare \
      bash run_triple_mix_p1.sh

The default outputs are isolated under:

    /data/robotarm/result/grasp/rgbgrasp/triple_historical_20261027/p1_012_collision_20261029

Environment variables in run_triple_mix_p1.sh include OUTPUT_ROOT,
THREE_HIST_ROOT, TWO_HIST_ROOT, TWO_POINT_CKPT, POINT_CKPT, GVAR_CKPT,
GC6D_ROOT, GRASPNET_ROOT, GNTRANS_ROOT, FILTERED_SCENES, P0_RESULTS,
GPU, BATCH_SIZE, P11_RAW_MAX_SCENES, P11_FORWARD_FRAMES, P12_TRIPLETS,
P12_MODELS, RESUME and DRY_RUN.

## Suggested explicit environment for smoke / manual calls

    export THREE_HIST_ROOT=/data/robotarm/EconomicGrasp-triple-historical-20261027
    export MIX_ROOT=/data/robotarm/result/grasp/rgbgrasp/triple_historical_20261027
    export POINT_CKPT="$MIX_ROOT/rayrope_point/train/checkpoint_latest.pt"
    export GVAR_CKPT="$MIX_ROOT/rayrope_point_gvar/train/checkpoint_latest.pt"
    export GC6D_ROOT=/data/robotarm/dataset/GraspClutter6D

## Step 1: two-frame collision-on smoke

    CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN tools/mixed_p1_gc6d_collision_eval.py \
      --historical-root /data/robotarm/EconomicGrasp-triple-historical-20261027 \
      --checkpoint "$POINT_CKPT" \
      --gc6d-root "$GC6D_ROOT" \
      --output-root /tmp/gc6d_point_collision_smoke_20261029 \
      --mode infer --collision on --collision-thresh 0.01 \
      --collision-voxel-size 0.01 --collision-approach-dist 0.05 \
      --batch-size 1 --num-workers 0 --max-frames 2 \
      --accept-preprocessing-privilege

Repeat for GVAR. Successful smoke requires strict historical checkpoint
load, finite Nx17 saved predictions and actual collision filtering.
**Do not invoke official eval_all on a two-frame smoke**; the GC6D API
can silently treat missing prediction files as AP zeros.

## Step 2: three-source Point/GVAR collision ON

    STAGES=collision GPU=0 BATCH_SIZE=4 OFFICIAL_EVAL_WORKERS=12 \
      bash run_triple_mix_p1.sh

Models run sequentially on one GPU. Each must emit all 3,055 D435 views
in the 235 native GC6D test scenes, save finite Nx17 arrays, and return
official accuracy [235,13,50,6], AP, AP0.4 and AP0.8. The script recomputes
the returned metrics from the raw official array. Reports must include
the number of grasps removed by collision filter and empty-after-filter
frames. Do not confuse empty files with missing files.

New files:

    $OUTPUT_ROOT/three_point/collision_on/official_accuracy.npy
    $OUTPUT_ROOT/three_point/collision_on/official_summary.json
    $OUTPUT_ROOT/three_gvar/collision_on/official_accuracy.npy
    $OUTPUT_ROOT/three_gvar/collision_on/official_summary.json

The per-model inference_protocol.json includes checkpoint SHA, source SHA,
dataset path, score/geometry protocol and collision parameters. An existing
directory can be resumed with RESUME=1 only when its manifest is identical.
If the model-free filter removes every candidate from a frame, the saved
Nx17 empty dump remains intact for audit; an isolated temporary symlink
view intentionally omits that frame so the **native GC6D evaluator's
documented missing-file branch returns zero AP**, without inventing a
dummy grasp. The official_summary records how many frames used this
explicit empty-prediction fallback. No ordinary missing predictions are
accepted; 3,055 original dumps are checked before native evaluation.

**Modality boundary:** neural geometry is RGB-predicted, but workspace/crop
uses sensor depth and segmentation; collision-on additionally uses a
sensor-depth point cloud. Do NOT label this strict sensor-free RGB-only.
The native official evaluator also performs its own CAD collision/
background checks, independent of this post-filter.

## Step 3: original two-source Point baseline on GC6D

Once the exact earlier checkpoint is located and the matching source
worktree verified:

    export TWO_HIST_ROOT=/ACTUAL/TWO_SOURCE/HISTORICAL_SOURCE
    export TWO_POINT_CKPT=/ACTUAL/TWO_SOURCE/POINT_CHECKPOINT
    STAGES=baseline GPU=0 bash run_triple_mix_p1.sh

Produces both collision-on and collision-off from the same model forward
and native GC6D eval for each. Check old protocol: 2,600 GraspNet frames,
2,600 matched GN-Trans frames, LR 3e-4, WD .001, global batch 9,
seed0, 20 epochs, final e19. Differences from three-source may still
include **different optimizer updates per epoch**. Thus they are a
data-plus-budget comparison, not a causal GC6D-only ablation.

## Step 4: paired GC6D comparison

    STAGES=compare bash run_triple_mix_p1.sh

Reads existing P0 Point/GVAR OFF official accuracy arrays and new ON
official accuracy. If the two-source baseline ON+OFF arrays exist, adds
two-source versus three-source and baseline collision-on deltas.

Outputs: p10_comparisons/paired_gc6d_ap.csv,
paired_gc6d_scenes.csv and summary.json.

The statistics pair results by the same **235 scenes**, preserving
13 correlated camera views per scene; percentile scene-bootstrap 10,000
replicates, seed0. It is an uncertainty interval for scene sampling, not
train-seed variation. AP@k uses mean of rank-prefix precisions; do not
replace with precision at a single k.

## Step 5: P1-1 GC6D supervision scarcity

CPU (default 24 scenes, because uncompressed raw 300x12x4 grasp tensors
can be large):

    STAGES=p11_raw P11_RAW_MAX_SCENES=24 bash run_triple_mix_p1.sh

Use P11_RAW_MAX_SCENES=0 for all 232 filtered scenes if storage, memory
and CPU time permit.

Raw-object decomposition reads the exact object IDs and collision array
indices in the scene CDF cache, original object scores/offset width and
collision tensors. The candidate buckets form a **mutually exclusive**
partition: invalid friction, valid friction/invalid width, valid
friction+width/collision, successful. Point-retention uses the original
uncollided friction <=0.4 criterion (without a width test), while CDF
physical success considers <=1.2 and max width 0.10m. Thus the diagnostic
can measure how many potentially successful points were filtered before
the model ever sees them. Exact retained counts must agree with cache.

Single-GPU privileged matched-geometry counterfactual:

    STAGES=p11_geometry GPU=0 P11_FORWARD_FRAMES=24 \
      bash run_triple_mix_p1.sh

For the SAME image/checkpoint/labels, run with RGB-predicted metric depth
and with project-generated target metric depth at supervised pixels,
retaining original prediction elsewhere. The intervention precedes
RayRoPE/GVAR context and modifies seed geometry as well as label matching.
Output matched-query rate, positive matched action count, fraction of
zero-positive frames, and center-set spatial displacement. This is NOT
a test-time deployable result and is not GC6D AP.

Outputs: p11_raw/gc6d_raw_label_decomposition.csv and
p11_geometry_point/gc6d_pred_vs_target_geometry.csv with summaries.

## Step 6: P1-2 gradient norms and cross-source cosines

    STAGES=p12 GPU=0 P12_TRIPLETS=3 P12_MODELS=point \
      bash run_triple_mix_p1.sh

For GVAR as a separate diagnostic use P12_MODELS=point,gvar.
Each triplet contains the same paired GN/GN-Trans scene/frame plus one
independent GC6D scene. Compute exact autograd gradients with no
optimizer step, using historical loss weights: Depth 10, Objectness 1,
Graspness 10, View 100, CDF 1, Width 10 and Total.
Parameter groups are metric depth/FiLM, objectness/graspness proposal,
view head, RayRoPE grouping and CDF/width decoder (plus uncovered parameters
explicitly reported). Gradients are compared across data sources only for
the SAME head on the SAME parameter group.

If either gradient norm is zero, cosine is **undefined (null)**, not
negative/zero. Track zero-positive CDF batches separately. Save
per-source gradient norms, cosine, positive actions and matched queries.
The model is placed in training mode, RNG seeded, and its training buffers
restored after each measurement, with no weight update.

Output: p12_point/aggregate_cosine.csv,
per_head_source_gradients.csv, source_batch_loss_and_coverage.csv,
summary.json. These are final-checkpoint batch=1 local gradients, NOT
historical DDP training-gradient trajectories.

## Acceptance criteria and interpretation

- [ ] CPU tests, compileall and bash -n succeed in the actual environment.
- [ ] Historical three-source Point and GVAR source/checkpoint SHA verified;
      two-source baseline source/checkpoint validated independently.
- [ ] Point and GVAR collision-on have 3,055 complete finite Nx17 predictions,
      official [235,13,50,6] accuracy and independently recomputed AP.
- [ ] Original two-source Point GC6D native AP computed under BOTH protocols.
- [ ] No existing P0 collision-off dump overwritten or mutated.
- [ ] Pairing uses full scene identities, no frame-level pseudoreplication.
- [ ] Raw GC6D filter partition and object retention counts agree with labels.
- [ ] Privileged depth counterfactual changes no checkpoint/label cache.
- [ ] Per-head gradients finite or explicitly undefined and training buffers
      restored; no SGD/AdamW step or checkpoint write.
- [ ] New results markdown includes actual commands, checkpoint hash, source
      SHA, real GPU usage, AP table and uncertainty, error/skip details.

Available historical P0 references: three-source collision-OFF native
GC6D Point AP 7.92608%, GVAR 7.82878%; GC6D supervised query matching
~20.5-24.5% and positive matched CDF labels ~3.5-4.1%.
These are NOT collision-ON results, nor evidence of gradient conflict.
