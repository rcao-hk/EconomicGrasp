# Rep-B0 visual attribution and full-path depth-error tests

Branch: `exp/rep-a-depth-robust-reader`. Implemented after commit
`455b7e11552dcd5d08fcfef188e1e1974977e713`. Existing Rep-A/B code and checkpoints
are not modified. Python entry points are in the repository root; Bash in scripts.

## 1. Retrained visual-attribution controls

| Control | Image content | Cross-candidate interaction | Initialization/parameters |
|---|---|---|---|
| full | original features | original B0 | exactly original B0 |
| no_image | zeros during BOTH training and testing | original B0 | same modules/parameter count |
| independent_rgb | original features | none: Transformer runs on length-one sequences | same modules/parameter count |

The geometry-only control retains the actual action, offset, projected gripper
geometry, region IDs and visibility mask; it removes image CONTENT, not the
whole image-reader module. This is stricter than inference-time branch removal.
The independent model retains the original candidate-conditioned RGB reader,
action/offset inputs and scorer. It is trained independently, not the A3
`rgb_only` checkpoint intervention. Query/key attention parameters at length
one are redundant, but the parameter count and initialized tensors match B0.

Default: reuse the existing B0 full checkpoint; train only no_image and
independent_rgb. `RETRAIN_FULL=1` retrains all three. When reusing the original
B0, its model_spec is inherited (not its trained weights). Keep seed, learning
rate, training budget and cache equal to the reference run. The default BCE,
AdamW, 20 epochs, seed 0 and frame ordering match the original B0 settings.

Validation remains nominal test_seen, scenes 100-129. Test now includes
**test_seen, test_similar, test_novel**. Seen is explicitly validation_seen,
not an independent holdout. No new calibration on test errors is performed.

Each checkpoint is tested under fixed_0 and its validation-selected margin.
Reports include utility, Success@0.8, rescue/harm, candidate-optimal Top-1 recall
with utility ties, forced-argmax regret, selected regret, native-vs-exact-best-alt
sign accuracy, opportunity rescue recall, near-native pair accuracy and the old
all-pair metric. Exact-best-alt ties are credited using the highest scored tied
alternative; zero score gaps receive half credit. Do not confuse this metric
with a deployable oracle: labels are used only for evaluation.

```bash
python -m pytest -q tests/test_rep_followup.py

# Tiny execution smoke. Results are NOT a fair comparison to the fully trained B0.
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_b0_attr_smoke \
GPUS=0 MAX_TRAIN_FRAMES=8 MAX_VAL_FRAMES=4 MAX_TEST_FRAMES=4 EPOCHS=2 \
bash scripts/run_rep_b0_attribution.sh

# Formal two-control training + three-split test, existing full B0 as reference.
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_b0_attribution \
GPUS=0,3 PHASES=train,test,summary bash scripts/run_rep_b0_attribution.sh
```

`CACHE_ROOT`, `REP_A_WORK_ROOT`, `REP_B_WORK_ROOT`, `BASELINE_CKPT` can override
paths. Outputs: train/<control>/..., test/<name>/<split>/..., and
`test/comparison.csv`, `test/paired_effects.csv` (scene bootstrap; not seed CI).

## 2. What full-path means here

This is NOT merely replacing xyz in the old Rep-A cache. We run the actual
RGB-only Stage-1 forward once to retain its frozen backbone/depth/proposal
outputs. A replay then reruns the real spatial enhancer, image-FPS selection,
backprojection, ViewNet, CVA, CDF/width heads and decoder for each intervention.
Only frozen RGB/depth-head computation is cached; no downstream decoder output
is reused across an intervention. Nominal replay must match the unmodified
Stage-1's full decoded array within 5e-5 and preserve seed indices.

| Path intervention | Anchor/backprojection depth | Reader/enhancement depth |
|---|---|---|
| reader_only | original prediction | perturbed prediction |
| anchor_only | perturbed prediction | original prediction |
| joint | perturbed prediction | perturbed prediction |

Depth-validity masks used for image-FPS belong to the anchor path. The original
selector's [0.2,1.0] clamping/fallback behavior is retained and audited. Intrinsics,
RGB and workspace crop stay unchanged. Seed membership may change naturally;
we do not freeze the old proposal set. Consequently even reader_only may change
R/width/depth after decoding: it is NOT the old fixed-action reader-only test.

After each regenerated native grasp, K translations are expanded along its ray.
R,width,height,insertion-depth are fixed WITHIN that new K set, but are regenerated
ACROSS cases/modes. Every learned scorer receives the same new candidate set.
It outputs one grasp per query under fixed_0 and val_selected policies.
The original Stage-1 native output is always reported as stage1_native.

The default learned scorers are A0,A1,B0. Existing A1 is augmentation-trained
reader/scorer only; it is NOT a separately augmentation-trained full Stage-1.
This test reconnects it to the same frozen generator, and does not claim to
have trained a strong full-pipeline augmentation baseline. Optional controls
can be added with SCORERS=A0,A1,B0,no_image,independent_rgb after their training.

Inputs use the same original dataset workspace crop convention as Rep-A/P0.
No sensor/rendered depth, CAD or labels are passed to a model. This does not
establish a new GT-free crop pipeline or robot execution safety.

## 3. Fresh labels and AP

`infer_rep_fullpath.py` produces new actions, predictions, selected indices and
ordinary GraspNet .npy dumps. It never reads old P0 action labels.
`evaluate_rep_fullpath.py` evaluates those NEW physical actions with the existing
CAD/DexNet evaluator. A score or object-id change does not change action identity;
a change to t/R/w/h/d does. Memoization is limited to one frame. Scene CAD cache
is explicitly cleared on scene transitions. CPU evaluation defaults to one worker.

For speed, LABEL_SCOPE=selected evaluates the union of executed actions plus
native, deduplicated across methods/policies/cases within each frame. Unlabelled
valid hypotheses are stored as NaN with evaluated_mask=False; no oracle or
all-candidate ranking is claimed. LABEL_SCOPE=all optionally labels the full K
sets and enables oracle/decision metrics, at substantially higher cost.

Raw Top-1/10/50 Success@0.8 is a raw action metric, NOT official AP.
`eval_rep_fullpath_official.py` invokes the SAME sampled GraspNet API used by
repo eval.py. It requires QUERY_LIMIT=0, all required frames and the installed
API's anno_sample_ratio extension. Its NMS/collision/force-closure protocol is
not replaced by the raw exact-action metric. No predicted-depth collision
filter is added during inference. Official AP is a separate optional phase.

## 4. Run full-path tests, INCLUDING Seen

```bash
# First validate native replay and evaluator on 2 Seen frames and two biases.
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_smoke \
GPUS=0 SPLITS=test_seen MAX_FRAMES=2 QUERY_LIMIT=16 \
TEST_CASES=nominal,bias:-20,bias:20 EVAL_SHARDS=1 \
PHASES=infer,evaluate,summary bash scripts/run_rep_fullpath.sh

# Three-split diagnostic pilot (64 selected queries per frame).
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_pilot \
GPUS=0,3 QUERY_LIMIT=64 EVAL_SHARDS=1 \
SPLITS=test_seen,test_similar,test_novel \
PHASES=infer,evaluate,summary bash scripts/run_rep_fullpath.sh

# Formal full-query joint-path AP. NEW root because query/case/mode contracts differ.
# Reduce the case set initially: exact evaluation is still expensive.
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_joint_allqueries \
GPUS=0,3 QUERY_LIMIT=0 MODES=joint \
TEST_CASES=nominal,bias:-20,bias:20 \
SPLITS=test_seen,test_similar,test_novel \
PHASES=infer,official bash scripts/run_rep_fullpath.sh
```

Default splits are all three. Frame sampling is inherited from the Rep-A cache
schedule (formal cache: 0,10,...,250; 26 frames/scene), without a second sampling.
MAX_FRAMES caps inference per shard; evaluation uses all emitted frames and does
not truncate a second time. Official FRAME_STRIDE=10 corresponds to that 0.1
ratio and is NOT the same parameter convention as the old mining interval.

All source paths, case/mode sets, checkpoint hashes and code fingerprints are
recorded. Changing the inference contract requires a new WORK_ROOT. Inference
and exact evaluation resume per completed scenario; atomic writes protect
against kill -9. REPAIR_CORRUPT=1 repairs unreadable files, not changed protocols.
Training resumes from the last fully completed epoch. Output locks and owned
process groups prevent stray jobs on normal Ctrl+C. Do not run overlapping
launchers against the same root. Release GPU-holder VRAM before the smoke.

Outputs:

```
WORK_ROOT/inference/<split>/scene_*/ann_*_<mode>_<case>.npz
WORK_ROOT/evaluation/<split>/scene_*/ann_*_<mode>_<case>.npz
WORK_ROOT/dump/<method>/<policy>/<mode>/<case>/<split>/scene_*/realsense/*.npy
WORK_ROOT/comparison.csv, comparison.json, per_frame.csv
WORK_ROOT/official/<method>/<policy>/<mode>/<case>/<split>/summary.json
```

## 5. Test status and interpretation

CPU tests cover exact B0 replay, common parameter initialization/counts,
image-content removal, candidate independence, all-component gradients,
invalid masks, chunked scoring, path routing and method restoration, changed
physical-action labelling, deduplication, resume/summary and rejection of
query-subset runs as formal AP. A fake Stage-1/CAD evaluator tests routing/IO,
not real detector physics. Real Stage-1 checkpoint/CUDA and CAD evaluation must
pass the two-frame runtime smoke; no real performance results are asserted.

Read full vs no_image first (visual content); then full vs independent_rgb
(cross-candidate relations). For full-path tests read absolute nominal and
joint-corrupted outcomes against A1/Stage-1, not only feature drift or relative
headroom. B0's fixed-candidate depth invariance does not imply invariance once
its physical candidates are regenerated. Seen remains validation_seen.
