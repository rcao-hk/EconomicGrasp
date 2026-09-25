# EconomicGrasp Behavior Visualization Suite

Branch: `exp/e1-e2-center-hypothesis-cva`.

## Goal

This suite is intended for **model understanding**, not only publication
figures.  It visualizes what the RGB-only EconomicGrasp-DPT-CVA-CDF pipeline
observes, how predicted geometry changes candidate generation/local analysis,
how DCR changes the physical grasp center, and how the same predictions are
judged by the GraspNet evaluator.

The default dataset protocol is deliberately fixed for longitudinal comparison:

```text
camera: RealSense
splits: test_seen / test_similar / test_novel
every scene: annotation 0 / 128 / 255
```

The same tools are importable from training/inference scripts and include an
`every=N` scheduler for later iteration-based snapshots.

## What is visualized

### 1. Observation and geometry

```text
rgb.png
depth_pred_nominal.png
depth_active.png
depth_corruption_delta_mm.png
depth_sensor.png
depth_rendered.png

depth_pred_nominal_minus_sensor_mm.png
depth_pred_nominal_minus_rendered_mm.png
depth_active_minus_sensor_mm.png
depth_active_minus_rendered_mm.png

pointcloud_pred_nominal.ply
pointcloud_active.ply
pointcloud_sensor.ply
pointcloud_rendered.ply
```

"Active/corrupted depth" is the **internally RGB-predicted metric depth after
the synthetic DCR corruption**.  It is not an external depth input to the
RGB-only model.

The sensor/rendered maps are diagnostic references loaded from GraspNet and are
never fed into DCR inference.

### 2. Dense proposal behavior

```text
objectness_fg.png
objectness_overlay.png
graspness.png
graspness_overlay.png
```

The graspness visualization uses the deployed EconomicGrasp-DPT semantics:
channel 2 is clamped to [0,1], not passed through an extra sigmoid.

### 3. Representation before/after the spatial enhancer

```text
feature_pre_enhancer_pca.png
feature_post_enhancer_pca.png
feature_pre_norm.png
feature_post_norm.png

spatial_gate.png
spatial_delta_abs.png
spatial_update_abs.png
..._overlay.png
```

These panels are useful for checking whether a depth corruption changes the
GSE response more strongly than the underlying image feature.

### 4. View / angle / insertion-depth modeling

```text
view_entropy.png
view_margin.png
stage1_angle_depth.png
```

The view maps expose diffuse vs. sharp view decisions.  The angle-depth panel
shows the six-threshold CDF averaged into utility for selected high-Stage-1
queries.

### 5. DCR center hypotheses and CDF behavior

```text
dcr_center_cdf.png
query_stage1_score.png
query_best_local_utility.png
query_selected_offset_mm.png
dcr_center_correction_motion.png
```

`dcr_center_cdf.png` contains, for high-ranked queries:

- center offset × friction-threshold CDF probability;
- mean-CDF utility as a function of the physical center offset;
- the selected DCR center.

`dcr_center_correction_motion.png` draws the actual native-to-selected
translation.  Negative/positive signed offsets are visually separated.

### 6. DCR learned ranking path

DCR-E1-4 uses frozen Stage-1 ranking, but the trained DCR checkpoint also
contains the isolated bounded ranking residual.  It is shown only as a
diagnostic:

```text
dcr_rank_residual_selected.png
dcr_anchored_minus_stage1_score.png
candidate_latent_response.png
```

`candidate_latent_response.png` shows the CVA candidate latent magnitude and
its cosine similarity to the native-center latent across the DCR offset grid.
It is useful for checking whether nearby physical center hypotheses actually
produce distinguishable internal representations before center selection.

These images must not be confused with the deployed DCR-E1-4 score.

### 7. CVA local analysis

```text
local_attention_overlay.png
```

This panel uses the existing CVA grouping debug tensors and shows the
view-conditioned local image patch, attention weights and center location for
high-Stage-1 queries.  This is intended to answer questions such as:

- does the local region still lie on the intended object?
- does the grouping radius/patch move under corrupted depth?
- does attention concentrate on visible object evidence or background?
- is the readout aligned with the physical center being evaluated?

### 8. Final grasps

For each available method:

```text
grasps_native_overlay.png
grasps_dcr_overlay.png
grasps_e1_overlay.png       # if E1_CHECKPOINT is supplied
grasps_air_overlay.png      # if AIR_CHECKPOINT is supplied

grasps_<method>_scene.ply
```

The RGB overlay shows grasp center, approach direction and gripper width.  The
PLY combines the active-depth point cloud with sampled gripper geometry.

### 9. Nominal vs. corrupted geometry

For each non-nominal case:

```text
corruption_native_motion.png
corruption_motion.json
grasps_nominal_dcr_reference.png
```

Stage-1 queries are matched by the exact image token id when possible.  The
motion panel therefore shows how the *same image query* moves in metric space
when RGB-derived depth is corrupted.

This is a diagnostic correspondence, not an assertion that the full physical
grasp (R/w/d) is unchanged.

### 10. E1 vs. DCR

If `E1_CHECKPOINT` is supplied:

```text
e1/query_selected_offset_mm.png
e1/center_correction_motion.png
e1_vs_dcr_offset_delta_mm.png
```

Both correctors act on the same DCR-generated physical center grid for a direct
selection-behavior comparison.

### 11. AIR

If `AIR_CHECKPOINT` is supplied and `air` is selected:

```text
air/query_selected_offset_mm.png
air/air_logit_residual_selected.png
air/air_minus_dcr_offset_mm.png
air/center_correction_motion.png
```

### 12. GraspNet/Dex-Net evaluator outcome

The optional `evaluator` item runs the detailed evaluator logic already used
by `diagnose_graspnet_eval.py`:

```text
evaluator/<method>/ranked_eval.csv
evaluator/<method>/ranked_eval_overlay.png
evaluator/<method>/ranked_eval_scene.ply
evaluator/<method>/summary.json
```

The ranked overlay/PLY colors the predictions using the actual evaluator
outcome, exposing collision/empty/force-closure failures after NMS and
object-wise selection.

This is a **per-frame diagnostic evaluator**, not a replacement for official
split AP.  The default launcher evaluates only `nominal` because Dex-Net is
CPU heavy.  Set `EVAL_CASES=all` only when the extra cost is intentional.

## Presets

### `ITEMS=light`

```text
rgb,depth,proposal,feature,query_response,grasps
```

Useful for frequent snapshots.

### `ITEMS=core` (default)

```text
rgb,depth,pointcloud,proposal,feature,spatial,view,cdf,
query_response,local,grasps,corruption_delta
```

Recommended first full run.

### `ITEMS=all`

Adds:

```text
evaluator,air
```

AIR panels are produced only if an AIR checkpoint is provided.

## Output organization

```text
OUTPUT_ROOT/
  index.html
  summary.csv

  test_seen/
    scene_0100/
      ann_0000/
        index.html
        nominal/
          overview.png
          manifest.json
          ...
        bias_m25/
          ...
      ann_0128/
      ann_0255/
    ...
  test_similar/
  test_novel/
```

Each frame gets a small static HTML browser page plus per-case contact sheets.
The original PNG/PLY/CSV/JSON files remain available for detailed inspection.

## Pre-flight

Use module-style pytest so the repository root is always on `sys.path`:

```bash
cd /home/robotarm/EconomicGrasp
python -m pytest -q \
  tests/test_grasp_behavior_viz.py \
  tests/test_dcr_air.py \
  tests/test_dcr_cva.py \
  tests/test_e1e2_cva.py
```

## One-scene smoke

Start with one Seen scene and two depth conditions:

```bash
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/behavior_viz_smoke \
GPUS=0 \
SCENE_IDS=100 \
FRAMES=0,128,255 \
CASES=nominal,smooth:10 \
ITEMS=core \
bash scripts/run_grasp_behavior_viz.sh
```

Open:

```text
/data2/robotarm/result/grasp/rgbgrasp/behavior_viz_smoke/index.html
```

## Formal three-split run

The default selects all 90 test scenes and frames 0/128/255:

```bash
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/behavior_viz \
GPUS=0,1,2 \
FRAMES=0,128,255 \
CASES=nominal,bias:-25,bias:25,smooth:10 \
ITEMS=core \
RESUME=1 \
bash scripts/run_grasp_behavior_viz.sh
```

Scenes are sharded at scene level, so a single scene remains on one process.

## Add E1 / DCR comparison

```bash
E1_CHECKPOINT=/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct/train/E1/checkpoint_best.pt \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/behavior_viz_e1_dcr \
GPUS=0,1,2 \
ITEMS=core \
bash scripts/run_grasp_behavior_viz.sh
```

## Add evaluator outcome

Recommended first on selected scenes:

```bash
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/behavior_viz_eval \
GPUS=0,1,2 \
SCENE_IDS=100,130,160 \
CASES=nominal,smooth:10 \
ITEMS=all \
EVAL_METHODS=native,dcr \
EVAL_CASES=nominal \
bash scripts/run_grasp_behavior_viz.sh
```

To evaluate corrupted cases as well:

```bash
EVAL_CASES=all
```

## Add AIR later

```bash
AIR_CHECKPOINT=/data2/robotarm/result/grasp/rgbgrasp/dcr_air_10pct/train/checkpoint_best.pt \
ITEMS=all \
EVAL_METHODS=dcr,air \
bash scripts/run_grasp_behavior_viz.sh
```

## Reuse from another experiment every N iterations

The reusable code lives in:

```text
tools/grasp_behavior_viz.py
tools/dcr_visual_probe.py
```

A training/inference script can schedule cheap panels without importing the
offline Dex-Net evaluator:

```python
from tools.grasp_behavior_viz import (
    BehaviorVizWriter,
    imagenet_rgb,
    save_feature_bundle,
    save_heatmap,
)

viz = BehaviorVizWriter(
    root="/path/to/vis",
    items="feature,spatial",
    every=500,
)

if viz.due(global_step):
    out = viz.frame_dir(
        split="train",
        scene_id=scene_id,
        anno_id=anno_id,
        case="nominal",
        iteration=global_step,
    )
    if viz.enabled("feature"):
        save_feature_bundle(out, feature_before, feature_after)
```

Keep expensive operations out of training hooks:

- GraspNet/Dex-Net evaluator;
- Open3D gripper scene export for hundreds of grasps;
- large multi-case sweeps.

Use the offline runner for those.  The intended workflow is to use `light`
during iteration, then use `core`/`all` on the fixed RealSense frame set for
model-to-model comparison.
