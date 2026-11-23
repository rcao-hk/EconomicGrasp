# RGB-only Grasp: Four Controlled Validation Experiments

This branch is based on main. It adds the first implementation layer for four
validation groups. It is an experiment harness, not a claim that experiments
have already been run on GraspNet, GN-Trans or GC6D.

## Scientific question

How can 6D grasps be synthesized from RGB when metric geometry is uncertain and
camera/image observation parameterization changes? The four validations isolate
geometry double-use, action-conditioned GVAR evidence, observation
parameterization consistency, and global metric-scale bias versus local shape
error.

## Implemented model hooks

- The model accepts a diagnostic-only geometry override that separates
  evidence_depth from action_depth.
- A ray-shift center mode changes only the selected action anchor along its
  camera ray to GT z while retaining predicted local depth.
- A fixed image-seed index override reuses identical seed pixels across
  counterfactual conditions.
- GVAR supports inference-only masks over six spatial roles. These masks add no
  parameters and do not change checkpoint state dictionaries.
- RayRoPE supports three positional coordinate modes: ray, normalized 2D pixel,
  and metric 3D point coordinates in the candidate grasp frame.
- The three positional encodings require separately trained checkpoints for a
  fair comparison. Inference-time switching on a checkpoint trained with a
  different encoding is not a controlled comparison.

## V1 — geometry double-coupling counterfactual

The per-forward key cva_geometry_override accepts:
- evidence_source: pred, gt or override
- action_source: pred, gt or override
- center_z_source: action or gt_ray_shift
- override depth tensors: depth_override, evidence_depth or action_depth

Recommended same-checkpoint conditions:

| Condition | Evidence | Action | Center | Oracle |
|---|---|---|---|---|
| Baseline | Pred | Pred | Pred | No |
| A: fix evidence | GT | Pred | Pred | Yes |
| B1: ray-shift center | Pred | Pred local map | GT z on same ray | Yes |
| B2: full action geometry | Pred | GT | GT | Yes |
| Oracle upper bound | GT | GT | GT | Yes |

Cache baseline token_sel_idx, intersect it with valid GT depth, and pass the
result as cva_fixed_token_sel_idx to every condition. Fail if any condition
changes seed count/order. Use batch size 1 for the initial implementation.

Report official AP/AP0.4/AP0.8 and exact evaluator Top-1/Top-10 collision,
empty, and success@0.4/0.8. Keep all GT-assisted results explicitly marked
oracle.

## V2 — fixed action bank and GVAR role masks

The six roles are ordered:
left_contact, right_contact, closing, finger_body, palm, approach.

Available masks in the geometry utility:
all, contact_closing, clearance_approach, six only-role masks and six
leave-one-role-out masks.

For a fixed-bank comparison:
1. Run the all-role baseline and cache common-valid seed indices.
2. Freeze each query's selected angle and insertion-depth indices from the
   all-role baseline CDF output.
3. Freeze baseline action geometry and width.
4. Re-run each role mask and read only its score at the same query/angle/depth
   index. Do not re-decode angle/depth or replace baseline width.
5. Assert candidate IDs and all physical GraspGroup columns are identical;
   only confidence may change.

This isolates ranking/readout. It does not claim a role mask is independently
trained. Fail fast if the checkpoint does not contain the GVAR reader.

## V3 — zero-shot observation parameterization test

The geometry utility implements:
- Virtual zoom/focal scaling: centre crop then resize, with K' = H K.
- Asymmetric crop/principal-point shift: crop then resize, with K' = H K.
- Camera roll: H = K Rz.T inv(K), new camera coordinates are Rz.T X, and
  predictions are transformed back with Rz.

For each separately trained encoding checkpoint (pixel2d, point3d, ray),
evaluate the same physical observation before/after deterministic transforms.
Sample the same physical 3D probe points from GT depth, project them into each
transformed image, and report feature cosine consistency. Decode top-K grasps,
map camera-roll predictions back to the original camera frame, and report
matched centre displacement (mm), orientation geodesic error (deg), and score
Spearman correlation.

GT depth defines corresponding physical probe points only; it must not enter
model prediction. Feature consistency is not an AP improvement.

## V4 — global scale versus local geometry

The analyzer diagnostics/analyze_depth_scale_geometry.py reads a CSV manifest:

    dataset,split,scene_id,anno_id,pred_depth_path,gt_depth_path,foreground_mask_path,background_mask_path,K_json
    graspnet,test_novel,165,90,/data/pred.npy,/data/gt.npy,/data/fg.npy,/data/bg.npy,"[[420,0,223.5],[0,420,223.5],[0,0,1]]"

Run:

    python diagnostics/analyze_depth_scale_geometry.py --manifest /data/depth_manifest.csv --output-dir /data/v4_scale_geometry --plane-source gt_fit

The analyzer writes raw, global median-ratio aligned, and support-plane aligned
depth maps plus depth MAE/RMSE/bias, local-relative depth error, local gradient
error, and surface-normal angular error. Global median ratio uses GT foreground
depth and is an oracle diagnostic. A GT-fitted support plane is also oracle.
A calibrated plane is non-oracle only if it comes from independent calibration.

The output depth_override_manifest.csv is intended for an inference pass with
both evidence_source=override and action_source=override. Run the official
GraspNet evaluator on each resulting grasp dump to obtain AP; do not infer AP
from depth metrics.

## Reproducibility and fairness

- Fix sample order, checkpoint SHA, camera, split, sample interval and seed.
- Use identical RGB and annotations across variants.
- For V1/V2, freeze the common-valid image seed bank.
- For V2, freeze action geometry, angle/depth indices and width; vary only the
  reader role mask and confidence.
- For V3, transform image/depth/intrinsics consistently and map actions back to
  a common camera frame before measuring consistency.
- For V4, separate calibrated-plane alignment from GT-assisted oracle alignment.
- Keep model-free collision and exact CAD/DexNet evaluator collision separate.
- Record command, git SHA, checkpoint SHA, manifest, logs and output directory.

## Smoke tests

    python -m pytest -q tests/test_rgb_only_four_validations.py

These tests are synthetic CPU geometry/contract tests. They do not replace
real checkpoint inference or GraspNet/GC6D evaluation.


## Geometry intervention semantics (important)

The model exposes three distinct depth roles:

- evidence depth: conditions the image feature enhancement / visual representation;
- action depth: back-projects selected seed pixels to grasp action anchors;
- local geometry depth: supplies the local depth geometry to CVA/GVAR readers.

The V1 override dictionary therefore also accepts local_source = pred, gt or
override. The refined conditions are:

| Condition | Evidence depth | Action anchor | Local geometry depth |
|---|---|---|---|
| Baseline | Pred | Pred | Pred |
| A | GT | Pred | Pred |
| B1 | Pred | Pred, then GT z-shift on the same ray | Pred |
| B2 | Pred | GT | GT |
| Oracle | GT | GT | GT |

This distinction matters: replacing only the action anchor does not replace
local finger/closing-region geometry. B2 now replaces both the anchor and
local depth geometry while retaining predicted RGB-evidence conditioning.
The tensor shortcut cva_depth_override used by V4 applies the same override to
all three roles.
