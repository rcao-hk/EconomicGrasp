# Four RGB-only Grasp Validations — Implementation Status

Date: 2026-11-23  
Base: main  
Branch: exp/rgb-only-four-validation-suite

## Result audit

This branch contains experiment instrumentation and runners. It does not contain
new AP results from running the four validations. No dataset/checkpoint inference
or official GraspNet/GC6D evaluation was executed as part of the branch creation.

Historical results such as Global-FiLM 47.99 Mean AP, the PKD +0.547 AP result,
the DPT Top-10 collision diagnostics, and the prior GN-Trans C1-Detach numbers
are adjacent evidence. They are not results of the four controlled validations.

Reports named A1_A2_A3_实验结果总结_20261104.md,
GC6D_P6_2A_experiment_results_report.md,
rayrope_deployment_handoff_20261123_report.md and
height_mix50_p0_abc_report.md should be audited against the protocols below
before any of their results are claimed as completing a validation. Similar
names or shared modules do not establish protocol equivalence.

## Status by validation

### V1 — Geometry double-coupling counterfactual

Implemented:
- Separate evidence depth, action-anchor depth and local-geometry depth roles.
- Same fixed image-seed indices across conditions.
- Pred/GT depth source overrides and GT ray-shifted action centre.
- Candidate dumps and exact evaluator Top-1/10/50 collision/success labels.

Conditions:
- Baseline: predicted evidence, action anchor and local geometry.
- A: GT evidence conditioning, predicted action and local geometry.
- B1: predicted evidence/local geometry, predicted action anchor shifted along
  the same camera ray to GT z.
- B2: predicted evidence, GT action anchor and GT local depth geometry.
- Oracle: GT evidence, action anchor and local geometry.

Important interpretation boundary:
- The evidence-depth branch conditions the spatially enhanced image
  representation. It is not yet an independent GVAR RGB-probe anchor. The
  current GVAR reader still builds its action probes from the query/action
  centres. Therefore A measures evidence-depth conditioning, not a fully
  independent relocation of RGB probe coordinates. If the intended claim is
  specifically “read RGB at the GT-projected 3D probe positions while keeping
  the output action centre predicted”, add a separate evidence-centre input
  to the GVAR reader before making that claim.
- B2 replaces the action anchor and local depth map; it is not a full GT mesh
  or full GT point-cloud intervention.

### V2 — Fixed action bank and GVAR role masks

Implemented:
- Freeze seed indices, selected angle/depth, width, centre and rotation from
  the all-role GVAR pass.
- Re-score the identical action bank with all-role, contact/closing,
  clearance/approach, six single-role and six leave-one-role-out masks.
- Assert that physical action geometry does not change.
- Exact evaluator script reports per-sample and aggregate Top-1/10/50
  collision, empty and success@0.4/0.8; writes oracle re-ranked dumps.

Not yet implemented:
- A single unified fixed-action-bank comparison across Center ROI, original
  CVA and GVAR readers. This requires a common scoring interface or separately
  trained reader checkpoints that can score the same frozen action bank.
- The role masks are inference-time readout ablations, not separately trained
  models.

### V3 — RayRoPE observation-parameterization consistency

Implemented:
- Virtual zoom/focal scaling.
- Asymmetric crop/principal-point shift.
- Camera roll at configurable angles.
- Feature cosine consistency at corresponding physical probes.
- Top-K action centre displacement, orientation geodesic error and score
  Spearman correlation after mapping predictions to a common camera frame.
- Position-coordinate modes ray, pixel2d and point3d.

Important interpretation boundary:
- pixel2d is a normalized image-plane 2D RoPE baseline, not an additive
  absolute 2D sine-cosine embedding.
- Each positional-coordinate mode must be trained as a separate checkpoint;
  changing the inference flag on one checkpoint is not a fair comparison.
- The runner currently uses the GraspNet dataset adapter. Additional datasets
  require their dataset adapter.

### V4 — Global metric scale versus local geometry

Implemented:
- Predicted/GT depth export with camera intrinsics and mask provenance.
- Global foreground median-ratio alignment.
- Optional support-plane alignment using a GT-fitted plane or an independently
  calibrated plane.
- Depth MAE/RMSE/bias, local-relative depth error, local gradient error and
  surface-normal angular error.
- Depth-override grasp inference with a common valid seed bank across variants.
- Dataset adapter hooks for GN-Trans and GC6D.

Important interpretation boundary:
- GT foreground median ratio and GT-fitted support-plane alignment are oracle
  diagnostics.
- A calibrated plane is non-oracle only if obtained independently of the
  evaluation GT.
- Object-only scale analysis requires a valid foreground mask. Support-plane
  alignment additionally requires a valid background/support-plane mask.
- If the exporter falls back to all valid depth pixels, the scale ratio is not
  object-only and there is no usable background mask; use plane_source=none.
- GN-Trans/GC6D require the exact dataset module/class and any extra constructor
  arguments to be supplied to the adapter flags. Dataset indices should be
  preserved in the manifest.

## Required execution before drawing conclusions

1. Run V1 on seen/similar/novel with a fixed checkpoint and official evaluator.
2. Run V2 on the same samples and report aggregate Top-1/10/50 collision and
   success plus official AP for each role mask.
3. Train separate pixel2d, point3d and ray checkpoints; run V3 under the same
   deterministic observation transforms and report action consistency and AP.
4. Export depth manifests for GraspNet, GN-Trans and GC6D; run V4 scale/local
   geometry diagnostics, then grasp inference and official evaluation.
5. Record checkpoint SHA, branch commit, dataset split, camera, sample manifest,
   exact command, logs and evaluator version for every run.

Smoke tests:

    python -m pytest -q tests/test_rgb_only_four_validations.py

These tests cover synthetic geometry transforms, scale alignment, role-mask
contracts, fixed-action-bank identity and RayRoPE coordinate-mode validation.
They do not certify GPU inference or benchmark evaluation.
