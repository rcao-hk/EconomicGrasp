# P6: action-opportunity, geometry-transmission and independent AP protocols

Date: 2026-11-17.
Target branch: exp/shared-visible-geometry-multibatch.
State: IMPLEMENTED IN SOURCE; FORMAL P6 SERVER RUNS AND OFFICIAL AP HAVE NOT BEEN EXECUTED.

## Research objective and P5 reference

Existing P5 data: P4 full-image depth MAE changes 12.5213 -> 7.3849 mm, but action-relative UDF MAE changes only 1.8814 -> 1.8592 mm. The fixed P2 reader has Corrected-Predicted selected-Q gain +0.2962 Q-pp and P5-1 matched re-training has DiD -0.0553 Q-pp; both scene-paired intervals cross zero. Clean geometry remains a privileged offline comparison. These results refer to a fixed bank of physical actions, not GraspNet AP.

Use only:
- Valid P5-0 geometry report from shared_visible_geometry_p5_run01.
- Corrected P5-1 reader comparison / weights from repaired shared_visible_geometry_p5_run02.
- Previous P1 cache from shared_visible_geometry_p0p1_run01 and P4 depth_evaluate from shared_visible_geometry_p4_run01.

WARNING: Old P5 run01 Corrected cache failed consistency under cuDNN TF32. P5 run02 repaired TF32 and achieved 160/160 identical geometry tokens. The 117 server repair/fix.patch has not been synced as identical source to the original GitHub P5 code; never rebuild/reuse P5 corrected geometry with mismatched cache source hashes. P6-0 and P6-1 consume signed existing outputs rather than rebuilding P5.

## P6-0: opportunity/rescue/harm and action flip

Module: shared_geometry/p6_opportunity.py

Inputs: signed P5-1 p51_reader_comparison/summary.json and its gzip family artifacts. Confirm SHA256, physical IDs, label truth, 5120 families on scene80-99, all modes same physical bank and original P5 gain replicated.

Outputs: P6_ROOT/p60_opportunity/summary.json and decision_flips.jsonl.gz.

Includes:
- Oracle headroom and opportunity fraction, original-Q upper-bound frequency.
- Conditional rescue among families with improvement available.
- Harm/unnecessary movement among families with no improvement available.
- Native-vs-other and all-pair ranking accuracy, native/selected Q and move rate.
- Matched Pred/Corrected/P2/Clean sources and paired frozen/retrained contrasts.
- Transition counts: new harm, harm avoided, new rescues, lost rescues.
- 5000 scene-cluster bootstrap resamples, domains balanced within scene. No IID family significance tests.

Hypothesis tests:
- Little oracle headroom: fixed camera-ray-only action bank limits achievable Q.
- Headroom available with missed rescues: action ranking or action evidence is insufficient.
- Good pairwise ordering but poor selected Q: inspect top choice, score margins, and over-movement.

## P6-1: action-common vs action-discriminative geometry

Module: shared_geometry/p6_transmission.py

Load signed P1 predicted and Clean-Visible physics, and P4 corrected physics, with exact P5 frozen-reader action identities and gain parity. Never rerun the depth corrector, retrain reader, generate new action quality labels or access official test.

Same 75 gripper-STENCIL indices across seven physical grasp actions do NOT mean the same 3D point. A strictly common observation mask includes a stencil site only if it is non-unknown across all seven actions and all three geometry sources. Do not interpret unknown geometry as zero. A strictly masked result can differ numerically from P5's pairwise-valid MAE.

For UDF and projective channel, per family and predeclared region (all, palm proxy, lateral-finger proxy, opening-core proxy), let e(g) = geometry(source,g) - geometry(Clean,g). Decompose e into action-shared mean and zero-mean action-specific component:

    e(g,s) = mean_action e(g,s) + (e(g,s) - mean_action e(g,s))
    MSE_total = MSE_shared + MSE_action_specific

Report total/shared/contrast MSE in mm^2 before/after correction; their improvements; native-relative geometry MAE; common-mode correction energy fraction; observation coverage; subgroup opportunity and decision-rescue/harm analysis; within-scene centered descriptive correlations with decision Q delta. No causal claims from these associations.

Outputs: P6_ROOT/p61_transmission/summary.json and transmission_families.jsonl.gz.

## P6-2A: independent, frozen reader evaluation

Module: shared_geometry/p6_independent_eval.py

The previously reused GraspNet scene80-99 cannot serve as independent holdout. Official test scenes100-189 MUST NOT be used to tune a reader. An independently sourced, previously unseen external action bank is required.

Input: a signed manifest with schema shared-visible-geometry-p6-external-bank-v1, partition independent_holdout, external_dataset_id (not GraspNet or GN-Trans), the P4 depth corrector SHA256, explicit training_scene_keys, development_scene_keys, official_test_scene_keys, frozen_model_before_holdout=true, quality_independently_evaluated=true, clean_geometry_never_model_input=true, offsets_m (seven original/P1 ordered offsets), entries with row={scene,anno,domain}, path to NPZ and its SHA256. The manifest is signed with shared_geometry.data.signed.

Each external NPZ must contain EXACTLY:
- predicted_physics and corrected_physics: [F,7,75,6].
- local_xyz: [F,7,75,3], bank_dims: [F,7,3].
- physical_id: [F,7], family_id: [F], quality_count: [F,7].
- No clean-visible depth, Clean geometry, CAD labels or other teacher inputs.

The frozen P5 predicted and corrected readers are applied to those geometry tokens before the independent count/6 labels are used for evaluation. Results are fixed-bank Q, NOT AP.

The external geometry/quality-bank creation step is not included and has not been executed. Manifest provenance is an assertion and still requires independent audit. Reject official GraspNet test scenes as development holdout.

Output: P6_ROOT/p62_independent/summary.json.

## P6-2B: official GraspNet AP from a full deployable RGB model

Module: shared_geometry/p6_official_ap.py

P5/P6 13k-parameter readers are NOT end-to-end grasp predictors. Provide an actual deployable EconomicGrasp RGB checkpoint separately. The official runner invokes the repository's inference_cva.py and eval.py (graspnetAPI) for all test_seen/test_similar/test_novel scenes100-189, all 256 frames each, with camera=realsense and sample interval 1.0. It refuses privileged observed/GT depth, sparse sampling, frozen-code/weight changes, and checkpoint sweeps.

Stage register locks SHA256 of checkpoint, inference/evaluator source and imported model/dataset/utility Python modules plus a JSON array of architectural inference flags, documents whether independent holdout Q exists, and requires attestation of no test-based model selection. IMPORTANT: inference_cva.py may still access observed sensor point clouds for model-free collision filtering when collision_thresh>0; this is logged separately as observed_sensor_depth_used_for_collision_postfilter, so an RGB-only perceptual model must not be misrepresented as a fully sensor-depth-free pipeline. Stage evaluate is dry-run unless an explicit --run is passed. It writes a start-state before viewing official test and refuses auto resume. For each split it requires 30x256 output NPY files before calling graspnetAPI. Artifacts are recorded with SHA256. Do not compare official AP to fixed-bank Q or present official AP as a P5-reader result.

A sample model-args JSON is an array of strings:

    ["--multi_modal","--use_cdf","--kview_mode","A1",
     "--pose_depth_mode","global_film","--enable_point_rayrope",
     "--enable_gvar","--collision_thresh","0.01","--batch_size","1"]

Use flags matching YOUR ACTUAL deployable checkpoint, not inferred from the example.

## Execution on 117

    cd /home/robotarm/EconomicGrasp-shared-geometry-batch
    git fetch origin exp/shared-visible-geometry-multibatch
    git pull --ff-only origin exp/shared-visible-geometry-multibatch

    export P0P1_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p0p1_run01
    export P4_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p4_run01
    export P5_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p5_run02
    export P6_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p6_run01
    export DEVICE=cuda:0

    bash scripts/shared_geometry/run_p6.sh check
    bash scripts/shared_geometry/run_p6.sh p60
    bash scripts/shared_geometry/run_p6.sh p61

When an external independent dataset, frozen geometry and independent exact-action quality count are available:

    export P62_MANIFEST=/absolute/path/p6_external_bank_manifest.json
    bash scripts/shared_geometry/run_p6.sh p62-independent

For full official AP, register an ACTUAL end-to-end RGB grasp checkpoint and frozen model flags:

    export DEPLOY_CKPT=/absolute/path/frozen_EconomicGrasp_checkpoint.tar
    export DATASET_ROOT=/data/robotarm/dataset/graspnet
    export P62_MODEL_ARGS_JSON=/absolute/path/locked_inference_flags.json
    export P62_ATTEST=1
    bash scripts/shared_geometry/run_p6.sh p62-register

    export P62_OFFICIAL_OUTPUT=/data/robotarm/result/grasp/rgbgrasp/p6_official_once
    bash scripts/shared_geometry/run_p6.sh p62-dry-run
    # Only after independent review of the frozen run:
    P62_EXECUTE=1 bash scripts/shared_geometry/run_p6.sh p62-official

If P62_INDEP_SUMMARY points to a signed independent P6-2A summary at registration time, its SHA256 is included. Otherwise the registry records independent_gate_completed=false. Always keep old P4/P5 roots read-only and use a distinct new P6_ROOT.

## Acceptance criteria / limitations

1. CPU regression tests and compile checks pass.
2. P6-0 replays P5-1 signed fixed-reader Q exactly, with 5120 aligned families.
3. P6-1 replays fixed-P5 Corrected-Predicted Q delta to 2e-6 and verifies orthogonal MSE decomposition on the strict valid mask.
4. No independent-validation statement is justified until the external frozen bank actually exists and executes.
5. No official AP statement is justified until the registered full RGB inference/eval executes with complete grasp outputs.
6. Existing validation has been used to select models and analyze P0-P5; resulting confidence intervals are exploratory only. Do not reinterpret them as independent generalization intervals.

CURRENT STATE: code/documentation only; no P6 numerical evaluation was performed on 117 by this authoring operation.
