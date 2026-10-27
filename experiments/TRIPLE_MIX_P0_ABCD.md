# Three-source Mixed-v2: P0-A/B/C/D diagnostics

**Scope:** Diagnosis only; do not change model weights, original 20-epoch training, or existing AP reports. Latest historical Mixed-v2 (2026-10-27) uses 8,216 mixed training frames: GraspNet 2,600, paired GN-Trans 2,600, filtered GC6D 3,016 from 232 scenes. The comparable Point and Point+GVAR runs use the same batch 9, LR 3e-4, WD .001, 20 epochs, RayRoPE/GVAR chunk 2048 and no activation checkpointing. Earlier runs with different settings are not strict ablations.

Main entrypoint: **run_triple_mix_p0_abcd.sh**

## Running on server 10.30.7.117

Checkout:

    cd /home/robotarm/EconomicGrasp
    git fetch origin
    git switch codex/triple-mix-p0-abcd-20261027
    python -m unittest discover -s tests -p 'test_mixed_p0_abcd.py' -v
    python -m compileall -q tools/mixed_p0_*.py
    bash -n run_triple_mix_p0_abcd.sh

Historical default locations are defined in the shell script. Override with MIX_ROOT, POINT_RUN, GVAR_RUN, POINT_CKPT, GVAR_CKPT, GRASPNET_ROOT, GNTRANS_RGB_ROOT, GC6D_ROOT, GC6D_FILTERED_JSON and OUTPUT_ROOT to match real mounts.

CPU-only A / B cache / C:

    STAGES=A,B,C bash run_triple_mix_p0_abcd.sh

Optional fixed-list physical-quality evaluation (expensive CPU; defaults to 3 scenes per split, 6 sampled views each):

    STAGES=A ENABLE_CANDIDATE_EVAL=1 bash run_triple_mix_p0_abcd.sh

Optional true checkpoint-dependent matching/loss probe, 24 frames per source per checkpoint on one GPU:

    STAGES=B ENABLE_FORWARD_PROBE=1 GPU=0 P0_B_MAX_FORWARD_FRAMES=24 \
      bash run_triple_mix_p0_abcd.sh

First smoke-test GC6D Point model, just two frames, no official AP:

    export GC6D_ROOT=/data/robotarm/dataset/GraspClutter6D
    export POINT_CKPT=/data/robotarm/result/grasp/rgbgrasp/triple_historical_20261027/rayrope_point/train/checkpoint_latest.pt
    CUDA_VISIBLE_DEVICES=0 python tools/mixed_p0_d_gc6d_cva_eval.py \
      --gc6d-root "$GC6D_ROOT" --checkpoint "$POINT_CKPT" \
      --output-dir /tmp/p0_d_point_smoke \
      --pose-depth-mode global_film --batch-size 1 --workers 0 \
      --sample-fraction .1 --max-infer-frames 2 \
      --collision-thresh 0 --do-infer

Full GC6D native test inference and official evaluator, sequential Point and GVAR:

    STAGES=D GPU=0 INFER_BATCH_SIZE=4 bash run_triple_mix_p0_abcd.sh

Full GC6D evaluator deliberately refuses subsampled test predictions. Use the official GraspClutter6DEval.eval_all() and its raw AP/AP0.4/AP0.8 return. Do not replace it with nearest GT-pose matching or GraspNet AP. Do not invent results before test completion.

## P0-A: official AP@k, paired bootstrap, candidate oracle

- tools/mixed_p0_a_rank_stats.py loads all 12 **official** collision-off accuracy arrays of shape [30,26,50,6], verifies finite values and SHA256, and reports mean prefix-precision AP@1/5/10/50 alongside exact precision@k. 
- The scene bootstrap (default 5000 replicates) resamples 30 scene IDs per split with replacement. The same indices are used for Point/GVAR and original/GNTrans within each split; the macro result averages six splits. CI is percentile-based scene-sampling uncertainty and does NOT include random-seed training variability.
- Optional tools/mixed_p0_a_candidate_eval.py uses official GraspNet CAD collision/DexNet quality to evaluate fixed saved NMS/global-TopK grasp lists. Its candidate_rows.csv is consumed by rank_stats --candidate-csv to get a privileged fixed-list oracle. This is NOT official AP: native evaluator also applies object-level selection/ranking.
- Official accuracy.npy alone cannot identify raw grasp friction/empty/collision, so no oracle is fabricated without independently computed candidate physical qualities.

Outputs: p0_a_official_ap/official_ap_at_k.csv, scene_pair_deltas.csv, scene_paired_bootstrap.csv, summary.json; optional p0_a_candidate/candidate_rows.csv and p0_a_with_candidate_oracle/candidate_oracle_diagnostic.csv.

## P0-B: source supervision consistency and real match rate

- tools/mixed_p0_b_source_supervision.py offline: for all GN and filtered GC6D training scene caches, collect retained points, active object count, GC6D missing objects, CDF positive bins and width-valid ratios. In addition, sample two evenly spaced training frames per scene for **source-specific graspness arrays**: GN from graspness/, GNTrans from virtual_graspness/, GC6D from graspness_instance/ (configurable). GNTrans shares the **same GraspNet CDF geometry**, but its appearance-side/virtual graspness target may differ; do not count scene geometry as independent.
- forward: instantiate production dataset and CDF adapter per source, restore the checkpoint strictly, evaluate a small scene-diverse sample on a single GPU without optimizer updates. Record batch_grasp_cdf_valid_mask matched actions, matched query fraction, bins>0 among valid actions, and B: CDF/Width/Graspness/DepthReg Loss.
- A source accounting for 36.7% of frames may contribute a very different fraction of valid matched queries. Offline positives and checkpoint-dependent valid-matching rates are *different estimands* and remain separate reports.
- GraspNet uses scene graspness with foreground-rendered+fused-background supervision, GNTrans uses rendered depth with paired geometry, and GC6D historical protocol uses instance graspness and project-generated virtual depth.

Outputs: p0_b_cache/cache_scene_coverage.csv, graspness_frames.csv and summary.json; optional p0_b_forward_point/forward_matched_frames.csv, p0_b_forward_gvar/forward_matched_frames.csv.

## P0-C: depth-target alignment and camera pose semantics

- tools/mixed_p0_c_gc6d_geometry_audit.py uses the **actual workspace_depth mask, its bounding crop, 448x448 nearest resize and depth regression GT-valid mask** (GT in [0.2,1.0] metres). Measures foreground/background joint-valid pixel difference, target-only and sensor-only ratio, weighted MAE, P95 and >50 mm discrepancy. Default 3016-frame flattened stride 20 selects 151 frames.
- The GC6D virtual_scenes target is **project generated** (CAD object + TSDF background), not an official noise-free depth truth. Sensor depth is not a noise-free truth either. These are disagreements, not standalone GT errors.
- Camera hypothesis test computes multi-view object world-pose displacement with supplied scene_camera interpreted as C2W or its inverse. Multiple objects of the same ID are Hungarian-matched within frames. Separately computes angular differences between *existing loader global_film camera vectors* and vectors consistent with the C2W hypothesis.
- Pose consistency cannot independently establish that scene-world +Z is physical up. If the best C2W orientation disagrees with training global_film, **do not fix only inference**: compare an entirely corrected training protocol.

Outputs: p0_c_gc6d_depth_pose/train_crop_depth_frames.csv, camera_film_vectors.csv, pose_consistency_frames.csv, summary.json.

## P0-D: native GC6D test AP for Point / GVAR checkpoints

- tools/mixed_p0_model_checkpoint.py derives enabled designs, grouping/action chunk, checkpoint toggles from saved grasp_design metadata; restores full state strictly. If main and historical source cannot reproduce the checkpoint architecture, stop with a clear error. Never use strict=False or disable missing heads.
- tools/mixed_p0_d_gc6d_cva_eval.py runs official GC6D test scene/frame identities with the existing EconomicGrasp-DPT CVA-CDF decoder, saves Nx17 predictions under six-digit scene/realsense-d435/six-digit img IDs, verifies all expected files, runs GraspClutter6DEval.eval_all() and stores raw evaluator results and a checkpoint-fingerprinted manifest.
- The unlabeled GC6D loader adds camera_pose_vec/camera_gravity_vec/camera_tilt_deg to match the **original training convention**, not an untrained C2W correction.
- collision_thresh=0 disables model-free sensor-cloud postfilter. **But preprocessing still uses sensor depth and label to select workspace/crop.** Thus the model predicts grasp geometry from RGB, but the overall evaluation is **not sensor-free end-to-end RGB-only**; label it RGB grasp prediction with legacy depth-assisted preprocessing.
- GC6D test AP measures its native evaluation distribution. It is not on the same scene difficulty / grasp protocol scale as GraspNet AP, and unseen-object claims require an independent train/test object-union audit.

Outputs: p0_d_gc6d_point/ and p0_d_gc6d_gvar/: full scene/camera/frame Nx17 dumps, gc6d_p0_d_manifest.json, gc6d_official_raw_accuracy.npy when the API returns numeric arrays.

## Acceptance checklist (server execution required)

- [ ] Python compile and unittest for P0-A/D synthetic protocol tests; shell bash -n.
- [ ] A reads all 12 valid official arrays; scene pairing matches checkpoint and per-split scene/frame IDs; scene bootstrap reproducible with seed0.
- [ ] A candidate evaluator records NMS/global-TopK physical labels without claiming official AP.
- [ ] B scanned 100 GN/100 GNTrans (shared cache)/232 GC6D scenes and recorded both micro and scene-average coverage; any actual 1-GPU forward recorded finite valid masks/loss for all 3 sources.
- [ ] C audited 232-scene metadata, 151 training-crop depth samples at stride20, two camera hypotheses, duplicated object categories handled by assignment.
- [ ] D strict checkpoint smoke successful for Point AND GVAR; complete native GC6D test predictions, no missing/mixed-checkpoint dumps, official AP/AP0.4/AP0.8 returned, each protocol fingerprinted.
- [ ] Create experiments/results/TRIPLE_MIX_P0_ABCD_RESULTS.md on server with source commit/checkpoint SHA, exact commands, tables, failures and diagnostic interpretation. No fabricated AP.

## Decisions

If paired bootstrap AP gain is weak and confidence interval includes zero, do not oversell GVAR. If GC6D actual valid query coverage is much lower, prioritize matching/target balance. If C finds pose metadata disagreement or large valid-region depth discrepancies, test a fully consistent *training* correction. If D improves over a comparable held-out GC6D baseline, that supports domain-specific benefit even when GraspNet AP drops; it does not by itself prove universal generalization.
