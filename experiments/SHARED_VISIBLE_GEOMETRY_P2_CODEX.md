# P2: P1 Native-relative Calibration and Shared-weight Geometry Source Control

Branch: `exp/shared-visible-geometry-multibatch` (same branch). The scripts extend the completed 2026-11-16 P0–P1 study without modifying its source weights, cached physical actions, labels, or official evaluator.

## Research questions

**P2-A: P1 reader native-relative failure modes.** Re-use the **new matched** P1 Predicted and Clean-Visible `best.pt` models, not the older full-field `predicted`/`h2` weights studied by P0. Apply the same `metrics_for_family` / `summarize` definitions on the same 160-frame validation cache and the same 5,120 action families. Report candidate-vs-native bias, calibrated relative improvements, rescue/harm, native count=6 protection, opportunities/missed opportunities, offset, score margin, and visible/unknown state. Pair clean/predicted observations by scene, domain, frame and family before contrasts. Scene-bootstrap estimates are descriptive, conditional on previously selected weights; no new alpha/margin tuning.

**P2-B: Shared-reader geometry source test.** Train **ONE** `PredictedGeometryControl.point_reader + quality` whose weights are shared across Predicted and Clean-Visible analytic 75-point geometry. On each of the **same** training images, calculate both sources, using `L=0.5*L_pred+0.5*L_clean` (same `10*MSE+0.5*within-family pair` loss on each source), one optimizer update per real batch, not twice the number of image updates. No model receives the source name as a feature. Evaluate the identical frozen reader on both sources and pair per-family outputs on the same validation bank. Checkpoint selection uses the **mean** of both validation selected-Q gains; it never preferentially optimizes one source. This is a distribution-robustness / fixed-weight source diagnostic and differs from the two independently trained P1 readers. It does **not** make the Clean teacher available to deployment.

## Locked protocol

- P1 cache and checkpoint identities must match (`val_index` fingerprint, teacher config SHA, train/validation source, plan, offsets), all files are read-only. Original P1 renderer provenance is inherited from the signed source cache.
- `train`: 2,560 paired RGB-domain frames (scene 0–79), `validation`: 160 paired-domain frames (scene 80–99); 16 vs 32 families/frame, seven actual actions/family, 75 geometry query sites/action.
- True B=4, accum=1, hidden=64, seed=0, AdamW lr=1e-4, wd=1e-4, 20 epochs (12,800 updates), per-image family loss. No new exact-action labels or GT-derived inputs. The Clean geometry is privileged and allowed **only in existing offline P1 train/validation**; no official test/deployment path.
- Independent P1 arms have 13,313 trainable parameters each; shared arm has one matching 13,313-parameter copy. It sees both evidence sources, therefore the input evidence budget is larger than either isolated P1 arm; the images/quality labels/update count remain the same. Treat this as a fixed-weight diagnostic, not a fully equivalent data-exposure ablation.
- Reject partial cached inputs in formal mode; `--smoke` is for synthetic tests only. Validate physical family identity via the already signed P1 cache and paired IDs. No test use, no manual scoring threshold tuning, no overwrite of old results.
- Crucial: same trained weights does **not** remove all source-distribution effects. A positive Clean-vs-Pred effect in this control supports geometry information value under a shared reader, but is not a causal estimate of an ideal deployment system or proof that student geometry can realize Clean quality.

## Run on 10.30.7.117

```bash
cd /home/robotarm/EconomicGrasp-shared-geometry-batch
git fetch origin exp/shared-visible-geometry-multibatch
git pull --ff-only origin exp/shared-visible-geometry-multibatch

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_fp16_run01
export P0P1_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p0p1_run01
export P2_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p2_run01
export DEVICE=cuda:0
export BATCH_SIZE=4 ACCUMULATE=1 VAL_BATCH_SIZE=4

bash scripts/shared_geometry/run_p2.sh check
bash scripts/shared_geometry/run_p2.sh p2-diagnose
bash scripts/shared_geometry/run_p2.sh p2-shared-train
bash scripts/shared_geometry/run_p2.sh p2-shared-compare
```

For interrupted training only, re-run `RESUME=1 bash scripts/shared_geometry/run_p2.sh p2-shared-train` with unchanged protocol. Re-running a completed diagnostics output refuses overwrite. Use a new `P2_ROOT` for alternative seeds or settings; do not mutate P1 history.

## Artifacts and required interpretation

- `p1_native_relative_diagnostics.json`: P1 matched-reader native-relative errors, native-full protection, opportunity/missed opportunity, offsets/margin, per-family records and paired scene confidence intervals. Distinguish this from old P0 analysis of five **other** models.
- `shared_reader/{protocol.json,history.json,last.pt,best.pt,selection.json}`: one shared checkpoint trained on two sources and selected by balanced average gain. If its averaged gain is negative, `admitted=false` rather than suppressing negative numbers.
- `shared_fixed_weight_comparison.json`: same shared reader evaluated on both geometry sources, family-level results and scene-paired contrasts. Privileged info never enters inference.
- The headline metrics are validation `Q=count/6`, NOT GraspNet AP or actual robot success. The same validation scenes were used for earlier research/model selection; CIs are exploratory and do not reflect seed variance.
- **Decision logic:** if P1 Clean reduces native-relative overestimation and full-native harm, better geometric evidence explains part of improvement; if shared weights show the same source effect, it is less dependent on independently trained readers. If both sources retain serious relative bias, investigate decision supervision and selection-noise amplification before adding a new gate. Do not treat any outcome as proof that a certain loss must improve AP.

## Verification and current status

The new module's synthetic CPU tests cover P1 cache/weights identity, per-family P0 equivalence, same-bank source pairing, true B batching, equal-weight two-source training, resume, no overwrites and signed outputs. Run original `tests.test_shared_geometry_p0_p1` plus new `tests.test_shared_geometry_p2` before real execution. Formal 117 training and validation are NOT performed by submitting the code. Record real JSON/weights after execution rather than pre-populating results.