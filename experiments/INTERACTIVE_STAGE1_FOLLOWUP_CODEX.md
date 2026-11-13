# Interactive Grasp Stage 1 Follow-up — Codex Experiment Plan

Date: 2026-11-13. Branch: `exp/interactive-grasp-followup` (forked from `exp/interactive-grasp-stage1@a1caddd2de2637648730b4bdf2973b6b1607d56c`, **not main**). No production-model, GVAR, loss or official evaluator edits.

## 0. Why this experiment is necessary

Existing 100% GraspNet Original AP (Seen/Similar/Novel, 30 scenes×256 frames per split):

|Model|Seen|Similar|Novel|Mean|
|---|---:|---:|---:|---:|
|Frozen GVAR+MSE|77.4480|57.3712|20.8879|51.9024|
|A independent|74.2709|53.9307|18.5198|48.9071|
|B read-again|73.5289|52.9106|18.7750|48.4048|
|C interaction|73.7672|53.2373|19.0054|48.6699|
|D interact → read|74.1367|53.4944|18.9514|48.8608|

All new models regress by approximately 3 AP pp. D−B is positive but D≯A/C. The 2×2 interaction is positive overall, negative on Novel. Existing results archive provides only aggregate AP/acceptance JSON; **training histories, actual server source diff, validation_last, benchmark, individual AP arrays and D pre-interaction-read quality effects are missing**. Do not assume overfitting, architecture insufficiency or NMS interaction before diagnostics.

**The required next experiments are diagnostic, not another model training run.** They aim to falsify competing explanations:

1. Common refinement/residual update destroys the original strong MSE ordering, even without attention (A).
2. The learned score changes are useful but excessively strong (measure alpha sweep).
3. Interaction enhances the read query, as opposed to merely mixing features (D pre-interaction-read control).
4. Better labelled-subset quality fails to transfer to top-ranked full candidate set (full-validation Q and AP-like replay).

## 1. Implementation

New package `interactive_followup` is **outside** the original `interactive_grasp` directory. This preserves `interactive_grasp.contracts.parameter_source_hash()` for original training/checkpoint/cache provenance.

|Entrypoint|Role|
|---|---|
|`python -m interactive_followup.audit`|Read all A/B/C/D histories/protocols/validation_last/last.pt SHA and acceptance; report missing/changed files and epoch0/final trends; no training|
|`python -m interactive_followup.labels label`|For **stage-1 held-out validation only** (scenes80–99), evaluate all 1024 frozen physical actions with official exact CAD/Dex-Net evaluator; verify the existing sparse 128 labels, store action IDs/assignment/collision/friction/count; shard support|
|`python -m interactive_followup.labels verify`|Require all 160 validation frames and a common evaluator version; sign immutable full-label manifest|
|`python -m interactive_followup.sweep`|For each A/B/C/D and optional D_pre_read, compute frozen logits then `sigmoid(z0 + alpha*(zmodel-z0))` for pre-registered alpha values; report sparse-label MSE/repair/damage, full1024 top-k membership/coverage, optionally exact full-label validation AP-like proxy and paired-scene CI|
|`python -m interactive_followup.confirm`|**Optional only after validation lock**: full-1024-query streaming test inference with alpha chosen from validation-only exact Q proxy, unchanged physical candidates and original sensor collision, official NMS evaluator unchanged|
|`python -m interactive_followup.summary`|Validate all three test-split manifests used the same signed validation lock then call existing official AP summary|
|`scripts/interactive_followup/run.sh`|One stage at a time; immutable old Stage1 results; isolated output roots|

The D_read-control calls the previously implemented `bypass_interaction_query=True` with the **same trained D weights**, so it is a frozen inference intervention, not a fifth trained architecture. A/B/C/D all share the same frozen physical actions and the same 1024 candidates per image.

The validation sweep uses alpha **0, .05, .10, .20, .40, .70, 1**. Alpha0 must use the exact original 17D dump probability, without sigmoid round-trip. All alphas affect the score only; no new width, center, angle or insertion, no manually defined graph/leader prior, no deleted NMS.

## 2. Validation and statistical boundaries

- The existing validation cache has only 128 labelled actions per frame, and the dataset contains 160 frames. These Q labels are insufficient to know whether any particular top-1 or NMS survivor is physically good. Sparse sweep reports labelled-subset Q-MSE and pair repair/damage **separately from** full-pool top1/top50 identity/turnover, and NEVER claims a full-pool AP result.
- The optional full-label stage evaluates the same **unchanged** 1024 actions per validation frame: 163,840 labels. This costs CPU/Dex-Net time, unlike merely rescoring frozen features. The full-label stage verifies previously known labels and action SHA before writing. Do a two-domain smoke and throughput/disk estimate before 160-frame run.
- With full labels, alpha sweep replays **official GraspGroup.nms**, per-object top-10, global top-50 and six-friction score using these exact labels. This is called **AP-like validation proxy**, not official GraspNet test AP. Inspect possible implementation/evaluator disagreement on a small sample before using it to pick alpha.
- Alpha is locked by maximising **validation** domain-balanced, scene-balanced proxy AP; ties favour smaller alpha, and if no positive gain over alpha0 choose **alpha0**. Validation was seen by the frozen base/RARE earlier: it is held out only for the added interaction training. Never use Seen/Similar/Novel official test results to select alpha.
- Scene bootstrap resamples the **20 held-out stage-1 validation scene identities**, keeping Original and GNTrans views paired; it does not estimate training-seed variation. No alpha selected from a partial/smoke run.
- To test, reuse exactly one signed `alpha_full/summary.json` across all three 7680-frame Original GraspNet splits, with no test quality labels. Report whether alpha0 was chosen: it is a negative result, not an excuse to search for a better test alpha.
- If original server source hash differs from checked-out GitHub code, audit the actual server adapter/diff first. Both sweep and test support explicit `--allow-code-drift` solely to document intentional compatibility; it is **not** a license to skip native decoder/parity or checkpoint SHA checks.

## 3. Worktree and source audit (do this FIRST)

```bash
git fetch origin exp/interactive-grasp-followup
git worktree add /home/robotarm/EconomicGrasp-interactive-followup \
  -b codex/interactive-followup-run origin/exp/interactive-grasp-followup
cd /home/robotarm/EconomicGrasp-interactive-followup

export STAGE1_ROOT=/data/robotarm/result/grasp/rgbgrasp/interactive_stage1_run01
export FOLLOWUP_ROOT=/data/robotarm/result/grasp/rgbgrasp/interactive_stage1_followup_run01
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export DEVICE=cuda:0

bash scripts/interactive_followup/run.sh check
bash scripts/interactive_followup/run.sh audit
```

`STAGE1_ROOT` is a **placeholder**: locate the actual output folder from previous `FINAL_ACCEPTANCE.json`, stage1 execution logs, and GitHub branch, and inspect its `plan.json`, `source.json`, `validation_index.json`, `models/{A,B,C,D}` and existing `test` roots. The audit reports missing `history.json`, `validation_last.json`, `protocol.json`, `complete.json`, `last.pt`; do not auto-create replacement files or silently accept different historical weights. Keep source SHA and actual server diff in final report.

## 4. Validation-only score diagnostics

First run on the **existing sparse 128 labels** without invoking the expensive evaluator:

```bash
bash scripts/interactive_followup/run.sh sweep-sparse
```

No alpha is selected here. Output includes per-model/alpha residual magnitude, full-context top1 changes and top5/top50 overlap, labelled-subset Q-MSE, tie-excluded pair repair/damage and D_pre_read control. Interpret labelled top1 quality only where labels exist; missing labels must NEVER be assumed negative.

Then optional full-label pilot and complete labelling (do not compete with the running graph oracle for evaluator CPU resources):

```bash
bash scripts/interactive_followup/run.sh label-smoke
# verify one frame from each domain before full labelling; use --shard / --shards when needed
bash scripts/interactive_followup/run.sh labels
bash scripts/interactive_followup/run.sh verify-labels
bash scripts/interactive_followup/run.sh sweep-full
```

`--full-labels-root` is disallowed for test index; code asserts scenes <100 and partition == validation. `sweep-full` refuses to lock alpha unless all 160 frames and every action have labels and matching source/action IDs.

If test/adapter code hash differs, locate and document the actual running source. Only after verifying that difference is intentional set `ALLOW_CODE_DRIFT=1`; record exact repo commit and source file hashes. For missing/incompatible source checkpoints, STOP rather than using `strict=False` or changing checkpoint metadata.

## 5. Optional locked-alpha test (after valid full-label sweep)

```bash
export REFERENCE_ROOT=/ABS/path/to/unchanged_MSE_pre-NMS_full_dumps
SPLITS='test_seen' bash scripts/interactive_followup/run.sh infer
SPLITS='test_seen' bash scripts/interactive_followup/run.sh eval
# Then the other two splits, with their correct source-reference roots.
SPLITS='test_similar test_novel' bash scripts/interactive_followup/run.sh infer
SPLITS='test_similar test_novel' bash scripts/interactive_followup/run.sh eval
bash scripts/interactive_followup/run.sh summarize
```

The inference runner insists on the same frozen MSE source, immutable 1024 physical IDs, pre-sensor-filter full-context inputs, and original MSE dump parity ≤1e-6. Every method uses only the signed alpha selected on held-out validation. Full official `eval.py` still produces `[30,256,50,6]` arrays for each split. New output directories are required. No test-label based adjustment.

The `REFERENCE_ROOT` may need to change per split; use separate invocations, not a single mixed root whose paths do not exist.

## 6. Decision criteria and final deliverables

- If alpha0 is best for most methods and validation Q/membership worsens for positive alpha, **the residual correction direction is suspect**. Stop adding leader feedback, graph losses, extra read layers and complexity; audit training distribution and score calibration first.
- If small positive alpha improves full-validation AP proxy, but alpha1 is worse, **residual magnitude** may be the issue. Freeze alpha and check same choice on test, but do not claim generality from one training seed.
- If D−D_pre_read loses performance on exact full-label quality after removing the updated read query, this supports the effect of interaction on evidence interpretation, conditional on trained weights. A logit-change-only result is insufficient. It is still an inference intervention, not a trained causal ablation.
- If D−B or D−C does not exceed the same-alpha controls, do not claim a validated Interact→Read mechanism.
- If val Q improves but validation proxy/replay AP degrades, focus on high-rank selection and the existing graph oracle's finding as a **diagnostic**; do not inject graph into the new model.
- Do not use test Novel to tune alpha, report only the locked values. If all alphas choose zero, a no-update selection is a legitimate negative result.

Deliver `audit.json`, `alpha_sparse/{summary.json,per_frame.csv,REPORT_ZH.md}`, optional `full_quality` signed exact-action cache and `alpha_full/summary.json` plus scene bootstrap, and optional official test AP outputs with `locked_alpha_ap.json`. Keep a `RESULTS_CN_FOLLOWUP.md` with findings/unknowns, training-source identity and next-step decision. Do not claim execution of 117 real CAD or official AP until logs and full artifacts are verified.