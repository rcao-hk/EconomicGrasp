# P0 Distribution-Matched Diagnostic + P1 Physical Evidence Decodability

**2026-11-15 · Continuation on the SAME branch:** `exp/quality-strata-conflict-supervision`.

**This is an implementation/work plan, not a claim that new 117 CUDA/CAD training or GraspNet AP already ran.** Do not update main or overwrite the accepted original 4-arm, 3-seed or Expanded-support results.

## 0. Scientific motivation and what changed

Source: `Expanded-support_DETAILED_ANALYSIS_CN.md` (COMPLETE_ACCEPTED, 2026-11-15).

Budget-matched Old64 vs Expanded32-old+32-new, 3 head seeds, fixed native depth:
- Seen: −0.475879 pp; Similar: −0.192947 pp; Novel: **−0.182761 pp** (3/3 seeds negative).
- On identical validation actions: MSE 0.129538→0.129571 (flat), unequal-Q pair agreement 0.767287→0.762036 (worse), Top10 exact Q 0.728333→0.723056.
- Despite new high-score zero-quality actions, Expanded training distribution had **fewer** zeros (43.64%→38.62%) and higher sensor-valid share (85.12%→92.53%).
- The old U∪D cache plus 25,600 additional native-depth training labels now exists. Merely changing sampling, pair loss or scalar ranker has not improved Novel.

Questions:
1. **P0**: on identical actions, precisely which native-score/quality/failure-source/NMS-role strata change score or pair orientation? This is a diagnostic, not a new training arm.
2. **P1**: is physical quality/failure type decodable from pre-reader features, post-reader features, role-attended gripper-volume appearance, raw predicted-depth geometry, or only the 64D final score branch? Same actions/labels and **matched parameter count** for all evidence modes.

The frozen base is historic Point RayRoPE+GVAR36, MIX50 e15, SHA:
`2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`.

This is NOT a three-independent-backbone experiment. The base trained on train scenes 0..99; scorer train 0..79, validation 80..89, audit 90..99 (previously inspected). Official test Seen/Similar/Novel 100..189 is used only AFTER scorer freeze, and has been repeatedly inspected in prior exploratory rounds.

## 1. Modules

- `diagnostics/evidence_decodability/core.py` — same-capacity Quality or four-way Failure probe; strictly aligned [N,17] physical-action identity, six friction labels.
- `capture.py` — attach temporary **read-only** hooks to the REAL `rayrope_gvar.reader.GripperVolumeReader`; intercept pre-reader Q×A×D query features, post-reader outputs and MHA role attention/token data, capture predicted-depth relative geometry from the existing pure-PyTorch geometry function; restore all hooks/patches after each frozen forward. No new model parameters or state keys.
- `cache.py` — 400 accepted native-depth train frames: old U∪D queried actions + 64 new support-audit actions, strict physical-ID/logit/score-feature parity, source SHA and resume. Optional reference GT/CAD depth is *explicitly declared, pixel-aligned and train-only*.
- `p0_match.py` — Old/Expanded quality heads on identical native-executed actions, conditional old/new/quality/native-score/sensor-valid/NMS-role/failure diagnostics, pair inversions.
- `train.py` — independent fixed10-epoch, 64-actions-per-frame, same-data, same-capacity trained **quality** and **failure** probes; frozen GVAR, 3 train-head seeds, validation 80..89.
- `infer.py` — native-depth-only score replacement for all 1024 Original test queries; original sensor collision, GraspNet NMS/Top10-object/Top50 and official evaluation unchanged. Reject failure classifiers and privileged reference.
- `compare.py` — paired split/scene AP with conditional CIs; seed variability separate.
- `scripts/evidence_p0_p1/run.sh` — staged P0/P1 commands.
- `tests/test_evidence_decodability.py` — synthetic CPU unit tests; must run against current checkout before CUDA work.

## 2. P1 evidence semantics — do not relabel role tokens as ground-truth contacts

The actual frozen GVAR `volume_rel` reader uses 36 projected probe positions, six each for:
`left_contact`, `right_contact`, `closing`, `finger_body`, `palm`, `approach`.

Captured groups:
- `score64`: original post-GVAR CDF score branch 64D feature.
- `pre_reader`: reader input action queries before GVAR evidence update.
- `post_reader`: queries after GVAR evidence update, before the shared adapter's finishing FFN.
- `reader_delta`: post−pre.
- `role_tokens`: six attention-weighted 64D role token means + attention masses/valid ratios. Tokens include projected appearance, learned role embeddings, action metadata and geometry; **not** measured contact forces, not isolated pure RGB.
- `pred_geometry`: six-by-six means of reader's local surfaceXYZ/dz/known/valid features, computed from **RGB predicted depth** only.
- `score_plus_roles`, `score_plus_geometry`, `all`: ablations with same input projection dimensions, inactive groups masked to zero. Same number of trainable parameters within task for all modes.
- `reference_geometry` (optional): the same predicted-depth relative geometry calculation re-evaluated on an independently provided, pixel-aligned CAD/reference depth raster. This is a privileged **offline train/validation evidence ceiling**, prohibited during test inference, and cannot be called an RGB-only method.

CAUTION: The reader geometry treats missing/occluded surfaces as unknown, **not** as confirmed free space. Collision/empty/contact failure classification from exact CAD labels is an offline supervision task, not an ability to sense true contacts from a single image.

Quality task: regress mean six-friction success in [0,1] with MSE of sigmoid residual score. Failure task: 4 mutually exclusive labels: success, pure collision, empty (if no collision), other zero-quality. Separate probes, equal architecture per task; failure logits never enter the deployable quality selection.

## 3. Inputs (do not invent paths)

Create a new worktree on the same branch; inspect AGENTS.md/git status and server resources, never kill other jobs:

```bash
git fetch origin exp/quality-strata-conflict-supervision
git worktree add /home/robotarm/EconomicGrasp-evidence-p0p1 \
  -b codex/evidence-p0p1-exec origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-evidence-p0p1
git status --short
git worktree list

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/evidence_p0p1_20261115
export ACCEPTED_PLAN=/ABS/quality_next_20261114/plan.json
export SUPPORT_ROOT=/ABS/quality_followup/p1_support/cache
export SUPPORT_SUMMARY=/ABS/quality_followup/p1_support/summary/summary.json
export BOUND_CONFIG=/ABS/historical_mix50_gvar_bound.json
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/ABS/GN-Trans
export DEVICE=cuda:0
export FACTORY=rare_grasp.runtime:Runtime

bash scripts/evidence_p0_p1/run.sh check
```

The data must be the **accepted** 800-frame native trace plan, the **accepted** 400-frame exact new support-label audit, and the base GVAR checkpoint bound by SHA. If observed source keys differ, adapt with an explicit, tested schema adapter; do not silently fill dummy values, apply strict=False or edit checkpoint metadata.

## 4. Evidence caching: first smoke, then complete 400 frames

Before training verify one frozen forward has exactly Q=1024 × A=12 × D=4 reader actions, GVAR volume_rel with 64D reader tokens, and the captured role chunks concatenate in exact flattened Q*A*D order.

Run independently named smoke so it cannot be silently passed into formal training:

```bash
EVIDENCE_ROOT="$RUN_ROOT/smoke_evidence" CACHE_LIMIT=2 \
  bash scripts/evidence_p0_p1/run.sh cache
```

Check:
- hook activation count and output dimensions;
- native selected angle, native depth, width/center/R and physical SHA equal accepted cached labels;
- the 64D score feature and CDF original base logit equal cached values within 1e-6;
- all labels are from scenes below 100;
- runtime/capture source SHA, support artifact SHA and failure labels align;
- hook cleanup restores model state and GVAR geometry function exactly.

Then cache full fixed 400 frames:

```bash
bash scripts/evidence_p0_p1/run.sh cache
# Same immutable contract only after interruption:
RESUME=1 bash scripts/evidence_p0_p1/run.sh cache
```

Every frame stores ~96 old-native actions plus 64 new-native actions (exact count determined by old cache) and immutable, label-aligned six-role evidence; new CAD labels are **never recomputed**.

Optional privileged reference must supply explicit aligned depth image paths:
`REFERENCE_ROOT/domain/scene_XXXX/realsense/NNNN.npy`; shape must equal the model's GVAR depth map H×W, units `m` or `mm` are declared via `REFERENCE_UNIT`. Do not infer units or automatically resize. Keep its output separate and do not include GT/reference tensors in deployable inference.

## 5. P0 matched error analysis, no new fit

Prepare `OLD_EXPANDED_HEADS` JSON of the three fixed seed pairs, with exact full-file SHA256:

```json
{
  "20261113": {
    "old": {"file": "/ABS/old/last.pt", "sha256": "ACTUAL_64HEX_SHA256"},
    "expanded": {"file": "/ABS/expanded/last.pt", "sha256": "ACTUAL_64HEX_SHA256"}
  },
  "20261114": {
    "old": {"file": "/ABS/old2/last.pt", "sha256": "ACTUAL_64HEX_SHA256"},
    "expanded": {"file": "/ABS/expanded2/last.pt", "sha256": "ACTUAL_64HEX_SHA256"}
  },
  "20261115": {
    "old": {"file": "/ABS/old3/last.pt", "sha256": "ACTUAL_64HEX_SHA256"},
    "expanded": {"file": "/ABS/expanded3/last.pt", "sha256": "ACTUAL_64HEX_SHA256"}
  }
}
```

Existing weights must be the audited 5,633-param `score_feature_geometry` quality head, same base SHA and exact state dict. No rebuilding or guessing missing weights. If Expanded checkpoint metadata differs from existing QualityProbe contracts, **stop and implement/test an explicit adapter** before analyzing.

```bash
export OLD_EXPANDED_HEADS=/ABS/old_vs_expanded_head_map.json
bash scripts/evidence_p0_p1/run.sh p0
```

Outputs: `p0_distribution_matched/per_frame.csv`, `per_stratum_frame.csv`, `strata_summary.csv`, `summary.json`.
Interpret only sufficiently populated cells. Report score shifts/error by old vs new, source native-score quartile, actual Q/failure, sensor-valid and NMS-role. High-score q0-vs-positive inversions are candidate **risk indicators**, not certified actual suppressor outcomes. Training/validation/audit groups separately reported. Existing test Novel has been viewed repeatedly; do not use it to select hyperparameters.

## 6. P1 matched physical decodability probes

Freeze backbone/GVAR/CDF entirely. For each of the same 400 frames, fix exactly **32 old-native + 32 new-native actions**; select without reference to CAD Q using stable SHA-derived frame RNG, identical actions across all modes and seeds.

Train 0–79 (320 two-domain frames), validate 80–89 (40 frames), separately report previously inspected 90–99 (40 frames). Batch4, 10epochs → 800 optimizer steps per trained head. Three preregistered scorer initialization seeds. Same AdamW optimizer/learning rate/parameter count across evidence modes within task. The primary checkpoint is always **last epoch10**; do not select by Novel AP.

```bash
# Quality and four-way physical failure probes, matched 10 epochs.
TASKS="quality failure" \
MODES="score64 pre_reader post_reader reader_delta role_tokens pred_geometry all" \
bash scripts/evidence_p0_p1/run.sh p1-fit
```

Primary report:
- quality MSE, unequal-Q pair agreement, common-pool Top10 true Q, Top10 q0 fraction;
- failure per-class recall/macro-F1 and confusion matrix;
- trends split by old/new and native high score;
- same-head-seed paired scene analysis, not IID action CIs;
- interpret role tokens as mixed learned evidence, not ground-truth contact readings.

If privileged CAD/reference aligned raster is supplied, train `reference_geometry` as a **separate offline oracle** with same train/validation scenes, but never include in deployable test or compare it as a standard RGB-only model.

## 7. Official inference and AP

Only train-head `task=quality` models whose evidence mode is deployable can be tested on Original test scenes 100–189. Do not supply test quality labels or reference-depth oracle.

At test inference, read frozen GVAR once and score **only each query's native executed depth** using action- and angle-aligned evidence. All unexecuted depth bins are left at native score, and `rank_executed` selects the original depth/width/center/R unchanged. Run the original sensor collision and official GraspNet NMS / per-object 10 / global top-50.

```bash
# Matched frozen native baseline:
LABEL=native bash scripts/evidence_p0_p1/run.sh p1-infer
LABEL=native bash scripts/evidence_p0_p1/run.sh p1-eval

# One learned evidence head, fixed last epoch10:
export LABEL=post_reader_20261113
export P1_WEIGHTS="$RUN_ROOT/p1_heads/quality/20261113/post_reader/last.pt"
bash scripts/evidence_p0_p1/run.sh p1-infer
bash scripts/evidence_p0_p1/run.sh p1-eval
```

Formal 10% eval expects each split 780 frames and official AP array [30,26,50,6]; optional full 100% uses `INTERVAL=1` in a **new** result directory. All inference dumps have per-file SHA; eval refuses partial/smoke. Original sensor collision is **outside** the RGB model; do not call the entire stack sensor-depth-free.

To aggregate modes and seeds, create `P1_RUNS_JSON` from fully evaluated roots:
`seed -> native/mode -> test_seen,test_similar,test_novel -> inference path`.
Native baseline can be shared across seeds only with identical base, sampling seed, geometry and collision protocol.

```bash
export P1_RUNS_JSON=/ABS/p1_inference_runs.json
bash scripts/evidence_p0_p1/run.sh p1-compare
```

The comparison uses paired scenes stratified by Seen/Similar/Novel; training-head seed variability is reported separately and not conflated with independent backbone-seed uncertainty.

## 8. Stop conditions and scientific interpretation

1. **No native identity parity:** stop and repair alignment; never assign a cached CAD Q to a different 17D executed grasp.
2. **No valid 36-probe reader capture:** stop; no synthetic “contact evidence” substituted.
3. **P0 Old/Expanded weights have different source base/architecture:** stop, explicitly map only verified compatible checkpoints.
4. **No complete400 native labels:** do not train the formal P1 probes on a smoke cache.
5. **P1 failure task:** no official grasp AP; only classifier diagnostics on train/validation/reused-audit labels.
6. **Reference geometry arm:** offline train-scene quality/failure upper-bound comparison only, no official test.
7. **P1 post-reader > pre-reader and score64 on matched validation failure classes:** supports reader adding discriminative information; if all modes fail, does **not** prove RGB contains no physical evidence.
8. **Raw relative geometry better than compressed score feature:** suggests evidence-to-quality compression/readout bottleneck; requires a causal ablation or retrained model for design claim.
9. **Strong train/validation classification but Novel AP still degrades:** feature/decode/domain transfer remains unresolved; do not tune on reused Novel.
10. **Do not add GNN, new pointwise/pair weights or full backbone retrain as automatic next stage.** Diagnose physical failure type and geometry error support first.
11. Run CPU synthetic test and 117 CUDA single-frame smoke before claiming complete. This ChatGPT turn did not run any actual 117 CUDA/CAD experiment.

Final Codex deliverables: immutable frame cache manifest, fully verified 400-frame evidence NPZ, P0 matched-strata report, all P1 train histories + fixed last checkpoints, failure-type confusion, official per-split AP arrays and paired scene report, exact sources/checkpoints SHA, per-stage execution/validation logs, and concise Chinese scientific conclusions distinguishing facts from hypotheses.
