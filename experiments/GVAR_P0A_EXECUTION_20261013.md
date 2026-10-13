# GVAR P0-A — 2026-10-13 targeted execution

Scope: **P0-A only** on `exp/gripper-volume-action-reader`. RayRoPE U3 is owned by a separate experiment; matched RGB P0-C and inverse-hybrid P0-B are explicitly out of scope.

## Code already available on this branch

- `models/gripper_volume_reader.py`: `set_counterfactual(r0...r4)` on a frozen `volume_rel` reader. R0 dynamic feature/metadata/mask, R1 fixed feature only, R2 fixed position metadata only, R3 fixed attention validity mask only, R4 fixed feature+metadata+mask. Predicted relative geometry remains dynamic in all cases.
- `inference_gvar_p0.py` and `scripts/run_gvar_p0_eval.sh`: identical frozen checkpoint and per-frame sampler across modes, Original RealSense 10% Seen/Similar/Novel, standard sensor-cloud collision, official AP.
- `analyze_gvar_p0a.py`: scene-paired bootstrap, friction-specific and prefix metrics; `audit_gvar_p0_support.py`: AP-blind Novel probe projection and mask-conflict audit.
- `verify_gvar_p0a_checkpoint.py`: **new reader-only CPU checkpoint preflight**, strict reader weight loading, R0 replay, R4 vs same-weights Hybrid equivalence, finite R1–R4, and immutable reader state. This is **not** a full-model/server parity test.

## Runbook (on the authorized training server, not on GitHub)

1. Inspect `AGENTS.md` (if present), current GPU/CPU/memory/disk occupancy, working trees and uncommitted work. Do not alter other experiments or `main`.
2. Compare current GitHub branch against the historically validated training source SHA `59336c461ad40f5e6d8d759c08e1347b877435d3` on the server, and audit required dataset/width/DDP fixes. That SHA is not currently resolvable in the GitHub history. Strict-load frozen `$ROOT/train/volume_rel/checkpoint_epoch_019.tar` and check **full** R0 depth, center, view, CDF, width and decode parity on identical real input with the validated server model. Do not proceed to official AP on mismatch.
3. Run the reader-only preflight:

```bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
python verify_gvar_p0a_checkpoint.py \
  --checkpoint "$ROOT/train/volume_rel/checkpoint_epoch_019.tar"
```

4. Use unique result roots (examples below are placeholders; choose non-existing directories), and only **KIND=p0a**:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet
GPUS=0 KIND=p0a SPLITS=test_seen MAX_BATCHES=2 RUN_EVAL=0 \
  OUT_ROOT="$ROOT/analysis/p0a_smoke_UNIQUE" bash scripts/run_gvar_p0_eval.sh

GPUS=0,1,2 KIND=p0a BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
  OUT_ROOT="$ROOT/analysis/p0a_formal_UNIQUE" bash scripts/run_gvar_p0_eval.sh
```

GPU IDs are *examples only*, never assume availability. The formal run covers 5 modes × 3 splits × 780 frames. Save all dumps, protocols, actual SHA256, smoke logs, official AP arrays and paired-scene confidence intervals. Optionally run `audit_gvar_p0_support.py` with the previously frozen AP-blind 120-frame selection and associated depth sidecars, not with outcome-picked frames.

## Interpretation / acceptance

A test-time R1–R4 drop measures **frozen-model sensitivity**, not the effect of re-training a different architecture or pure fixed-candidate re-ranking. A GVAR R4 vs historical Hybrid AP difference includes different learned weights and must not be used as a one-change causal contrast.

No real-data P0-A AP or server R0 full-model parity is claimed in this commit. The branch only contains code/CPU preflight support; server execution and audited AP remain pending.
