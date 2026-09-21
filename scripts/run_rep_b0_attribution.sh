#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
REP_A_WORK_ROOT=${REP_A_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness}
REP_B_WORK_ROOT=${REP_B_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_b_hypothesis_image}
CACHE_ROOT=${CACHE_ROOT:-$REP_A_WORK_ROOT/cache}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_b0_attribution}
BASELINE_CKPT=${BASELINE_CKPT:-$REP_B_WORK_ROOT/train/B0/checkpoint_best.pt}
# Reuse existing B0. Set RETRAIN_FULL=1 to retrain all three in this launcher.
RETRAIN_FULL=${RETRAIN_FULL:-0}
GPUS=${GPUS:-0}
PHASES=${PHASES:-train,test,summary}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
EPOCHS=${EPOCHS:-20}
SEED=${SEED:-0}
LR=${LR:-1e-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
MAX_TEST_FRAMES=${MAX_TEST_FRAMES:-0}
RESUME=${RESUME:-1}
source "$ROOT_DIR/scripts/rep_followup_jobs.sh"
parse_gpus
IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a TESTS <<< "$SPLITS"
CONTROLS=(no_image independent_rgb)
if [[ "$RETRAIN_FULL" == 1 ]]; then CONTROLS=(full no_image independent_rgb); BASELINE_CKPT="$WORK_ROOT/train/full/checkpoint_best.pt"; fi
for phase in "${STEPS[@]}"; do
  case "$phase" in
    train)
      slot=0
      for c in "${CONTROLS[@]}"; do
        args=("$ROOT_DIR/train_rep_b0_controls.py" --cache-root "$CACHE_ROOT" --output-dir "$WORK_ROOT/train/$c"
          --control "$c" --epochs "$EPOCHS" --seed "$SEED" --lr "$LR" --grad-accum-steps "$GRAD_ACCUM_STEPS"
          --max-train-frames "$MAX_TRAIN_FRAMES" --max-val-frames "$MAX_VAL_FRAMES")
        [[ -f "$BASELINE_CKPT" && "$RETRAIN_FULL" != 1 ]] && args+=(--reference-checkpoint "$BASELINE_CKPT")
        [[ "$RESUME" == 1 ]] && args+=(--resume)
        launch "B0/train/$c" "${GPU_IDS[$slot]}" "$WORK_ROOT/train/$c/train.log" "${args[@]}"
        slot=$((slot+1)); if ((slot==${#GPU_IDS[@]})); then wait_wave; slot=0; fi
      done
      wait_wave ;;
    test)
      [[ -f "$BASELINE_CKPT" ]] || { echo "Missing original B0: $BASELINE_CKPT" >&2; exit 2; }
      slot=0
      for c in full no_image independent_rgb; do
        ckpt="$WORK_ROOT/train/$c/checkpoint_best.pt"
        [[ "$c" == full ]] && ckpt="$BASELINE_CKPT"
        [[ -f "$ckpt" ]] || { echo "Missing $ckpt" >&2; exit 2; }
        for split in "${TESTS[@]}"; do
          out="$WORK_ROOT/test/$c/$split"
          args=("$ROOT_DIR/test_rep_b0_controls.py" --cache-root "$CACHE_ROOT" --checkpoint "$ckpt"
            --name "$c" --split "$split" --output-dir "$out" --max-frames "$MAX_TEST_FRAMES")
          [[ "$RESUME" == 1 ]] && args+=(--resume)
          launch "B0/test/$c/$split" "${GPU_IDS[$slot]}" "$out/test.log" "${args[@]}"
          slot=$((slot+1)); if ((slot==${#GPU_IDS[@]})); then wait_wave; slot=0; fi
        done
      done
      wait_wave ;;
    summary) "$PYTHON_BIN" "$ROOT_DIR/summarize_rep_followup.py" --kind attribution --root "$WORK_ROOT/test" --splits "$SPLITS" ;;
    *) echo "Unknown phase $phase" >&2; exit 2 ;;
  esac
done
