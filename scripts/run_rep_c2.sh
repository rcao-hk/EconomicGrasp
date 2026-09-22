#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

CACHE_ROOT=${CACHE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness/cache}
REP_A_WORK_ROOT=${REP_A_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness}
A1_CHECKPOINT=${A1_CHECKPOINT:-$REP_A_WORK_ROOT/train/A1/checkpoint_best.pt}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_c2_verifier}
GPUS=${GPUS:-0}
PHASES=${PHASES:-train,test}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
EPOCHS=${EPOCHS:-12}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
MAX_TEST_FRAMES=${MAX_TEST_FRAMES:-0}
RESUME=${RESUME:-1}

source "$ROOT_DIR/scripts/rep_followup_jobs.sh"
parse_gpus
IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a TESTS <<< "$SPLITS"

for phase in "${STEPS[@]}"; do
  case "$phase" in
    train)
      args=("$ROOT_DIR/train_rep_c2.py"
        --cache-root "$CACHE_ROOT"
        --a1-checkpoint "$A1_CHECKPOINT"
        --output-dir "$WORK_ROOT/train"
        --epochs "$EPOCHS"
        --max-train-frames "$MAX_TRAIN_FRAMES"
        --max-val-frames "$MAX_VAL_FRAMES")
      [[ "$RESUME" == 1 ]] && args+=(--resume)
      launch "rep-c2/train" "${GPU_IDS[0]}" "$WORK_ROOT/train/train.log" "${args[@]}"
      wait_wave
      ;;
    test)
      slot=0
      for split in "${TESTS[@]}"; do
        out="$WORK_ROOT/test/$split"
        launch "rep-c2/test/$split" "${GPU_IDS[$slot]}" "$out/test.log"           "$ROOT_DIR/test_rep_c2.py"           --cache-root "$CACHE_ROOT"           --a1-checkpoint "$A1_CHECKPOINT"           --c2-dir "$WORK_ROOT/train"           --output-dir "$out"           --split "$split"           --max-frames "$MAX_TEST_FRAMES"
        slot=$((slot+1))
        if ((slot==${#GPU_IDS[@]})); then wait_wave; slot=0; fi
      done
      wait_wave
      ;;
    *)
      echo "Unknown Rep-C2 phase: $phase" >&2
      exit 2
      ;;
  esac
done
