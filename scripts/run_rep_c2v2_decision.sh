#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

SOURCE_ROOT=${SOURCE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_formal_joint_ap}
REP_A_WORK_ROOT=${REP_A_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness}
CACHE_ROOT=${CACHE_ROOT:-$REP_A_WORK_ROOT/cache}
C2V2_DIR=${C2V2_DIR:-/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_fullpath_verifier_v2/train}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_decision_diagnostic}

GPUS=${GPUS:-0,1,2}
PHASES=${PHASES:-fit,eval,summary}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
CASES=${CASES:-nominal,bias:-20,bias:20}
QUERY_CHUNK=${QUERY_CHUNK:-128}
SEEN_TRAIN_FRACTION=${SEEN_TRAIN_FRACTION:-0.5}
SPLIT_SEED=${SPLIT_SEED:-2029}
RIDGE=${RIDGE:-0.01}
RIDGE_GAIN_WEIGHT=${RIDGE_GAIN_WEIGHT:-4.0}
THRESHOLD_GRID=${THRESHOLD_GRID:-201}
SEEN_SUBSET=${SEEN_SUBSET:-val}
MAX_FIT_FILES=${MAX_FIT_FILES:-0}
MAX_EVAL_FILES=${MAX_EVAL_FILES:-0}

source "$ROOT_DIR/scripts/rep_followup_jobs.sh"
parse_gpus
IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a TESTS <<< "$SPLITS"

for phase in "${STEPS[@]}"; do
  case "$phase" in
    fit)
      launch "rep-c2v2-decision/fit" "${GPU_IDS[0]}" "$WORK_ROOT/logs/fit.log"         "$ROOT_DIR/fit_rep_c2v2_decision.py"         --source-root "$SOURCE_ROOT"         --cache-root "$CACHE_ROOT"         --c2v2-dir "$C2V2_DIR"         --output-root "$WORK_ROOT/calibration"         --cases "$CASES"         --query-chunk "$QUERY_CHUNK"         --seen-train-fraction "$SEEN_TRAIN_FRACTION"         --split-seed "$SPLIT_SEED"         --ridge "$RIDGE"         --ridge-gain-weight "$RIDGE_GAIN_WEIGHT"         --threshold-grid "$THRESHOLD_GRID"         --max-files "$MAX_FIT_FILES"
      wait_wave
      ;;
    eval)
      [[ -f "$WORK_ROOT/calibration/calibration.json" ]] || {
        echo "Run PHASES=fit first: missing calibration.json" >&2
        exit 2
      }
      slot=0
      for split in "${TESTS[@]}"; do
        out="$WORK_ROOT/eval/$split"
        launch "rep-c2v2-decision/eval/$split" "${GPU_IDS[$slot]}" "$WORK_ROOT/logs/eval_${split}.log"           "$ROOT_DIR/eval_rep_c2v2_decision.py"           --source-root "$SOURCE_ROOT"           --cache-root "$CACHE_ROOT"           --c2v2-dir "$C2V2_DIR"           --calibration "$WORK_ROOT/calibration/calibration.json"           --output-dir "$out"           --split "$split"           --cases "$CASES"           --query-chunk "$QUERY_CHUNK"           --seen-subset "$SEEN_SUBSET"           --max-files "$MAX_EVAL_FILES"
        slot=$((slot+1))
        if ((slot==${#GPU_IDS[@]})); then wait_wave; slot=0; fi
      done
      wait_wave
      ;;
    summary)
      "$PYTHON_BIN" "$ROOT_DIR/summarize_rep_c2v2_decision.py"         --root "$WORK_ROOT"         --splits "$SPLITS"
      ;;
    *)
      echo "Unknown Rep-C2-v2 decision phase: $phase" >&2
      exit 2
      ;;
  esac
done
