#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

SOURCE_ROOT=${SOURCE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_formal_joint_ap}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_c1_oracle_accept}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
CASES=${CASES:-nominal,bias:-20,bias:20}
MODE=${MODE:-joint}
SCORER=${SCORER:-A1}
POLICIES=${POLICIES:-fixed_0,val_selected}
SCORE_SOURCES=${SCORE_SOURCES:-stage1,a1}
TIE_EPS=${TIE_EPS:-1e-7}
PHASES=${PHASES:-analyze}
FRAME_STRIDE=${FRAME_STRIDE:-10}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-4}
RESUME=${RESUME:-1}
OVERWRITE=${OVERWRITE:-0}

IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a TESTS <<< "$SPLITS"

for phase in "${STEPS[@]}"; do
  case "$phase" in
    analyze)
      extra=()
      [[ "$OVERWRITE" == 1 ]] && extra+=(--overwrite)
      "$PYTHON_BIN" "$ROOT_DIR/rep_c1_oracle_acceptance.py"         --source-root "$SOURCE_ROOT"         --output-root "$WORK_ROOT"         --splits "$SPLITS"         --cases "$CASES"         --mode "$MODE"         --scorer "$SCORER"         --policies "$POLICIES"         --score-sources "$SCORE_SOURCES"         --tie-eps "$TIE_EPS"         "${extra[@]}"
      ;;
    official)
      [[ -f "$WORK_ROOT/protocol.json" ]] || {
        echo "Run PHASES=analyze first: missing $WORK_ROOT/protocol.json" >&2
        exit 2
      }
      off=()
      [[ "$RESUME" == 1 ]] && off+=(--resume)
      for split in "${TESTS[@]}"; do
        "$PYTHON_BIN" "$ROOT_DIR/eval_rep_fullpath_official.py"           --dataset-root "$DATASET_ROOT"           --work-root "$WORK_ROOT"           --split "$split"           --frame-stride "$FRAME_STRIDE"           --workers "$OFFICIAL_WORKERS"           "${off[@]}"
      done
      ;;
    *)
      echo "Unknown Rep-C1 phase: $phase" >&2
      exit 2
      ;;
  esac
done
