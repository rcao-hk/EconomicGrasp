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
EVAL_SHARDS=${EVAL_SHARDS:-1}
EVAL_CHUNK=${EVAL_CHUNK:-128}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
MIN_HOST_FREE_GIB=${MIN_HOST_FREE_GIB:-4}
LABEL_WORKERS=${LABEL_WORKERS:-2}
RESUME=${RESUME:-1}
OVERWRITE=${OVERWRITE:-0}

source "$ROOT_DIR/scripts/rep_followup_jobs.sh"
[[ "$EVAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid EVAL_SHARDS=$EVAL_SHARDS" >&2; exit 2; }
[[ "$LABEL_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid LABEL_WORKERS=$LABEL_WORKERS" >&2; exit 2; }

IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a TESTS <<< "$SPLITS"

for phase in "${STEPS[@]}"; do
  case "$phase" in
    label)
      # Optional: create the exact native/A1-selected labels C1 needs.
      # This operates on SOURCE_ROOT and is resumable. For a cheap first test,
      # point SOURCE_ROOT at the QUERY_LIMIT=64 pilot instead of the full-query run.
      extra=()
      [[ "$RESUME" == 1 ]] && extra+=(--resume)
      for split in "${TESTS[@]}"; do
        for ((s=0;s<EVAL_SHARDS;s++)); do
          launch "rep-c1/label/$split/$s" "" "$WORK_ROOT/logs/label_${split}_${s}.log"             "$ROOT_DIR/evaluate_rep_fullpath.py"             --dataset-root "$DATASET_ROOT" --work-root "$SOURCE_ROOT"             --split "$split" --shard-id "$s" --num-shards "$EVAL_SHARDS"             --eval-chunk "$EVAL_CHUNK" --fc-mode "$FC_MODE" --verify-n "$VERIFY_N"             --label-scope selected --min-host-free-gib "$MIN_HOST_FREE_GIB"             "${extra[@]}"
          if (( ${#PIDS[@]} >= LABEL_WORKERS )); then
            wait_wave
          fi
        done
        wait_wave
      done
      ;;
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
