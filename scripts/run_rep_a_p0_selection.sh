#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/cache}
TRAIN_ROOT=${TRAIN_ROOT:-${WORK_ROOT}/train}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/diagnostics/rep_a_p0_selection}
GPUS=${GPUS:-0,1}
SPLITS=${SPLITS:-test_similar,test_novel}
VARIANTS=${VARIANTS:-A0,A1,A2,A3}
REFERENCE_VARIANT=${REFERENCE_VARIANT:-A0}
FIXED_MARGINS=${FIXED_MARGINS:-0,0.1}
TEST_CASES=${TEST_CASES:-nominal,bias:-5,bias:5,bias:-10,bias:10,bias:-20,bias:20,bias:-40,bias:40,scale:-0.02,scale:0.02,scale:-0.05,scale:0.05,smooth:5,smooth:10,smooth:20,edge:2}
TEST_SEED=${TEST_SEED:-2026}
MAX_FRAMES=${MAX_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-50}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1
command -v setsid >/dev/null || { echo "setsid is required" >&2; exit 2; }
[[ -f "$CACHE_ROOT/reader_init.pt" ]] || { echo "Missing Rep-A cache: $CACHE_ROOT" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
IFS=',' read -r -a SPLIT_IDS <<< "$SPLITS"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "No GPUs" >&2; exit 2; }
PIDS=(); NAMES=()
cleanup(){ local p; for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 1; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
wait_wave(){ local i failed=0; for i in "${!PIDS[@]}"; do if ! wait "${PIDS[$i]}"; then echo "[REP-A-P0 ERROR] ${NAMES[$i]}" >&2; failed=1; fi; done; PIDS=(); NAMES=(); [[ $failed -eq 0 ]]; }

slot=0
for raw_split in "${SPLIT_IDS[@]}"; do
  split="$(echo "$raw_split" | xargs)"
  out="$OUTPUT_ROOT/$split"; mkdir -p "$out"
  gpu="${GPU_IDS[$slot]}"
  echo "[REP-A-P0] split=$split gpu=$gpu"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$ROOT_DIR/diagnose_rep_a_p0_selection.py" \
    --cache-root "$CACHE_ROOT" --train-root "$TRAIN_ROOT" --output-dir "$out" \
    --split "$split" --variants "$VARIANTS" --reference-variant "$REFERENCE_VARIANT" \
    --fixed-margins "$FIXED_MARGINS" "--cases=$TEST_CASES" --seed "$TEST_SEED" \
    --max-frames "$MAX_FRAMES" --max-val-frames "$MAX_VAL_FRAMES" \
    --progress-every "$PROGRESS_EVERY" >"$out/p0.log" 2>&1 &
  PIDS+=("$!"); NAMES+=("$split")
  slot=$((slot+1))
  if ((slot == ${#GPU_IDS[@]})); then wait_wave; slot=0; fi
done
(( ${#PIDS[@]} == 0 )) || wait_wave
trap - EXIT INT TERM
echo "[REP-A-P0] complete: $OUTPUT_ROOT"
