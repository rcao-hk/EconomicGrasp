#!/usr/bin/env bash
# Source-only helpers. Own process groups, no broad pkill patterns.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1
command -v setsid >/dev/null || { echo 'setsid is required' >&2; exit 2; }
PIDS=(); NAMES=()
cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  if (( ${#PIDS[@]} )); then
    sleep 2
    for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_wave() {
  local i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[FOLLOWUP ERROR] ${NAMES[$i]} failed; see log" >&2
      exit 1
    fi
  done
  PIDS=(); NAMES=()
}
launch() {
  local name="$1" gpu="$2" log="$3"; shift 3
  mkdir -p "$(dirname "$log")"
  echo "[FOLLOWUP] $name GPU=$gpu LOG=$log"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$log" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}
parse_gpus() {
  IFS=',' read -r -a GPU_IDS <<< "$GPUS"
  (( ${#GPU_IDS[@]} )) || { echo 'Empty GPUS' >&2; exit 2; }
  local g
  declare -A used=()
  for g in "${GPU_IDS[@]}"; do
    [[ -n "$g" && "$g" != *' '* && -z "${used[$g]:-}" ]] || { echo "Invalid/duplicate GPU $g" >&2; exit 2; }
    used[$g]=1
  done
}
