#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/cache}
TRAIN_ROOT=${TRAIN_ROOT:-${WORK_ROOT}/train}
TEST_ROOT=${TEST_ROOT:-${WORK_ROOT}/test}
SOURCES=${SOURCES:-pred,sensor,rendered,cad_full}
SPLITS=${SPLITS:-test_similar,test_novel}
TEST_GPUS=${TEST_GPUS:-0,1,2,3}
MAX_FRAMES=${MAX_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}
SAVE_PER_QUERY=${SAVE_PER_QUERY:-0}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

IFS=',' read -r -a SOURCE_ARRAY <<< "${SOURCES}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
IFS=',' read -r -a GPU_ARRAY <<< "${TEST_GPUS}"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No TEST_GPUS specified" >&2; exit 2; }
mkdir -p "${TEST_ROOT}"

PIDS=()
ACTIVE=()
wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-P0-TEST][ERROR] ${ACTIVE[$i]} failed" >&2
      failed=1
    fi
  done
  PIDS=(); ACTIVE=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_job() {
  local source="$1" split="$2" gpu="$3"
  local ckpt="${TRAIN_ROOT}/${source}/checkpoint_best.tar"
  local cache="${CACHE_ROOT}/${split}"
  local out="${TEST_ROOT}/${source}/${split}"
  [[ -f "${ckpt}" ]] || { echo "Missing checkpoint: ${ckpt}" >&2; exit 2; }
  [[ -d "${cache}" ]] || { echo "Missing cache: ${cache}" >&2; exit 2; }
  mkdir -p "${out}"
  local args=(
    "${ROOT_DIR}/test_rep_p0_geometry_probe.py"
    --cache_root "${cache}"
    --checkpoint "${ckpt}"
    --output_dir "${out}"
    --device cuda:0
    --max_frames "${MAX_FRAMES}"
    --progress_every "${PROGRESS_EVERY}"
  )
  [[ "${SAVE_PER_QUERY}" == "1" ]] && args+=(--save_per_query)
  echo "[REP-P0-TEST] source=${source} split=${split} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}"     >"${out}/test.log" 2>&1 &
  PIDS+=("$!"); ACTIVE+=("${source}/${split}")
}

slot=0
for raw_source in "${SOURCE_ARRAY[@]}"; do
  source="$(echo "${raw_source}" | xargs)"
  [[ -n "${source}" ]] || continue
  for raw_split in "${SPLIT_ARRAY[@]}"; do
    split="$(echo "${raw_split}" | xargs)"
    [[ -n "${split}" ]] || continue
    launch_job "${source}" "${split}" "${GPU_ARRAY[$slot]}"
    slot=$((slot+1))
    if [[ ${slot} -ge ${#GPU_ARRAY[@]} ]]; then
      wait_wave
      slot=0
    fi
  done
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

"${PYTHON_BIN}" "${ROOT_DIR}/summarize_rep_p0_geometry_sources.py"   --test_root "${TEST_ROOT}"   --output_dir "${TEST_ROOT}"   --sources "${SOURCES}"   --splits "${SPLITS}"

echo "[REP-P0-TEST] completed: ${TEST_ROOT}"
