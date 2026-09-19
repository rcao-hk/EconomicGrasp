#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/cache}
TRAIN_ROOT=${TRAIN_ROOT:-${WORK_ROOT}/train}
SOURCES=${SOURCES:-pred,sensor,rendered,cad_full}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}

EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-256}
DROPOUT=${DROPOUT:-0.1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-5}
SEED=${SEED:-0}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -d "${CACHE_ROOT}/train" ]] || { echo "Missing train cache" >&2; exit 2; }
[[ -d "${CACHE_ROOT}/test_seen" ]] || { echo "Missing test_seen validation cache" >&2; exit 2; }
mkdir -p "${TRAIN_ROOT}"

IFS=',' read -r -a SOURCE_ARRAY <<< "${SOURCES}"
IFS=',' read -r -a GPU_ARRAY <<< "${TRAIN_GPUS}"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No TRAIN_GPUS specified" >&2; exit 2; }

PIDS=()
ACTIVE=()
wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-P0-TRAIN][ERROR] ${ACTIVE[$i]} failed; inspect train.log" >&2
      failed=1
    fi
  done
  PIDS=(); ACTIVE=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_source() {
  local source="$1" gpu="$2" out="${TRAIN_ROOT}/${source}"
  mkdir -p "${out}"
  echo "[REP-P0-TRAIN] source=${source} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT_DIR}/train_rep_p0_geometry_probe.py"     --train_cache_root "${CACHE_ROOT}/train"     --val_cache_root "${CACHE_ROOT}/test_seen"     --output_dir "${out}"     --source "${source}"     --epochs "${EPOCHS}"     --learning_rate "${LR}"     --weight_decay "${WEIGHT_DECAY}"     --hidden_dim "${HIDDEN_DIM}"     --dropout "${DROPOUT}"     --grad_accum_steps "${GRAD_ACCUM_STEPS}"     --early_stop_patience "${EARLY_STOP_PATIENCE}"     --seed "${SEED}"     --device cuda:0     --max_train_frames "${MAX_TRAIN_FRAMES}"     --max_val_frames "${MAX_VAL_FRAMES}"     --progress_every "${PROGRESS_EVERY}"     >"${out}/train.log" 2>&1 &
  PIDS+=("$!"); ACTIVE+=("${source}")
}

slot=0
for raw_source in "${SOURCE_ARRAY[@]}"; do
  source="$(echo "${raw_source}" | xargs)"
  [[ -n "${source}" ]] || continue
  launch_source "${source}" "${GPU_ARRAY[$slot]}"
  slot=$((slot+1))
  if [[ ${slot} -ge ${#GPU_ARRAY[@]} ]]; then
    wait_wave
    slot=0
  fi
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

echo "[REP-P0-TRAIN] completed: ${TRAIN_ROOT}"
