#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
TRAIN_CACHE_ROOT=${TRAIN_CACHE_ROOT:-${WORK_ROOT}/cache}
VAL_CACHE_ROOT=${VAL_CACHE_ROOT:-${WORK_ROOT}/cache_val}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/feature_ablation_3layer}

FEATURE_MODES=${FEATURE_MODES:-raw_offset,selected_residual,mean_residual,selected_mean,full}
ABLATION_GPUS=${ABLATION_GPUS:-0,3,5,6}
VAL_SCENE_START=${VAL_SCENE_START:-100}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-256}
DROPOUT=${DROPOUT:-0.1}
REG_WEIGHT=${REG_WEIGHT:-1.0}
SIGN_WEIGHT=${SIGN_WEIGHT:-0.5}
LISTWISE_WEIGHT=${LISTWISE_WEIGHT:-0.5}
TARGET_TEMPERATURE=${TARGET_TEMPERATURE:-0.15}
THRESHOLD_MAX=${THRESHOLD_MAX:-0.30}
THRESHOLD_STEPS=${THRESHOLD_STEPS:-31}
SEED=${SEED:-0}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

[[ -d "${TRAIN_CACHE_ROOT}" ]] || { echo "Missing train cache: ${TRAIN_CACHE_ROOT}" >&2; exit 2; }
[[ -d "${VAL_CACHE_ROOT}" ]] || { echo "Missing val cache: ${VAL_CACHE_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a MODES <<< "${FEATURE_MODES}"
IFS=',' read -r -a GPUS <<< "${ABLATION_GPUS}"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "No ABLATION_GPUS" >&2; exit 2; }

PIDS=(); NAMES=()
wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[ABLATE-RUN][ERROR] mode=${NAMES[$i]} failed; inspect ${OUTPUT_ROOT}/${NAMES[$i]}/train.log" >&2
      failed=1
    fi
  done
  PIDS=(); NAMES=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_mode() {
  local mode="$1" gpu="$2" out="${OUTPUT_ROOT}/${mode}"
  mkdir -p "${out}"
  echo "[ABLATE-RUN] mode=${mode} gpu=${gpu} out=${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT_DIR}/train_ray_pairwise_feature_ablation.py" \
    --train_cache_root "${TRAIN_CACHE_ROOT}" \
    --val_cache_root "${VAL_CACHE_ROOT}" \
    --output_dir "${out}" \
    --feature_mode "${mode}" \
    --val_scene_start "${VAL_SCENE_START}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --dropout "${DROPOUT}" \
    --reg_weight "${REG_WEIGHT}" \
    --sign_weight "${SIGN_WEIGHT}" \
    --listwise_weight "${LISTWISE_WEIGHT}" \
    --target_temperature "${TARGET_TEMPERATURE}" \
    --threshold_max "${THRESHOLD_MAX}" \
    --threshold_steps "${THRESHOLD_STEPS}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --max_train_frames "${MAX_TRAIN_FRAMES}" \
    --max_val_frames "${MAX_VAL_FRAMES}" \
    --progress_every "${PROGRESS_EVERY}" \
    >"${out}/train.log" 2>&1 &
  PIDS+=("$!"); NAMES+=("${mode}")
}

slot=0
for mode_raw in "${MODES[@]}"; do
  mode="$(echo "${mode_raw}" | xargs)"
  [[ -n "${mode}" ]] || continue
  launch_mode "${mode}" "${GPUS[$slot]}"
  slot=$((slot+1))
  if [[ ${slot} -ge ${#GPUS[@]} ]]; then
    wait_wave
    slot=0
  fi
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

echo "[ABLATE-RUN] training complete: ${OUTPUT_ROOT}"
find "${OUTPUT_ROOT}" -maxdepth 2 -name best.json -print
