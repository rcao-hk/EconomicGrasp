#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
TRAIN_CACHE_ROOT=${TRAIN_CACHE_ROOT:-${WORK_ROOT}/cache}
VAL_CACHE_ROOT=${VAL_CACHE_ROOT:-${WORK_ROOT}/cache_val}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/relational_rep_ablation}

REPRESENTATION_MODES=${REPRESENTATION_MODES:-G0_current_full,G1_no_abs_raw,G2_residual_only,G3_residual_profile,G4_mean_profile}
GPUS=${GPUS:-0,1,2,3,4}
VAL_SCENE_START=${VAL_SCENE_START:-100}

EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
D_MODEL=${D_MODEL:-128}
NHEAD=${NHEAD:-4}
NUM_LAYERS=${NUM_LAYERS:-2}
FF_DIM=${FF_DIM:-256}
DROPOUT=${DROPOUT:-0.1}
GATE_WEIGHT=${GATE_WEIGHT:-1.0}
SELECTOR_WEIGHT=${SELECTOR_WEIGHT:-1.0}
DELTA_WEIGHT=${DELTA_WEIGHT:-0.5}
SELECTOR_TEMPERATURE=${SELECTOR_TEMPERATURE:-0.15}
DELTA_BETA=${DELTA_BETA:-0.1}
THRESHOLD_STEPS=${THRESHOLD_STEPS:-41}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-5}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
SEED=${SEED:-0}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -d "${TRAIN_CACHE_ROOT}" ]] || { echo "Train cache not found: ${TRAIN_CACHE_ROOT}" >&2; exit 2; }
[[ -d "${VAL_CACHE_ROOT}" ]] || { echo "Validation cache not found: ${VAL_CACHE_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a MODE_ARRAY <<< "${REPRESENTATION_MODES}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }

PIDS=()
ACTIVE=()
wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-ABL-RUN][ERROR] ${ACTIVE[$i]} failed; inspect train.log" >&2
      failed=1
    fi
  done
  PIDS=(); ACTIVE=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_mode() {
  local mode="$1" gpu="$2" out="${OUTPUT_ROOT}/${mode}"
  mkdir -p "${out}"
  echo "[REP-ABL-RUN][LAUNCH] mode=${mode} gpu=${gpu} out=${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT_DIR}/train_ray_relational_rep_ablation.py" \
    --train_cache_root "${TRAIN_CACHE_ROOT}" \
    --val_cache_root "${VAL_CACHE_ROOT}" \
    --output_dir "${out}" \
    --representation_mode "${mode}" \
    --val_scene_start "${VAL_SCENE_START}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --d_model "${D_MODEL}" \
    --nhead "${NHEAD}" \
    --num_layers "${NUM_LAYERS}" \
    --ff_dim "${FF_DIM}" \
    --dropout "${DROPOUT}" \
    --gate_weight "${GATE_WEIGHT}" \
    --selector_weight "${SELECTOR_WEIGHT}" \
    --delta_weight "${DELTA_WEIGHT}" \
    --selector_temperature "${SELECTOR_TEMPERATURE}" \
    --delta_beta "${DELTA_BETA}" \
    --threshold_steps "${THRESHOLD_STEPS}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --max_train_frames "${MAX_TRAIN_FRAMES}" \
    --max_val_frames "${MAX_VAL_FRAMES}" \
    --progress_every "${PROGRESS_EVERY}" \
    >"${out}/train.log" 2>&1 &
  PIDS+=("$!"); ACTIVE+=("${mode}")
}

slot=0
for raw_mode in "${MODE_ARRAY[@]}"; do
  mode="$(echo "${raw_mode}" | xargs)"
  [[ -n "${mode}" ]] || continue
  case "${mode}" in
    G0_current_full|G1_no_abs_raw|G2_residual_only|G3_residual_profile|G4_mean_profile) ;;
    *) echo "Unknown representation mode: ${mode}" >&2; exit 2 ;;
  esac
  launch_mode "${mode}" "${GPU_ARRAY[$slot]}"
  slot=$((slot+1))
  if [[ ${slot} -ge ${#GPU_ARRAY[@]} ]]; then
    wait_wave; slot=0
  fi
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

echo "[REP-ABL-RUN] completed: ${OUTPUT_ROOT}"
for raw_mode in "${MODE_ARRAY[@]}"; do
  mode="$(echo "${raw_mode}" | xargs)"
  [[ -n "${mode}" ]] || continue
  [[ -f "${OUTPUT_ROOT}/${mode}/best.json" ]] && echo "  ${mode}: ${OUTPUT_ROOT}/${mode}/best.json"
done
