#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
TRAIN_CACHE_ROOT=${TRAIN_CACHE_ROOT:-${WORK_ROOT}/cache}
VAL_CACHE_ROOT=${VAL_CACHE_ROOT:-${WORK_ROOT}/cache_val}
OUTPUT_DIR=${OUTPUT_DIR:-${WORK_ROOT}/relational_selective}
TRAIN_GPU=${TRAIN_GPU:-0}
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
mkdir -p "${OUTPUT_DIR}"

echo "[REL-RUN] train_cache=${TRAIN_CACHE_ROOT}"
echo "[REL-RUN] val_cache=${VAL_CACHE_ROOT}"
echo "[REL-RUN] output=${OUTPUT_DIR} gpu=${TRAIN_GPU}"
echo "[REL-RUN] architecture d_model=${D_MODEL} heads=${NHEAD} layers=${NUM_LAYERS} ff=${FF_DIM}"

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${PYTHON_BIN}" "${ROOT_DIR}/train_ray_relational_selective.py" \
  --train_cache_root "${TRAIN_CACHE_ROOT}" \
  --val_cache_root "${VAL_CACHE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
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
  --seed "${SEED}" \
  --device cuda:0 \
  --max_train_frames "${MAX_TRAIN_FRAMES}" \
  --max_val_frames "${MAX_VAL_FRAMES}" \
  --progress_every "${PROGRESS_EVERY}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "[REL-RUN] completed"
echo "  best:   ${OUTPUT_DIR}/checkpoint_best.tar"
echo "  latest: ${OUTPUT_DIR}/checkpoint_latest.tar"
