#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${INIT_CKPT:?Set INIT_CKPT to the controlled Stage-1 checkpoint}"

GPUS="${GPUS:-0,1,2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/ray_confidence_10pct}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EPOCH="${MAX_EPOCH:-20}"
LR="${LR:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-1}"
TRAIN_SAMPLE_INTERVAL="${TRAIN_SAMPLE_INTERVAL:-0.1}"
EVAL_SAMPLE_INTERVAL="${EVAL_SAMPLE_INTERVAL:-0.1}"
RC_MAX_BATCHES="${RC_MAX_BATCHES:-0}"
CALIBRATION_WEIGHT="${CALIBRATION_WEIGHT:-1.0}"
RANKING_WEIGHT="${RANKING_WEIGHT:-1.0}"
RANK_TEMPERATURE="${RANK_TEMPERATURE:-0.10}"
GRASPNESS_MODE="${GRASPNESS_MODE:-scene}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
if [[ "${NPROC}" -lt 1 ]]; then echo "No GPU specified" >&2; exit 2; fi
mkdir -p "${OUTPUT_ROOT}"

echo "[RC][TRAIN] gpus=${GPUS} nproc=${NPROC} out=${OUTPUT_ROOT}"
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
  "${SCRIPT_DIR}/train_ray_confidence.py" \
  --dataset_root "${DATASET_ROOT}" \
  --checkpoint_path "${INIT_CKPT}" \
  --log_dir "${OUTPUT_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LR}" \
  --max_epoch "${MAX_EPOCH}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_num_workers "${EVAL_NUM_WORKERS}" \
  --graspness_mode "${GRASPNESS_MODE}" \
  --rc_offsets_mm=-40,-20,-10,0,10,20,40 \
  --rc_train_sample_interval "${TRAIN_SAMPLE_INTERVAL}" \
  --rc_eval_sample_interval "${EVAL_SAMPLE_INTERVAL}" \
  --rc_calibration_weight "${CALIBRATION_WEIGHT}" \
  --rc_ranking_weight "${RANKING_WEIGHT}" \
  --rc_rank_temperature "${RANK_TEMPERATURE}" \
  --rc_max_batches "${RC_MAX_BATCHES}"
