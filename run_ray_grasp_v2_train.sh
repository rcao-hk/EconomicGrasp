#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${INIT_CKPT:?Set INIT_CKPT to completed P2-v1 checkpoint (or P2-v2 when RESUME=1)}"

GPUS="${GPUS:-0,1,2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/p2_ray_v2}"
CAMERA="${CAMERA:-realsense}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EPOCH="${MAX_EPOCH:-20}"
LR="${LR:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-1}"
GRASPNESS_MODE="${GRASPNESS_MODE:-scene}"
TRAIN_SAMPLE_INTERVAL="${TRAIN_SAMPLE_INTERVAL:-0.1}"
EVAL_SAMPLE_INTERVAL="${EVAL_SAMPLE_INTERVAL:-0.1}"
V2_HIDDEN="${V2_HIDDEN:-128}"
V2_LAYERS="${V2_LAYERS:-2}"
V2_HEADS="${V2_HEADS:-4}"
V2_DROPOUT="${V2_DROPOUT:-0.10}"
V2_ZERO_BIAS_INIT="${V2_ZERO_BIAS_INIT:-0.5}"
V2_TARGET_TEMPERATURE="${V2_TARGET_TEMPERATURE:-0.1}"
V2_LISTWISE_WEIGHT="${V2_LISTWISE_WEIGHT:-1.0}"
V2_CALIBRATION_WEIGHT="${V2_CALIBRATION_WEIGHT:-0.5}"
V2_MAX_BATCHES="${V2_MAX_BATCHES:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then echo "No GPU specified" >&2; exit 2; fi

args=(
  "${SCRIPT_DIR}/train_ray_grasp_v2.py"
  --dataset_root "${DATASET_ROOT}"
  --checkpoint_path "${INIT_CKPT}"
  --log_dir "${OUTPUT_ROOT}"
  --camera "${CAMERA}"
  --batch_size "${BATCH_SIZE}"
  --max_epoch "${MAX_EPOCH}"
  --learning_rate "${LR}"
  --num_workers "${NUM_WORKERS}"
  --eval_num_workers "${EVAL_NUM_WORKERS}"
  --graspness_mode "${GRASPNESS_MODE}"
  --v2_train_sample_interval "${TRAIN_SAMPLE_INTERVAL}"
  --v2_eval_sample_interval "${EVAL_SAMPLE_INTERVAL}"
  --v2_hidden "${V2_HIDDEN}"
  --v2_layers "${V2_LAYERS}"
  --v2_heads "${V2_HEADS}"
  --v2_dropout "${V2_DROPOUT}"
  --v2_zero_bias_init "${V2_ZERO_BIAS_INIT}"
  --v2_target_temperature "${V2_TARGET_TEMPERATURE}"
  --v2_listwise_weight "${V2_LISTWISE_WEIGHT}"
  --v2_calibration_weight "${V2_CALIBRATION_WEIGHT}"
  --v2_max_batches "${V2_MAX_BATCHES}"
)
if [[ -n "${CDF_LABEL_FOLDER:-}" ]]; then
  args+=(--cdf_label_folder "${CDF_LABEL_FOLDER}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  args+=(--resume)
fi

echo "[P2-V2] GPUs=${GPUS} init=${INIT_CKPT} out=${OUTPUT_ROOT} "
echo "[P2-V2] selector=${V2_LAYERS}x${V2_HIDDEN}/${V2_HEADS} dropout=${V2_DROPOUT} "
echo "[P2-V2] loss=listwise:${V2_LISTWISE_WEIGHT} calibration:${V2_CALIBRATION_WEIGHT} tau=${V2_TARGET_TEMPERATURE}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m torch.distributed.run --standalone \
  --nproc_per_node="${#GPU_ARRAY[@]}" "${args[@]}" "$@"
