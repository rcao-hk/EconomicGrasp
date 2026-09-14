#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?Set DATASET_ROOT to GraspNet root}"

GPUS="${GPUS:-0,1,2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/log/cva_cdf_graspnet_20pct}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.2}"
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EPOCH="${MAX_EPOCH:-20}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
DEPTH_WEIGHT_DECAY="${DEPTH_WEIGHT_DECAY:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-5}"
POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
USE_FUSE_DEPTH="${USE_FUSE_DEPTH:-1}"
SEED="${SEED:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
if [[ ${NPROC} -lt 1 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPUS}"
mkdir -p "${OUTPUT_ROOT}"

extra=()
if [[ "${USE_FUSE_DEPTH}" == "1" ]]; then
  extra+=(--use_fuse_depth)
elif [[ "${USE_FUSE_DEPTH}" != "0" ]]; then
  echo "USE_FUSE_DEPTH must be 0 or 1" >&2
  exit 2
fi

# Fresh RGB-only CVA-CDF training. With TRAIN_FRACTION=0.2 this selects
# frames 0,5,10,...,255 in every train scene: 52 * 100 = 5200 images/epoch,
# matching the exposure of the 10% GraspNet + 10% GN-Trans mixed run.
torchrun --standalone --nproc_per_node="${NPROC}" train_cva_graspnet_subset_ddp.py \
  --dataset_root "${DATASET_ROOT}" \
  --log_dir "${OUTPUT_ROOT}" \
  --camera realsense \
  --batch_size "${BATCH_SIZE}" \
  --max_epoch "${MAX_EPOCH}" \
  --learning_rate "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --depth_weight_decay "${DEPTH_WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_num_workers "${EVAL_NUM_WORKERS}" \
  --ckpt_save_interval "${CKPT_SAVE_INTERVAL}" \
  --train_fraction "${TRAIN_FRACTION}" \
  --eval_fraction "${EVAL_FRACTION}" \
  --multi_modal \
  --use_cdf \
  --extend_angle \
  --enable_eval \
  --eval_start_epoch 0 \
  --graspness_mode scene \
  --kview_mode A1 \
  --pose_depth_mode "${POSE_DEPTH_MODE}" \
  --seed "${SEED}" \
  "${extra[@]}"
