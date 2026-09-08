#!/usr/bin/env bash
set -euo pipefail

# P1 grasp-query center recentering training.
# No GraspNet/Dex-Net evaluator is called.
#
# Required:
#   DATASET_ROOT=/path/to/graspnet
#   INIT_CKPT=/path/to/stage1_checkpoint_20.tar
#
# Main control:
#   P1_TRAIN_MODE=center_only   # center_only | head_grasp | joint
#
# Useful:
#   GPUS=0,1,2,3,4,5
#   OUTPUT_ROOT=log/p1_center_recenter_center_only
#   POSE_DEPTH_MODE=global_film
#   USE_FUSE_DEPTH=1
#   MAX_EPOCH=5
#   BATCH_SIZE=4
#   LEARNING_RATE=              # auto: 1e-3 frozen-base, 1e-4 joint
#   CENTER_LOSS_WEIGHT=1.0
#   CENTER_BETA_M=0.01
#   MAX_RESIDUAL_M=0.08
#   P1_HIDDEN_DIM=128
#   ENABLE_EVAL=1
#   CDF_LABEL_FOLDER=...
#   GRASPNESS_MODE=...
#   EXTRA_ARGS="..."

: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${INIT_CKPT:?Set INIT_CKPT}"

GPUS="${GPUS:-0}"
P1_TRAIN_MODE="${P1_TRAIN_MODE:-center_only}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/p1_center_recenter_${P1_TRAIN_MODE}}"
CAMERA="${CAMERA:-realsense}"
POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
USE_FUSE_DEPTH="${USE_FUSE_DEPTH:-1}"
MAX_EPOCH="${MAX_EPOCH:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-1}"
CENTER_LOSS_WEIGHT="${CENTER_LOSS_WEIGHT:-1.0}"
CENTER_BETA_M="${CENTER_BETA_M:-0.01}"
MAX_RESIDUAL_M="${MAX_RESIDUAL_M:-0.08}"
P1_HIDDEN_DIM="${P1_HIDDEN_DIM:-128}"
ENABLE_EVAL="${ENABLE_EVAL:-1}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "${P1_TRAIN_MODE}" in
  center_only|head_grasp)
    DEFAULT_LR="0.001"
    ;;
  joint)
    DEFAULT_LR="0.0001"
    ;;
  *)
    echo "Invalid P1_TRAIN_MODE=${P1_TRAIN_MODE}" >&2
    exit 2
    ;;
esac
LEARNING_RATE="${LEARNING_RATE:-${DEFAULT_LR}}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
mkdir -p "${OUTPUT_ROOT}"

ARGS=(
  train_cva_p1_center_recenter_ddp.py
  --dataset_root "${DATASET_ROOT}"
  --camera "${CAMERA}"
  --log_dir "${OUTPUT_ROOT}"
  --checkpoint_path "${INIT_CKPT}"
  --distill_stage 1
  --multi_modal
  --use_cdf
  --extend_angle
  --pose_depth_mode "${POSE_DEPTH_MODE}"
  --max_epoch "${MAX_EPOCH}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --eval_num_workers "${EVAL_NUM_WORKERS}"
  --learning_rate "${LEARNING_RATE}"
  --seed "${SEED}"
  --p1_train_mode "${P1_TRAIN_MODE}"
  --p1_center_loss_weight "${CENTER_LOSS_WEIGHT}"
  --p1_center_beta_m "${CENTER_BETA_M}"
  --p1_max_residual_m "${MAX_RESIDUAL_M}"
  --p1_hidden_dim "${P1_HIDDEN_DIM}"
)

if [[ "${USE_FUSE_DEPTH}" == "1" ]]; then
  ARGS+=(--use_fuse_depth)
fi
if [[ "${ENABLE_EVAL}" == "1" ]]; then
  ARGS+=(--enable_eval)
fi
if [[ -n "${CDF_LABEL_FOLDER:-}" ]]; then
  ARGS+=(--cdf_label_folder "${CDF_LABEL_FOLDER}")
fi
if [[ -n "${GRASPNESS_MODE:-}" ]]; then
  ARGS+=(--graspness_mode "${GRASPNESS_MODE}")
fi

echo "[P1-TRAIN] mode=${P1_TRAIN_MODE} GPUs=${GPUS} nproc=${NPROC}"
echo "[P1-TRAIN] init=${INIT_CKPT}"
echo "[P1-TRAIN] output=${OUTPUT_ROOT} lr=${LEARNING_RATE}"

# EXTRA_ARGS is intentionally expanded last for local experiment controls.
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  "${ARGS[@]}" \
  ${EXTRA_ARGS}
