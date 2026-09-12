#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${INIT_CKPT:?Set INIT_CKPT to the controlled Stage-1 checkpoint}"

GPUS="${GPUS:-0,1,2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/log/p5_repair_10pct}"
TRAIN_SAMPLE_INTERVAL="${TRAIN_SAMPLE_INTERVAL:-0.1}"
EVAL_SAMPLE_INTERVAL="${EVAL_SAMPLE_INTERVAL:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EPOCH="${MAX_EPOCH:-20}"
LR="${LR:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-1}"
P5_MAX_BATCHES="${P5_MAX_BATCHES:-0}"

P5_HIDDEN="${P5_HIDDEN:-128}"
P5_LAYERS="${P5_LAYERS:-2}"
P5_HEADS="${P5_HEADS:-4}"
P5_NEIGHBORS="${P5_NEIGHBORS:-8}"
P5_DROPOUT="${P5_DROPOUT:-0.10}"
P5_MAX_DELTA_M="${P5_MAX_DELTA_M:-0.06}"
P5_TARGET_RADIUS_M="${P5_TARGET_RADIUS_M:-0.06}"
P5_CORRUPT_PROB="${P5_CORRUPT_PROB:-0.8}"
P5_SCENE_BIAS_SIGMA_M="${P5_SCENE_BIAS_SIGMA_M:-0.012}"
P5_SCALE_SIGMA="${P5_SCALE_SIGMA:-0.025}"
P5_REGION_SIGMA_M="${P5_REGION_SIGMA_M:-0.015}"
P5_REGION_GRID="${P5_REGION_GRID:-7}"
P5_REPAIR_WEIGHT="${P5_REPAIR_WEIGHT:-1.0}"
P5_UNKNOWN_IDENTITY_WEIGHT="${P5_UNKNOWN_IDENTITY_WEIGHT:-0.02}"
P5_BETA_M="${P5_BETA_M:-0.005}"
P5_LOG_EVERY="${P5_LOG_EVERY:-20}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
mkdir -p "${OUTPUT_ROOT}"

torchrun --standalone --nproc_per_node="${NPROC}" train_p5.py \
  --dataset_root "${DATASET_ROOT}" \
  --checkpoint_path "${INIT_CKPT}" \
  --log_dir "${OUTPUT_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epoch "${MAX_EPOCH}" \
  --learning_rate "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_num_workers "${EVAL_NUM_WORKERS}" \
  --ckpt_save_interval "${CKPT_SAVE_INTERVAL}" \
  --graspness_mode scene \
  --kview_mode A1 \
  --p5_train_sample_interval "${TRAIN_SAMPLE_INTERVAL}" \
  --p5_eval_sample_interval "${EVAL_SAMPLE_INTERVAL}" \
  --p5_hidden "${P5_HIDDEN}" \
  --p5_layers "${P5_LAYERS}" \
  --p5_heads "${P5_HEADS}" \
  --p5_neighbors "${P5_NEIGHBORS}" \
  --p5_dropout "${P5_DROPOUT}" \
  --p5_max_delta_m "${P5_MAX_DELTA_M}" \
  --p5_target_radius_m "${P5_TARGET_RADIUS_M}" \
  --p5_corrupt_prob "${P5_CORRUPT_PROB}" \
  --p5_scene_bias_sigma_m "${P5_SCENE_BIAS_SIGMA_M}" \
  --p5_scale_sigma "${P5_SCALE_SIGMA}" \
  --p5_region_sigma_m "${P5_REGION_SIGMA_M}" \
  --p5_region_grid "${P5_REGION_GRID}" \
  --p5_repair_weight "${P5_REPAIR_WEIGHT}" \
  --p5_unknown_identity_weight "${P5_UNKNOWN_IDENTITY_WEIGHT}" \
  --p5_smooth_l1_beta_m "${P5_BETA_M}" \
  --p5_log_every "${P5_LOG_EVERY}" \
  --p5_max_batches "${P5_MAX_BATCHES}"
