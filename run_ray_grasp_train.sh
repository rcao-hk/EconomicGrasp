#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${INIT_CKPT:?Set INIT_CKPT to Stage-1 (or P2 when RESUME=1)}"
GPUS="${GPUS:-0}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
args=("${SCRIPT_DIR}/train_ray_grasp.py"
 --dataset_root "${DATASET_ROOT}" --checkpoint_path "${INIT_CKPT}"
 --log_dir "${OUTPUT_ROOT:-log/p2_ray_grasp}" --camera "${CAMERA:-realsense}"
 --batch_size "${BATCH_SIZE:-1}" --m_point "${M_POINT:-1024}"
 --max_epoch "${MAX_EPOCH:-20}" --learning_rate "${LR:-0.0001}"
 --num_workers "${NUM_WORKERS:-2}" --eval_num_workers "${EVAL_NUM_WORKERS:-1}"
 --graspness_mode "${GRASPNESS_MODE:-scene}"
 --ray_train_sample_interval "${TRAIN_SAMPLE_INTERVAL:-0.1}"
 --ray_eval_sample_interval "${EVAL_SAMPLE_INTERVAL:-0.1}"
 --ray_max_batches "${MAX_BATCHES:-0}" --ray_support_weight "${SUPPORT_WEIGHT:-1.0}")
# Pose/fuse settings are recovered and validated from the checkpoint, not guessed.
if [[ -n "${OFFSETS_MM:-}" ]]; then args+=("--ray_offsets_mm=${OFFSETS_MM}"); fi
if [[ -n "${CDF_LABEL_FOLDER:-}" ]]; then args+=(--cdf_label_folder "${CDF_LABEL_FOLDER}"); fi
if [[ "${RESUME:-0}" == "1" ]]; then args+=(--resume); fi
if [[ "${CHECKPOINT_DECODER:-1}" == "0" ]]; then args+=(--ray_no_checkpoint_decoder); fi
CUDA_VISIBLE_DEVICES="${GPUS}" python -m torch.distributed.run --standalone \
 --nproc_per_node="${#GPU_ARRAY[@]}" "${args[@]}" "$@"
