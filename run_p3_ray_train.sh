#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${INIT_CKPT:?Set INIT_CKPT to the controlled Stage-1 checkpoint, or a P3 checkpoint with RESUME=1}"

GPUS="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
[[ "${#GPU_ARRAY[@]}" -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }

args=("${SCRIPT_DIR}/train_p3_ray.py"
  --dataset_root "${DATASET_ROOT}"
  --checkpoint_path "${INIT_CKPT}"
  --log_dir "${OUTPUT_ROOT:-log/p3_ray_10pct}"
  --camera "${CAMERA:-realsense}"
  --batch_size "${BATCH_SIZE:-1}"
  --m_point "${M_POINT:-1024}"
  --max_epoch "${MAX_EPOCH:-20}"
  --learning_rate "${LR:-0.0001}"
  --weight_decay "${WEIGHT_DECAY:-0.0}"
  --num_workers "${NUM_WORKERS:-2}"
  --eval_num_workers "${EVAL_NUM_WORKERS:-1}"
  --graspness_mode "${GRASPNESS_MODE:-scene}"
  --p3_train_sample_interval "${TRAIN_SAMPLE_INTERVAL:-0.1}"
  --p3_eval_sample_interval "${EVAL_SAMPLE_INTERVAL:-0.1}"
  --p3_cdf_weight "${CDF_WEIGHT:-1.0}"
  --p3_width_weight "${WIDTH_WEIGHT:-10.0}"
  --p3_viability_weight "${VIABILITY_WEIGHT:-1.0}"
  --p3_joint_weight "${JOINT_WEIGHT:-1.0}"
  --p3_hidden "${P3_HIDDEN:-128}"
  --p3_layers "${P3_LAYERS:-2}"
  --p3_heads "${P3_HEADS:-4}"
  --p3_dropout "${P3_DROPOUT:-0.10}"
  --p3_max_batches "${P3_MAX_BATCHES:-0}"
)

if [[ -n "${OFFSETS_MM:-}" ]]; then args+=("--p3_offsets_mm=${OFFSETS_MM}"); fi
if [[ -n "${CDF_LABEL_FOLDER:-}" ]]; then args+=(--cdf_label_folder "${CDF_LABEL_FOLDER}"); fi
if [[ "${RESUME:-0}" == "1" ]]; then args+=(--resume); fi

# use_fuse_depth and pose_depth_mode are recovered from checkpoint metadata and
# validated inside the P3 runtime instead of being guessed by the launcher.
CUDA_VISIBLE_DEVICES="${GPUS}" python -m torch.distributed.run --standalone \
  --nproc_per_node="${#GPU_ARRAY[@]}" "${args[@]}" "$@"
