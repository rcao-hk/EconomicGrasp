#!/usr/bin/env bash
set -euo pipefail

# P1 RGB-only inference over GraspNet test splits.
#
# Required:
#   DATASET_ROOT=/path/to/graspnet
#   P1_CKPT=/path/to/p1/checkpoint.tar
#
# Useful:
#   GPUS=0,1,2
#   SPLITS=test_seen,test_similar,test_novel
#   OUTPUT_ROOT=/path/to/p1_predictions
#   POSE_DEPTH_MODE=global_film
#   USE_FUSE_DEPTH=1
#   BATCH_SIZE=3
#   COLLISION_THRESH=0.01
#   SAMPLE_INTERVAL=1.0
#   P1_FORCE_ZERO_RECENTER=0
#   GRASPNESS_MODE=...
#   EXTRA_ARGS="..."

: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${P1_CKPT:?Set P1_CKPT}"

GPUS="${GPUS:-0}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/p1_center_recenter_inference}"
CAMERA="${CAMERA:-realsense}"
POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
USE_FUSE_DEPTH="${USE_FUSE_DEPTH:-1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1.0}"
P1_FORCE_ZERO_RECENTER="${P1_FORCE_ZERO_RECENTER:-0}"
SAVE_NOCOLLISION="${SAVE_NOCOLLISION:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
NGPU="${#GPU_ARRAY[@]}"
if [[ "${NGPU}" -le 0 ]]; then
  echo "No GPU specified" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
declare -a ACTIVE_PIDS=()

launch_split() {
  local SPLIT="$1"
  local GPU="$2"
  local OUT="${OUTPUT_ROOT}/${SPLIT}"
  mkdir -p "${OUT}"

  local -a ARGS=(
    inference_cva_p1_center_recenter.py
    --dataset_root "${DATASET_ROOT}"
    --camera "${CAMERA}"
    --checkpoint_path "${P1_CKPT}"
    --save_dir "${OUT}"
    --test_mode "${SPLIT}"
    --multi_modal
    --use_cdf
    --pose_depth_mode "${POSE_DEPTH_MODE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --collision_thresh "${COLLISION_THRESH}"
    --collision_voxel_size "${COLLISION_VOXEL_SIZE}"
    --sample_interval "${SAMPLE_INTERVAL}"
  )

  if [[ "${USE_FUSE_DEPTH}" == "1" ]]; then
    ARGS+=(--use_fuse_depth)
  fi
  if [[ "${P1_FORCE_ZERO_RECENTER}" == "1" ]]; then
    ARGS+=(--p1_force_zero_recenter)
  fi
  if [[ "${SAVE_NOCOLLISION}" == "1" ]]; then
    ARGS+=(--save_nocollision)
  fi
  if [[ -n "${GRASPNESS_MODE:-}" ]]; then
    ARGS+=(--graspness_mode "${GRASPNESS_MODE}")
  fi

  echo "[P1-INFER] split=${SPLIT} GPU=${GPU} out=${OUT}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${GPU}" python "${ARGS[@]}" ${EXTRA_ARGS} \
    >"${OUT}/inference.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
}

wait_wave() {
  local pid
  local failed=0
  for pid in "${ACTIVE_PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  ACTIVE_PIDS=()
  if [[ "${failed}" != "0" ]]; then
    echo "At least one P1 inference worker failed. Check split inference.log files." >&2
    exit 1
  fi
}

slot=0
for SPLIT in "${SPLIT_ARRAY[@]}"; do
  SPLIT="$(echo "${SPLIT}" | xargs)"
  [[ -n "${SPLIT}" ]] || continue
  GPU="${GPU_ARRAY[$slot]}"
  launch_split "${SPLIT}" "${GPU}"
  slot=$((slot + 1))
  if [[ "${slot}" -ge "${NGPU}" ]]; then
    wait_wave
    slot=0
  fi
done

if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
  wait_wave
fi

echo "[P1-INFER] completed. Predictions: ${OUTPUT_ROOT}"
