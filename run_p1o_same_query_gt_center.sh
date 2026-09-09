#!/usr/bin/env bash
set -euo pipefail

# P1-O same-query GT-center oracle over GraspNet test splits.
# Generates paired Stage-1 baseline and GT-center-oracle predictions.
#
# Required:
#   DATASET_ROOT=/path/to/graspnet
#   STAGE1_CKPT=/path/to/controlled_stage1_checkpoint.tar
#
# Useful:
#   GPUS=0,1,2
#   SPLITS=test_seen,test_similar,test_novel
#   OUTPUT_ROOT=/path/to/p1o_same_query_gt_center
#   SAMPLE_INTERVAL=0.1
#   POSE_DEPTH_MODE=global_film
#   USE_FUSE_DEPTH=1
#   GRASPNESS_MODE=scene
#   COLLISION_THRESH=0.01
#   BATCH_SIZE=3
#   P1O_MAX_BATCHES=0
#   EXTRA_ARGS="..."

: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${STAGE1_CKPT:?Set STAGE1_CKPT}"

GPUS="${GPUS:-0}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/p1o_same_query_gt_center}"
CAMERA="${CAMERA:-realsense}"
POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
USE_FUSE_DEPTH="${USE_FUSE_DEPTH:-1}"
GRASPNESS_MODE="${GRASPNESS_MODE:-scene}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
SAVE_NOCOLLISION="${SAVE_NOCOLLISION:-0}"
P1O_MAX_BATCHES="${P1O_MAX_BATCHES:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "${GRASPNESS_MODE}" in
  scene|instance) ;;
  *)
    echo "Invalid GRASPNESS_MODE=${GRASPNESS_MODE}; expected scene or instance" >&2
    exit 2
    ;;
esac

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
    diagnose_p1o_same_query_gt_center.py
    --dataset_root "${DATASET_ROOT}"
    --camera "${CAMERA}"
    --checkpoint_path "${STAGE1_CKPT}"
    --distill_stage 1
    --save_dir "${OUT}"
    --test_mode "${SPLIT}"
    --multi_modal
    --use_cdf
    --pose_depth_mode "${POSE_DEPTH_MODE}"
    --graspness_mode "${GRASPNESS_MODE}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --collision_thresh "${COLLISION_THRESH}"
    --collision_voxel_size "${COLLISION_VOXEL_SIZE}"
    --p1o_max_batches "${P1O_MAX_BATCHES}"
  )

  if [[ "${USE_FUSE_DEPTH}" == "1" ]]; then
    ARGS+=(--use_fuse_depth)
  fi
  if [[ "${SAVE_NOCOLLISION}" == "1" ]]; then
    ARGS+=(--save_nocollision)
  fi

  echo "[P1-O] split=${SPLIT} GPU=${GPU} out=${OUT} sample=${SAMPLE_INTERVAL}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${GPU}" python "${ARGS[@]}" ${EXTRA_ARGS} \
    >"${OUT}/p1o.log" 2>&1 &
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
    echo "At least one P1-O worker failed. Check split p1o.log files." >&2
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

echo "[P1-O] completed. Outputs: ${OUTPUT_ROOT}"
