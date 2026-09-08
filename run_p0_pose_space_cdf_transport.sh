#!/usr/bin/env bash
set -euo pipefail

# P0 pose-space CDF-transport diagnostic.
# No GraspNet/Dex-Net evaluator is called.
#
# Required environment variables:
#   DATASET_ROOT=/path/to/graspnet
#   STUDENT_CKPT=/path/to/stage1_or_stage2_student.tar
#   TEACHER_CKPT=/path/to/stage0_teacher.tar
#
# Useful overrides:
#   GPUS=0,1,2,3,4,5
#   SPLITS=test_seen,test_similar,test_novel
#   OUTPUT_ROOT=/path/to/p0_results
#   POSE_DEPTH_MODE=global_film
#   USE_FUSE_DEPTH=1
#   BATCH_SIZE=3
#   QUERY_SAMPLE_PER_IMAGE=16
#   MAX_BATCHES=0
#   CDF_LABEL_FOLDER=...
#   EXTRA_ARGS="..."

: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${STUDENT_CKPT:?Set STUDENT_CKPT}"
: "${TEACHER_CKPT:?Set TEACHER_CKPT}"

GPUS="${GPUS:-0}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/p0_pose_space_cdf_transport}"
CAMERA="${CAMERA:-realsense}"
POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
USE_FUSE_DEPTH="${USE_FUSE_DEPTH:-1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-1}"
QUERY_SAMPLE_PER_IMAGE="${QUERY_SAMPLE_PER_IMAGE:-16}"
MAX_BATCHES="${MAX_BATCHES:-0}"
POINT_MATCH_THRESH_MM="${POINT_MATCH_THRESH_MM:-5.0}"
TOLERATED_PARALLEL_MM="${TOLERATED_PARALLEL_MM:-30.0}"
DEPTH_START_MM="${DEPTH_START_MM:-10.0}"
DEPTH_INTERVAL_MM="${DEPTH_INTERVAL_MM:-10.0}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"

mkdir -p "${OUTPUT_ROOT}"

for SPLIT in "${SPLIT_ARRAY[@]}"; do
  SPLIT="$(echo "${SPLIT}" | xargs)"
  [[ -n "${SPLIT}" ]] || continue
  OUT="${OUTPUT_ROOT}/${SPLIT}"
  mkdir -p "${OUT}"

  ARGS=(
    diagnose_cva_p0_pose_space_cdf_transport.py
    --dataset_root "${DATASET_ROOT}"
    --camera "${CAMERA}"
    --multi_modal
    --use_cdf
    --extend_angle
    --pose_depth_mode "${POSE_DEPTH_MODE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --eval_num_workers "${EVAL_NUM_WORKERS}"
    --seed "${SEED}"
    --log_dir "${OUT}/runtime_log"
    --distill_stage 2
    --teacher_checkpoint "${TEACHER_CKPT}"
    --p0_student_checkpoint "${STUDENT_CKPT}"
    --p0_split "${SPLIT}"
    --p0_output_dir "${OUT}"
    --p0_max_batches "${MAX_BATCHES}"
    --p0_point_match_thresh_mm "${POINT_MATCH_THRESH_MM}"
    --p0_tolerated_parallel_mm "${TOLERATED_PARALLEL_MM}"
    --p0_depth_start_mm "${DEPTH_START_MM}"
    --p0_depth_interval_mm "${DEPTH_INTERVAL_MM}"
    --p0_query_sample_per_image "${QUERY_SAMPLE_PER_IMAGE}"
    --p0_save_query_rows 1
  )

  if [[ "${USE_FUSE_DEPTH}" == "1" ]]; then
    ARGS+=(--use_fuse_depth)
  fi
  if [[ -n "${CDF_LABEL_FOLDER:-}" ]]; then
    ARGS+=(--cdf_label_folder "${CDF_LABEL_FOLDER}")
  fi
  if [[ -n "${GRASPNESS_MODE:-}" ]]; then
    ARGS+=(--graspness_mode "${GRASPNESS_MODE}")
  fi

  echo "[P0] split=${SPLIT} GPUs=${GPUS} nproc=${NPROC} out=${OUT}"
  # EXTRA_ARGS is intentionally shell-expanded last for local experimental flags.
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --standalone \
    --nproc_per_node="${NPROC}" \
    "${ARGS[@]}" \
    ${EXTRA_ARGS}
done
