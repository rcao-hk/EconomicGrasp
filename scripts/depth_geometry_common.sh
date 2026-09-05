#!/usr/bin/env bash
# Source this file from the launchers; activate your existing CUDA environment first.
set -euo pipefail

DEPTH_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$DEPTH_REPO_ROOT"
: "${DATASET_ROOT:?Set DATASET_ROOT to your GraspNet directory}"
: "${CHECKPOINT:?Set CHECKPOINT to a healthy Stage-1/2 RGB student checkpoint}"

PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$DEPTH_REPO_ROOT:$DEPTH_REPO_ROOT/libs/graspnetAPI:${PYTHONPATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEPTH_REPO_ROOT/results/depth_geometry}"
COMMON_DEPTH_ARGS=(
  --dataset_root "$DATASET_ROOT"
  --checkpoint_path "$CHECKPOINT"
  --camera "${CAMERA:-realsense}"
  --cdf_label_folder "${CDF_LABEL_FOLDER:-economic_grasp_label_300views_extend_angle_cdf_depth}"
  --graspness_mode "${GRASPNESS_MODE:-scene}"
  --num_workers "${NUM_WORKERS:-2}"
  --seed "${SEED:-0}"
  --min_depth "${MIN_DEPTH:-0.2}"
  --max_depth "${MAX_DEPTH:-1.0}"
  --m_point "${M_POINT:-1024}"
  --anchors_per_image "${ANCHORS_PER_IMAGE:-128}"
  --pairs_per_anchor "${PAIRS_PER_ANCHOR:-8}"
  --pair_radius_min_m "${PAIR_RADIUS_MIN_M:-0.005}"
  --pair_radius_max_m "${PAIR_RADIUS_MAX_M:-0.03}"
)
if [[ -n "${POSE_DEPTH_MODE:-}" ]]; then
  COMMON_DEPTH_ARGS+=(--pose_depth_mode "$POSE_DEPTH_MODE")
fi
if [[ -n "${USE_FUSE_DEPTH:-}" ]]; then
  COMMON_DEPTH_ARGS+=(--use_fuse_depth "$USE_FUSE_DEPTH")
fi

run_depth_command() {
  printf 'Running:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$@"
  fi
}
