#!/usr/bin/env bash
set -euo pipefail
# Evaluate the triple-mix checkpoint on 10% of each GraspNet and GN-Trans
# test_seen / test_similar / test_novel split (26 frames per scene by stride 10).
# GC6D is training-only in this script; its native evaluation is separate.

: "${DATASET_ROOT:?Set DATASET_ROOT to GraspNet root}"
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT to GN-Trans RGB root}"
: "${CKPT:?Set CKPT to a completed triple-mix checkpoint}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export DATASETS="${DATASETS:-both}"
export SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
export SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"
export POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/gntrans_gc6d_mix_eval}"
export ENABLE_INFERENCE="${ENABLE_INFERENCE:-1}"
export ENABLE_EVAL="${ENABLE_EVAL:-1}"
# Strict RGB-only by default. For legacy depth-based post-filter comparisons
# explicitly set COLLISION_THRESH=0.01 and GNTRANS_COLLISION_SOURCE=original_sensor.
export COLLISION_THRESH="${COLLISION_THRESH:-0}"
export GNTRANS_COLLISION_SOURCE="${GNTRANS_COLLISION_SOURCE:-none}"

if [[ "${COLLISION_THRESH}" != "0" && "${COLLISION_THRESH}" != "0.0" ]]; then
  if [[ "${GNTRANS_COLLISION_SOURCE}" == "none" ]]; then
    echo "Positive COLLISION_THRESH requires GNTRANS_COLLISION_SOURCE=original_sensor or virtual_gt" >&2
    exit 2
  fi
fi

bash run_gntrans_mix_cva_cdf_eval.sh
