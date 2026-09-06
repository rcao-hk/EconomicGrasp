#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/depth_controls_eval_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# CONTROLS_DIR may be explicit or derived from the training launcher variables.
if [[ -z "${CONTROLS_DIR:-}" && -n "${OUTPUT_ROOT:-}" && -n "${RUN_TAG:-}" ]]; then
  CONTROLS_DIR="$OUTPUT_ROOT/controls_${RUN_TAG}"
fi
INFER_ARGS=(
  --checkpoint_name "${CHECKPOINT_NAME:-checkpoint.tar}"
  --camera "${CAMERA:-realsense}"
  --frame_stride "${FRAME_STRIDE:-1}"
  --topk_views "${TOPK_VIEWS:-1}"
  --collision_thresh "${COLLISION_THRESH:-0}"
  --collision_voxel_size "${COLLISION_VOXEL_SIZE:-0.01}"
  --batch_size "${INFER_BATCH_SIZE:-1}"
  --num_workers "${INFER_NUM_WORKERS:-2}"
  --seed "${SEED:-0}"
  --m_point "${M_POINT:-1024}"
  --num_point "${NUM_POINT:-20000}"
  --graspness_threshold "${GRASPNESS_THRESHOLD:-0.1}"
  --graspness_mode "${GRASPNESS_MODE:-scene}"
  --min_depth "${MIN_DEPTH:-0.2}"
  --max_depth "${MAX_DEPTH:-1.0}"
  --bin_num "${BIN_NUM:-256}"
)
if [[ -n "${BASE_CHECKPOINT:-${CHECKPOINT:-}}" ]]; then
  INFER_ARGS+=(--base_checkpoint "${BASE_CHECKPOINT:-$CHECKPOINT}")
fi
if [[ -n "${CONTROLS_DIR:-}" ]]; then
  INFER_ARGS+=(--controls_dir "$CONTROLS_DIR")
fi
run_depth_eval_command "$PYTHON" -u inference_cva_depth_controls.py "${DEPTH_EVAL_ARGS[@]}" "${INFER_ARGS[@]}" "$@"
