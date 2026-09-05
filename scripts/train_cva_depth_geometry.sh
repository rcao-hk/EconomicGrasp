#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/depth_geometry_common.sh"

VARIANT="${VARIANT:-anchor}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_OUTPUT="${OUTPUT_DIR:-$OUTPUT_ROOT/train_${VARIANT}_${RUN_TAG}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LAUNCH=("$PYTHON")
if (( NPROC_PER_NODE > 1 )); then
  LAUNCH+=(-m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE")
fi
EXTRA_DEPTH_ARGS=()
if [[ "${RESUME:-0}" == 1 ]]; then
  EXTRA_DEPTH_ARGS+=(--resume)
fi
if [[ -n "${EVAL_SCENE_IDS:-}" ]]; then
  EXTRA_DEPTH_ARGS+=(--eval_scene_ids "$EVAL_SCENE_IDS")
fi
run_depth_command "${LAUNCH[@]}" train_cva_depth_geometry.py \
  "${COMMON_DEPTH_ARGS[@]}" \
  --output_dir "$TRAIN_OUTPUT" \
  --variant "$VARIANT" \
  --train_scope "${TRAIN_SCOPE:-joint}" \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --depth_lr "${DEPTH_LR:-0.00001}" \
  --grasp_lr "${GRASP_LR:-0.00001}" \
  --metric_depth_weight "${METRIC_DEPTH_WEIGHT:-10}" \
  --relative_weight "${RELATIVE_WEIGHT:-10}" \
  --relative_warmup_epochs "${RELATIVE_WARMUP_EPOCHS:-1}" \
  --clip_mode "${CLIP_MODE:-global}" \
  --max_frames "${MAX_TRAIN_FRAMES:-0}" \
  --max_steps_per_epoch "${MAX_STEPS_PER_EPOCH:-0}" \
  --eval_max_frames "${EVAL_MAX_FRAMES:-64}" \
  --eval_split "${EVAL_SPLIT:-train}" \
  --frame_stride "${FRAME_STRIDE:-1}" \
  --scene_ids "${SCENE_IDS:-}" \
  "${EXTRA_DEPTH_ARGS[@]}" "$@"
