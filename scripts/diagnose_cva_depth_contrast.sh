#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/depth_geometry_common.sh"

DIAG_SPLIT="${DIAG_SPLIT:-test_seen}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DIAG_OUTPUT="${OUTPUT_DIR:-$OUTPUT_ROOT/contrast_${DIAG_SPLIT}_${RUN_TAG}}"
EXTRA_DEPTH_ARGS=()
if [[ "${PROBE_GEOMETRY_GRADIENT:-0}" == 1 ]]; then
  EXTRA_DEPTH_ARGS+=(--probe_geometry_gradient)
fi
run_depth_command "$PYTHON" diagnose_cva_depth_contrast.py \
  "${COMMON_DEPTH_ARGS[@]}" \
  --output_dir "$DIAG_OUTPUT" \
  --split "$DIAG_SPLIT" \
  --max_frames "${DIAG_MAX_FRAMES:-32}" \
  --fixed_queries "${FIXED_QUERIES:-256}" \
  --frame_stride "${FRAME_STRIDE:-1}" \
  --scene_ids "${SCENE_IDS:-}" \
  --betas 0 0.25 0.5 1 1.25 \
  "${EXTRA_DEPTH_ARGS[@]}" "$@"
