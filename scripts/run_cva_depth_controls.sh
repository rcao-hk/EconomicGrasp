#!/usr/bin/env bash
# All arms start from CHECKPOINT; never chain one arm's output into another.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/depth_geometry_common.sh"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
for DEPTH_VARIANT in none foreground anchor; do
  VARIANT="$DEPTH_VARIANT" \
  OUTPUT_DIR="$OUTPUT_ROOT/controls_${RUN_TAG}/$DEPTH_VARIANT" \
  bash "$DEPTH_REPO_ROOT/scripts/train_cva_depth_geometry.sh" "$@"
done
