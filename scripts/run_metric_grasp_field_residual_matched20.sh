#!/usr/bin/env bash
# Train/evaluate the residual Metric Grasp Field under the mixed-matched20 regime.
#
# Final scorer:
#   final CDF = detached Base-CVA CDF + zero-initialized monotone Field residual
#
# The Base scorer remains supervised by its auxiliary BCE; final CDF BCE +
# listwise ranking optimize only the residual correction path on top of Base.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_residual_matched}
GPUS=${GPUS:-3,5,6}
INFER_GPUS=${INFER_GPUS:-$GPUS}
PHASES=${PHASES:-train,infer,eval}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-1}
RESUME=${RESUME:-0}

echo "============================================================"
echo "[MGF RESIDUAL MATCHED20]"
echo "  work root        : $WORK_ROOT"
echo "  GPUs             : $GPUS"
echo "  phases           : $PHASES"
echo "  score mode       : residual"
echo "  final decode     : residual-composed Field score"
echo "  collision eval   : 0.01 sensor-cloud filter"
echo "============================================================"

env \
  WORK_ROOT="$WORK_ROOT" \
  GPUS="$GPUS" \
  INFER_GPUS="$INFER_GPUS" \
  PHASES="$PHASES" \
  CHECKPOINT_KIND=latest \
  RESUME="$RESUME" \
  FIELD_SCORE_MODE=residual \
  SCORE_SOURCE=field \
  INFER_BATCH_SIZE="$INFER_BATCH_SIZE" \
  COLLISION_THRESH=0.01 \
  COLLISION_VOXEL_SIZE=0.01 \
  COLLISION_APPROACH_DIST=0.05 \
  bash "$ROOT_DIR/scripts/run_metric_grasp_field_mixed_matched20.sh"
