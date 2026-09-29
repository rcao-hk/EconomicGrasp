#!/usr/bin/env bash
# Run the two next-step experiments concurrently on disjoint GPU sets:
#
#   GPUs 0,1,2 -> zero-training Base/Field logit fusion sweep
#   GPUs 3,5,6 -> residual Metric Grasp Field matched20 train+infer+eval
#
# Both jobs are independent and share only read-only datasets/checkpoints.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SWEEP_GPUS=${SWEEP_GPUS:-0,1,2}
RESIDUAL_GPUS=${RESIDUAL_GPUS:-3,5,6}
MATCHED_ROOT=${MATCHED_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_mixed_matched}
RESIDUAL_ROOT=${RESIDUAL_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_residual_matched}
CONTROL_ROOT=${CONTROL_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_nextstep_parallel}

SWEEP_INFER_BATCH_SIZE=${SWEEP_INFER_BATCH_SIZE:-1}
RESIDUAL_INFER_BATCH_SIZE=${RESIDUAL_INFER_BATCH_SIZE:-1}
SWEEP_RESUME=${SWEEP_RESUME:-0}
RESIDUAL_RESUME=${RESIDUAL_RESUME:-0}
RESIDUAL_PHASES=${RESIDUAL_PHASES:-train,infer,eval}

mkdir -p "$CONTROL_ROOT"

IFS=',' read -r -a SWEEP_IDS <<< "$SWEEP_GPUS"
IFS=',' read -r -a RESIDUAL_IDS <<< "$RESIDUAL_GPUS"

for id in "${SWEEP_IDS[@]}" "${RESIDUAL_IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] Invalid GPU id: $id" >&2
    exit 2
  }
done
for a in "${SWEEP_IDS[@]}"; do
  for b in "${RESIDUAL_IDS[@]}"; do
    if [[ "$a" == "$b" ]]; then
      echo "[ERROR] GPU $a appears in both jobs; GPU sets must be disjoint." >&2
      exit 2
    fi
  done
done

SWEEP_LOG="$CONTROL_ROOT/blend_sweep.log"
RESIDUAL_LOG="$CONTROL_ROOT/residual_matched20.log"
SWEEP_PID=""
RESIDUAL_PID=""

cleanup() {
  local pid
  for pid in "$SWEEP_PID" "$RESIDUAL_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "============================================================"
echo "[MGF NEXT STEP PARALLEL]"
echo "  sweep GPUs       : $SWEEP_GPUS"
echo "  residual GPUs    : $RESIDUAL_GPUS"
echo "  absolute matched : $MATCHED_ROOT"
echo "  residual output  : $RESIDUAL_ROOT"
echo "  logs             : $CONTROL_ROOT"
echo "============================================================"

setsid env \
  GPUS="$SWEEP_GPUS" \
  INFER_GPUS="$SWEEP_GPUS" \
  MATCHED_ROOT="$MATCHED_ROOT" \
  INFER_BATCH_SIZE="$SWEEP_INFER_BATCH_SIZE" \
  RESUME="$SWEEP_RESUME" \
  bash "$ROOT_DIR/scripts/run_metric_grasp_field_blend_sweep.sh" \
  >"$SWEEP_LOG" 2>&1 &
SWEEP_PID=$!

setsid env \
  GPUS="$RESIDUAL_GPUS" \
  INFER_GPUS="$RESIDUAL_GPUS" \
  WORK_ROOT="$RESIDUAL_ROOT" \
  PHASES="$RESIDUAL_PHASES" \
  INFER_BATCH_SIZE="$RESIDUAL_INFER_BATCH_SIZE" \
  RESUME="$RESIDUAL_RESUME" \
  bash "$ROOT_DIR/scripts/run_metric_grasp_field_residual_matched20.sh" \
  >"$RESIDUAL_LOG" 2>&1 &
RESIDUAL_PID=$!

echo "[MGF NEXT STEP PARALLEL] jobs launched"
echo "  sweep pid    = $SWEEP_PID"
echo "  residual pid = $RESIDUAL_PID"
echo "  sweep log    = $SWEEP_LOG"
echo "  residual log = $RESIDUAL_LOG"

status=0
if ! wait "$SWEEP_PID"; then
  echo "[ERROR] blend sweep failed; inspect $SWEEP_LOG" >&2
  status=1
  if kill -0 "$RESIDUAL_PID" 2>/dev/null; then
    kill -TERM -- "-$RESIDUAL_PID" 2>/dev/null || true
  fi
fi

if [[ "$status" -eq 0 ]]; then
  if ! wait "$RESIDUAL_PID"; then
    echo "[ERROR] residual experiment failed; inspect $RESIDUAL_LOG" >&2
    status=1
  fi
else
  wait "$RESIDUAL_PID" 2>/dev/null || true
fi

SWEEP_PID=""
RESIDUAL_PID=""

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo
echo "============================================================"
echo "[MGF NEXT STEP PARALLEL] complete"
echo "Blend sweep summary:"
echo "  $MATCHED_ROOT/blend_logit_sweep/summary/sweep.md"
echo "Residual official AP:"
echo "  $RESIDUAL_ROOT/test_latest/official/"
echo "============================================================"
