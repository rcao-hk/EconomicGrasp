#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?check|p2-diagnose|p2-shared-train|p2-shared-compare}"
if [[ "$STAGE" == check ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest tests.test_shared_geometry_p2 -v
  python -m compileall -q shared_geometry/p2_geometry_diagnostics.py tests/test_shared_geometry_p2.py
  bash -n scripts/shared_geometry/run_p2.sh
  exit
fi
: "${P0P1_ROOT:?Existing P0/P1 root, read-only}"
: "${P2_ROOT:?New isolated P2 root}"
[[ "$(realpath -m "$P2_ROOT")" != "$(realpath -m "$P0P1_ROOT")" ]] || { echo 'P2_ROOT must differ from P0P1_ROOT' >&2; exit 2; }
[[ -z "${RUN_ROOT:-}" || "$(realpath -m "$P2_ROOT")" != "$(realpath -m "$RUN_ROOT")" ]] || { echo 'P2_ROOT must differ from original RUN_ROOT' >&2; exit 2; }
mkdir -p "$P2_ROOT"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
SEED="${SEED:-0}"
if [[ "${SMOKE:-0}" == 1 ]]; then SMOKE_ARGS=(--smoke); else SMOKE_ARGS=(); fi
if [[ "${RESUME:-0}" == 1 ]]; then RESUME_ARGS=(--resume); else RESUME_ARGS=(); fi
case "$STAGE" in
p2-diagnose)
  python -m shared_geometry.p2_geometry_diagnostics diagnose-p1 \
    --index "$P0P1_ROOT/p1_cache/validation/index.json" \
    --predicted-weight "$P0P1_ROOT/p1_models/predicted/best.pt" \
    --clean-weight "$P0P1_ROOT/p1_models/clean_visible/best.pt" \
    --output "$P2_ROOT/p1_native_relative_diagnostics.json" \
    --batch-size "$VAL_BATCH_SIZE" --bootstrap "$BOOTSTRAP" --seed "$SEED" \
    --device "$DEVICE" "${SMOKE_ARGS[@]}"
  ;;
p2-shared-train)
  python -m shared_geometry.p2_geometry_diagnostics shared-train \
    --train-index "$P0P1_ROOT/p1_cache/train/index.json" \
    --val-index "$P0P1_ROOT/p1_cache/validation/index.json" \
    --output "$P2_ROOT/shared_reader" \
    --hidden "${HIDDEN:-64}" --epochs "${EPOCHS:-20}" \
    --batch-size "$BATCH_SIZE" --accumulate "${ACCUMULATE:-1}" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --lr "${LR:-0.0001}" --weight-decay "${WEIGHT_DECAY:-0.0001}" \
    --cpu-threads "${CPU_THREADS:-2}" --seed "$SEED" --device "$DEVICE" \
    "${SMOKE_ARGS[@]}" "${RESUME_ARGS[@]}"
  ;;
p2-shared-compare)
  python -m shared_geometry.p2_geometry_diagnostics shared-compare \
    --index "$P0P1_ROOT/p1_cache/validation/index.json" \
    --weight "$P2_ROOT/shared_reader/best.pt" \
    --output "$P2_ROOT/shared_fixed_weight_comparison.json" \
    --batch-size "$VAL_BATCH_SIZE" --bootstrap "$BOOTSTRAP" --seed "$SEED" \
    --device "$DEVICE" "${SMOKE_ARGS[@]}"
  ;;
*) echo "Unknown P2 stage: $STAGE" >&2; exit 2;;
esac