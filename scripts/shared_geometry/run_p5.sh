#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?check|p50-smoke|p50|p51-cache|p51-train|p51-compare}"
if [[ "$STAGE" == check ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest tests.test_shared_geometry_p5 -v
  python -m compileall -q shared_geometry/p5_geometry_diagnostics.py \
    shared_geometry/p5_reader_adaptation.py tests/test_shared_geometry_p5.py
  bash -n scripts/shared_geometry/run_p5.sh
  exit 0
fi
: "${RUN_ROOT:?original frozen geometry root, read-only}"
: "${P0P1_ROOT:?existing P1 root, read-only}"
: "${P2_ROOT:?existing P2 root, read-only}"
: "${P4_ROOT:?existing P4 root, read-only}"
: "${P5_ROOT:?new P5 root; must differ from every earlier root}"
P5_ABS="$(realpath -m "$P5_ROOT")"
for p in "$RUN_ROOT" "$P0P1_ROOT" "$P2_ROOT" "$P4_ROOT"; do
  OLD="$(realpath -m "$p")"
  if [[ "$P5_ABS" == "$OLD" || "$P5_ABS" == "$OLD"/* || "$OLD" == "$P5_ABS"/* ]]; then
    echo "P5_ROOT must be distinct from and outside all earlier roots: $OLD" >&2; exit 2
  fi
done
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
ACCUMULATE="${ACCUMULATE:-1}"
SEED="${SEED:-0}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
SMOKE_ARGS=()
LIMIT_ARGS=()
if [[ "${SMOKE:-0}" == 1 || "$STAGE" == p50-smoke ]]; then
  SMOKE_ARGS=(--smoke)
  LIMIT_ARGS=(--limit "${LIMIT:-4}")
  ROOT="$P5_ABS/smoke"
else
  ROOT="$P5_ABS"
fi
RESUME_ARGS=()
if [[ "${RESUME:-0}" == 1 ]]; then RESUME_ARGS=(--resume); fi
mkdir -p "$ROOT"
case "$STAGE" in
p50|p50-smoke)
  python -m shared_geometry.p5_geometry_diagnostics \
    --depth-index "$P4_ROOT/depth_targets/validation/index.json" \
    --p1-index "$P0P1_ROOT/p1_cache/validation/index.json" \
    --original-index "$RUN_ROOT/validation_index.json" \
    --p4-summary "$P4_ROOT/depth_evaluate/summary.json" \
    --shared-weight "$P2_ROOT/shared_reader/best.pt" \
    --output "$ROOT/p50_geometry_diagnostics" --device "$DEVICE" \
    --bootstrap "$BOOTSTRAP" --seed "$SEED" \
    --edge-threshold-m "${EDGE_THRESHOLD_M:-0.02}" \
    --support-radius-px "${SUPPORT_RADIUS_PX:-2}" \
    "${SMOKE_ARGS[@]}" "${LIMIT_ARGS[@]}"
  ;;
p51-cache)
  for partition in train validation; do
    python -m shared_geometry.p5_reader_adaptation build \
      --partition "$partition" \
      --depth-index "$P4_ROOT/depth_targets/$partition/index.json" \
      --p1-index "$P0P1_ROOT/p1_cache/$partition/index.json" \
      --depth-weight "$P4_ROOT/depth_model/best.pt" \
      --output "$ROOT/p51_cache/$partition" --device "$DEVICE" \
      "${SMOKE_ARGS[@]}" "${RESUME_ARGS[@]}" "${LIMIT_ARGS[@]}"
  done
  ;;
p51-train)
  for arm in predicted corrected; do
    python -m shared_geometry.p5_reader_adaptation train --arm "$arm" \
      --train-index "$ROOT/p51_cache/train/index.json" \
      --val-index "$ROOT/p51_cache/validation/index.json" \
      --output "$ROOT/p51_models/$arm" --device "$DEVICE" \
      --epochs "${EPOCHS:-20}" --batch-size "$BATCH_SIZE" \
      --accumulate "$ACCUMULATE" --val-batch-size "$VAL_BATCH_SIZE" \
      --hidden "${HIDDEN:-64}" --lr "${LR:-0.0001}" \
      --weight-decay "${WEIGHT_DECAY:-0.0001}" \
      --cpu-threads "${CPU_THREADS:-2}" --seed "$SEED" \
      "${SMOKE_ARGS[@]}" "${RESUME_ARGS[@]}"
  done
  ;;
p51-compare)
  python -m shared_geometry.p5_reader_adaptation compare \
    --index "$ROOT/p51_cache/validation/index.json" \
    --p1-index "$P0P1_ROOT/p1_cache/validation/index.json" \
    --predicted-weight "$ROOT/p51_models/predicted/best.pt" \
    --corrected-weight "$ROOT/p51_models/corrected/best.pt" \
    --shared-weight "$P2_ROOT/shared_reader/best.pt" \
    --expected-p4 "$P4_ROOT/depth_evaluate/summary.json" \
    --output "$ROOT/p51_reader_comparison" --device "$DEVICE" \
    --batch-size "$VAL_BATCH_SIZE" --bootstrap "$BOOTSTRAP" \
    --seed "$SEED" "${SMOKE_ARGS[@]}"
  ;;
*) echo "Unknown P5 stage: $STAGE" >&2; exit 2;;
esac
