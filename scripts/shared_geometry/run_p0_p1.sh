#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?check|p0|p1-cache|p1-train|p1-compare}"
if [[ "$STAGE" == 'check' ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
    tests.test_shared_geometry_p0_p1 -v
  python -m compileall -q shared_geometry/p0_relative_decision.py \
    shared_geometry/p1_clean_visible.py tests/test_shared_geometry_p0_p1.py
  bash -n scripts/shared_geometry/run_p0_p1.sh
  exit
fi
# The ORIGINAL experiment cache and results are read-only. All new artefacts
# live in a different directory and overwrite protection stays enabled.
: "${RUN_ROOT:?Existing shared_visible_geometry_fp16_run01 root with train/validation index}"
: "${P0P1_ROOT:?New isolated P0/P1 output root (must NOT equal RUN_ROOT)}"
[[ "$(realpath -m "$RUN_ROOT")" != "$(realpath -m "$P0P1_ROOT")" ]] || {
  echo 'P0P1_ROOT must be distinct from RUN_ROOT' >&2; exit 2;
}
mkdir -p "$P0P1_ROOT"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
ACCUMULATE="${ACCUMULATE:-1}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-0}"
if [[ "${SMOKE:-0}" == 1 ]]; then
  SMOKE_ARGS=(--smoke)
else
  SMOKE_ARGS=()
fi
if [[ "${RESUME:-0}" == 1 ]]; then RESUME_ARGS=(--resume); else RESUME_ARGS=(); fi
case "$STAGE" in
p0)
  python -m shared_geometry.p0_relative_decision \
    --index "$RUN_ROOT/validation_index.json" \
    --weight "local=$RUN_ROOT/models/local/best.pt" \
    --weight "predicted=$RUN_ROOT/models/predicted/best.pt" \
    --weight "h1=$RUN_ROOT/models/h1/best.pt" \
    --weight "h2=$RUN_ROOT/models/h2/best.pt" \
    --weight "h2_stale=$RUN_ROOT/models/h2_stale/best.pt" \
    --batch-size "$VAL_BATCH_SIZE" --bootstrap "${BOOTSTRAP:-5000}" \
    --device "$DEVICE" --seed "$SEED" --visibility \
    --output "$P0P1_ROOT/p0_relative_decisions.json" "${SMOKE_ARGS[@]}"
  ;;
p1-cache)
  : "${TARGET_CONFIG:?Use the exact signed clean rendered visible teacher config of original export}"
  for partition in train validation; do
    python -m shared_geometry.p1_clean_visible build \
      --index "$RUN_ROOT/${partition}_index.json" \
      --targets "$TARGET_CONFIG" --output "$P0P1_ROOT/p1_cache/$partition" \
      "${SMOKE_ARGS[@]}" "${RESUME_ARGS[@]}"
  done
  ;;
p1-train)
  # Same hyperparameters, RNG, cached action banks, labels and real B=4 for both arms.
  for arm in predicted clean_visible; do
    python -m shared_geometry.p1_clean_visible train \
      --train-index "$P0P1_ROOT/p1_cache/train/index.json" \
      --val-index "$P0P1_ROOT/p1_cache/validation/index.json" \
      --arm "$arm" --output "$P0P1_ROOT/p1_models/$arm" \
      --hidden "${HIDDEN:-64}" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
      --accumulate "$ACCUMULATE" --val-batch-size "$VAL_BATCH_SIZE" \
      --lr "${LR:-0.0001}" --weight-decay "${WEIGHT_DECAY:-0.0001}" \
      --seed "$SEED" --cpu-threads "${CPU_THREADS:-2}" --device "$DEVICE" \
      "${SMOKE_ARGS[@]}" "${RESUME_ARGS[@]}"
  done
  ;;
p1-compare)
  python -m shared_geometry.p1_clean_visible compare \
    --index "$P0P1_ROOT/p1_cache/validation/index.json" \
    --predicted-weight "$P0P1_ROOT/p1_models/predicted/best.pt" \
    --clean-weight "$P0P1_ROOT/p1_models/clean_visible/best.pt" \
    --h2-weight "$RUN_ROOT/models/h2/best.pt" \
    --original-val-index "$RUN_ROOT/validation_index.json" \
    --device "$DEVICE" --batch-size "$VAL_BATCH_SIZE" \
    --bootstrap "${BOOTSTRAP:-5000}" --seed "$SEED" \
    --output "$P0P1_ROOT/p1_matched_comparison.json" "${SMOKE_ARGS[@]}"
  ;;
*) echo "Invalid P0/P1 stage $STAGE" >&2; exit 2;;
esac