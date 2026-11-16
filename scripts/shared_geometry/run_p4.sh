#!/usr/bin/env bash
# P4-0 diagnostics + P4-1 plain 2D depth baseline. No P4-2/official AP launch.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?check|p40-smoke|p40|depth-smoke|depth-cache|depth-train|depth-evaluate|depth-predict}"
if [[ "$STAGE" == check ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest tests.test_shared_geometry_p4 -v
  python -m compileall -q shared_geometry/p4_common.py shared_geometry/p4_score_diagnostics.py \
    shared_geometry/p4_depth_data.py shared_geometry/p4_depth_baseline.py tests/test_shared_geometry_p4.py
  bash -n scripts/shared_geometry/run_p4.sh
  exit 0
fi
: "${P0P1_ROOT:?Existing P1 cache root, read-only}" "${P2_ROOT:?Existing P2 result root, read-only}"
: "${P4_ROOT:?New isolated P4 output root}"
new="$(realpath -m "$P4_ROOT")"
for src in "$P0P1_ROOT" "$P2_ROOT" "${RUN_ROOT:-}" "${P3_ROOT:-}"; do
  [[ -z "$src" ]] && continue
  old="$(realpath -m "$src")"
  [[ "$new" != "$old" && "$new" != "$old/"* && "$old" != "$new/"* ]] || {
    echo 'P4_ROOT must not overlap any existing experiment root' >&2; exit 2;
  }
done
mkdir -p "$P4_ROOT"
DEVICE="${DEVICE:-cuda:0}";BATCH_SIZE="${BATCH_SIZE:-4}";VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
WEIGHT="$P2_ROOT/shared_reader/best.pt";EXPECTED="$P2_ROOT/shared_fixed_weight_comparison.json"
P1VAL="$P0P1_ROOT/p1_cache/validation/index.json"
resume=(); [[ "${RESUME:-0}" != 1 ]] || resume+=(--resume)
case "$STAGE" in
p40-smoke|p40)
  extra=();dest="$P4_ROOT/p40"
  if [[ "$STAGE" == p40-smoke ]]; then
    extra+=(--smoke --limit "${SMOKE_FRAMES:-5}" --repeats 2 --ig-steps 2);dest="$P4_ROOT/smoke/p40"
  else
    extra+=(--repeats "${MASK_REPEATS:-16}" --ig-steps "${IG_STEPS:-8}")
  fi
  python -m shared_geometry.p4_score_diagnostics --index "$P1VAL" \
    --weight "$WEIGHT" --expected-p2 "$EXPECTED" --output "$dest" \
    --batch-size "$VAL_BATCH_SIZE" --device "$DEVICE" --points 15 \
    --bootstrap "${BOOTSTRAP:-5000}" --seed 0 "${extra[@]}"
  ;;
depth-cache|depth-smoke)
  : "${RUN_ROOT:?Original shared-geometry FP16 cache root}" "${TARGET_CONFIG:?Exact original rendered-visible config}"
  base="$P4_ROOT/depth_targets";extra=()
  if [[ "$STAGE" == depth-smoke ]]; then
    base="$P4_ROOT/smoke/depth_targets";extra+=(--smoke --limit "${SMOKE_FRAMES:-5}")
  fi
  for part in train validation; do
    python -m shared_geometry.p4_depth_baseline build --index "$RUN_ROOT/${part}_index.json" \
      --targets "$TARGET_CONFIG" --output "$base/$part" "${extra[@]}" "${resume[@]}"
  done
  if [[ "$STAGE" == depth-smoke ]]; then
    python -m shared_geometry.p4_depth_baseline train --train-index "$base/train/index.json" \
      --val-index "$base/validation/index.json" --output "$P4_ROOT/smoke/depth_model" \
      --epochs 1 --batch-size "$BATCH_SIZE" --val-batch-size "$VAL_BATCH_SIZE" \
      --workers 0 --device "$DEVICE" --smoke
    python -m shared_geometry.p4_depth_baseline evaluate --depth-index "$base/validation/index.json" \
      --p1-index "$P1VAL" --original-index "$RUN_ROOT/validation_index.json" \
      --weight "$P4_ROOT/smoke/depth_model/best.pt" --shared-weight "$WEIGHT" \
      --expected-p2 "$EXPECTED" --output "$P4_ROOT/smoke/depth_evaluate" \
      --batch-size "$VAL_BATCH_SIZE" --bootstrap 100 --device "$DEVICE" --smoke
  fi
  ;;
depth-train)
  python -m shared_geometry.p4_depth_baseline train \
    --train-index "$P4_ROOT/depth_targets/train/index.json" \
    --val-index "$P4_ROOT/depth_targets/validation/index.json" \
    --output "$P4_ROOT/depth_model" --epochs "${EPOCHS:-20}" \
    --batch-size "$BATCH_SIZE" --accumulate "${ACCUMULATE:-1}" \
    --val-batch-size "$VAL_BATCH_SIZE" --hidden 32 --max-correction-m 0.05 \
    --gradient-weight 0.5 --lr 0.0001 --weight-decay 0.0001 --seed 0 \
    --workers "${NUM_WORKERS:-2}" --cpu-threads "${CPU_THREADS:-2}" --device "$DEVICE" "${resume[@]}"
  ;;
depth-evaluate)
  : "${RUN_ROOT:?Original shared-geometry FP16 cache root}"
  python -m shared_geometry.p4_depth_baseline evaluate \
    --depth-index "$P4_ROOT/depth_targets/validation/index.json" \
    --p1-index "$P1VAL" --original-index "$RUN_ROOT/validation_index.json" \
    --weight "$P4_ROOT/depth_model/best.pt" --shared-weight "$WEIGHT" --expected-p2 "$EXPECTED" \
    --output "$P4_ROOT/depth_evaluate" --batch-size "$VAL_BATCH_SIZE" \
    --bootstrap "${BOOTSTRAP:-5000}" --seed 0 --device "$DEVICE"
  ;;
depth-predict)
  : "${INPUT_NPZ:?Frozen RGB feature_map and pred_depth NPZ}" "${OUTPUT_NPZ:?New output file}"
  python -m shared_geometry.p4_depth_baseline predict --weight "$P4_ROOT/depth_model/best.pt" \
    --inputs "$INPUT_NPZ" --output "$OUTPUT_NPZ" --device "$DEVICE"
  ;;
*) echo "Unknown P4 stage: $STAGE" >&2;exit 2;;
esac
