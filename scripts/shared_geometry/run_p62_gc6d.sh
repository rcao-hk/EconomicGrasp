#!/usr/bin/env bash
# GC6D -> frozen P6-2A exact action bank. No training / no official AP.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?check|lineage|plan|parity|smoke|build|audit-smoke|audit|evaluate}"
if [[ "$STAGE" == check ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_p62_gc6d.py' -v
  python -m compileall -q shared_geometry/p62_gc6d*.py tests/test_p62_gc6d.py
  bash -n scripts/shared_geometry/run_p62_gc6d.sh
  exit 0
fi
P62_ROOT="${P62_ROOT:-/data/robotarm/result/grasp/rgbgrasp/gc6d_p62_run01}"
P62_ROOT="$(realpath -m "$P62_ROOT")"
P0P1_ROOT="${P0P1_ROOT:-/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p0p1_run01}"
P4_ROOT="${P4_ROOT:-/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p4_run01}"
P5_ROOT="${P5_ROOT:-/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p5_run02}"
ORIGINAL_ROOT="${ORIGINAL_ROOT:-/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_fp16_run01}"
for old in "$P0P1_ROOT" "$P4_ROOT" "$P5_ROOT" "$ORIGINAL_ROOT"; do
  old="$(realpath -m "$old")"
  if [[ "$P62_ROOT" == "$old" || "$P62_ROOT" == "$old"/* || "$old" == "$P62_ROOT"/* ]]; then
    echo "P62_ROOT must be separate from historical experiment roots" >&2; exit 2
  fi
done
PLAN="$P62_ROOT/plan.json"
LINEAGE="${LINEAGE:-$P62_ROOT/lineage.json}"
DEVICE="${DEVICE:-cuda:0}"
DEPTH_WEIGHT="${DEPTH_WEIGHT:-$P4_ROOT/depth_model/best.pt}"
PRED_WEIGHT="${PRED_WEIGHT:-$P5_ROOT/p51_models/predicted/best.pt}"
CORR_WEIGHT="${CORR_WEIGHT:-$P5_ROOT/p51_models/corrected/best.pt}"
RESUME_ARGS=(); [[ "${RESUME:-0}" != 1 ]] || RESUME_ARGS=(--resume)
case "$STAGE" in
lineage)
  : "${SOURCE_CONFIG:?Use the exact bound NativeRuntime config used by P1/P5}"
  python -m shared_geometry.p62_gc6d lineage-template --config "$SOURCE_CONFIG" \
    --depth-weight "$DEPTH_WEIGHT" --predicted-weight "$PRED_WEIGHT" \
    --corrected-weight "$CORR_WEIGHT" --output "$LINEAGE"
  echo "Review $LINEAGE: fill all source/exposure inventories; approval defaults to false."
  ;;
plan)
  : "${SOURCE_CONFIG:?Exact P1/P5 bound source config required}"
  : "${GC6D_ROOT:?Set the GC6D dataset root}"
  SCENES_JSON="${SCENES_JSON:-$GC6D_ROOT/split_info/grasp_test_scene_ids.json}"
  python -m shared_geometry.p62_gc6d plan --config "$SOURCE_CONFIG" \
    --p1-index "$P0P1_ROOT/p1_cache/validation/index.json" \
    --depth-index "$P4_ROOT/depth_targets/validation/index.json" \
    --depth-weight "$DEPTH_WEIGHT" --predicted-weight "$PRED_WEIGHT" \
    --corrected-weight "$CORR_WEIGHT" --lineage "$LINEAGE" \
    --scenes-json "$SCENES_JSON" --dataset-root "$GC6D_ROOT" \
    --camera "${CAMERA:-realsense-d435}" --scene-count "${SCENE_COUNT:-20}" \
    --annos "${ANNOS:-0,2,3,5,7,9,10,12}" --families "${FAMILIES:-32}" \
    --batch-size "${BATCH_SIZE:-4}" --seed "${SEED:-0}" --output "$PLAN"
  ;;
parity)
  : "${GRASPNET_ROOT:?Original GraspNet root for known-label parity}"
  python -m shared_geometry.p62_gc6d parity-graspnet --plan "$PLAN" \
    --source-index "$ORIGINAL_ROOT/validation_index.json" --graspnet-root "$GRASPNET_ROOT" \
    --frames "${PARITY_FRAMES:-4}" --chunk "${COLLISION_CHUNK:-64}" \
    --output "$P62_ROOT/graspnet_parity.json"
  ;;
smoke|build)
  EXTRA=(); OUT="$P62_ROOT/bank"
  if [[ "$STAGE" == smoke ]]; then EXTRA=(--smoke --limit "${SMOKE_FRAMES:-4}"); OUT="$P62_ROOT/smoke"; fi
  python -m shared_geometry.p62_gc6d build --plan "$PLAN" --output "$OUT" --device "$DEVICE" \
    --parity-report "$P62_ROOT/graspnet_parity.json" --chunk "${COLLISION_CHUNK:-64}" \
    --verify-per-frame "${VERIFY_PER_FRAME:-3}" "${EXTRA[@]}" "${RESUME_ARGS[@]}"
  ;;
audit-smoke)
  python -m shared_geometry.p62_gc6d audit --manifest "$P62_ROOT/smoke/smoke_index.json" \
    --allow-smoke --recompute-geometry
  ;;
audit)
  python -m shared_geometry.p62_gc6d audit --manifest "$P62_ROOT/bank/index.json" --recompute-geometry
  ;;
evaluate)
  python -m shared_geometry.p62_gc6d evaluate --manifest "$P62_ROOT/bank/index.json" \
    --output "$P62_ROOT/reader_evaluation" --device "$DEVICE" \
    --bootstrap "${BOOTSTRAP:-5000}" --seed "${SEED:-0}"
  ;;
*) echo "Unknown stage: $STAGE" >&2; exit 2;;
esac
