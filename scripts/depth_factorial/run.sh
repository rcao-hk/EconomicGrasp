#!/usr/bin/env bash
# A: spatially held-out scale/shift/affine ORACLE (diagnostic only).
# B: Blender physical camera translation × frozen depth × crop policy.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh check|a-smoke|a-run|b-render|b-infer|b-analyze|b-aggregate}"
: "${RUN_ROOT:?Set a NEW, independent depth factorial result root}"
mkdir -p "$RUN_ROOT"
case "$STAGE" in
check)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m unittest discover -s tests -p 'test_depth_factorial.py' -v
  python -m compileall -q diagnostics/depth_factorial tests/test_depth_factorial.py
  bash -n scripts/depth_factorial/run.sh
  ;;
a-smoke|a-run)
  : "${GC6D_CACHE_MANIFEST:?Complete signed GC6D 78x13 frozen P1 prediction cache}"
  extra=()
  if [[ "$STAGE" == a-smoke ]]; then
    extra+=(--smoke --limit "${A_SMOKE_FRAMES:-3}")
    output="${A_SMOKE_OUTPUT:-$RUN_ROOT/a_smoke}"
  else
    output="${A_OUTPUT:-$RUN_ROOT/a_oracle}"
  fi
  python -m diagnostics.depth_factorial.oracle_run \
    --cache-manifest "$GC6D_CACHE_MANIFEST" \
    --output "$output" \
    --guard "${SPATIAL_GUARD:-8}" \
    --min-train "${MIN_ORACLE_TRAIN_PIXELS:-512}" \
    --min-test "${MIN_ORACLE_TEST_PIXELS:-128}" \
    --affine-min-std "${AFFINE_MIN_STD_M:-0.003}" \
    --bootstrap "${BOOTSTRAP:-3000}" "${extra[@]}"
  ;;
b-render)
  : "${RENDER_PLAN:?Explicit physical CAD/render plan (no post-hoc image zoom)}"
  command -v blender >/dev/null 2>&1 || {
    echo "Blender not installed; physical renderer cannot be faked with image resize" >&2
    exit 2
  }
  blender --background --python diagnostics/depth_factorial/render_blender.py -- \
    --plan "$RENDER_PLAN" \
    --output "${B_RENDER_OUTPUT:-$RUN_ROOT/b_render}"
  ;;
b-infer)
  : "${BOUND_CONFIG:?Accepted frozen MIX50 GVAR36 e15 bound.json}"
  : "${GRASPNET_ROOT:?Required by strict checkpoint runtime}"
  source="${B_RENDER_OUTPUT:-$RUN_ROOT/b_render}/manifest.json"
  [[ -f "$source" ]] || { echo "Missing validated physical render manifest" >&2;exit 2; }
  extra=()
  if [[ "${REQUIRE_REAL_CAD:-0}" == 1 ]]; then extra+=(--require-cad); fi
  python -m diagnostics.depth_factorial.factorial_run \
    --source-manifest "$source" \
    --bound-config "$BOUND_CONFIG" \
    --graspnet-root "$GRASPNET_ROOT" \
    --device "${DEVICE:-cuda:0}" \
    --fixed-roi-margin "${B_FIXED_ROI_MARGIN:-0.20}" \
    --output "${B_PRED_OUTPUT:-$RUN_ROOT/b_frozen_preds}" "${extra[@]}"
  ;;
b-analyze)
  python -m diagnostics.depth_factorial.factorial_analyze \
    --source-manifest "${B_RENDER_OUTPUT:-$RUN_ROOT/b_render}/manifest.json" \
    --frozen-manifest "${B_PRED_OUTPUT:-$RUN_ROOT/b_frozen_preds}/manifest.json" \
    --min-motion "${B_MIN_REAL_Z_SHIFT:-0.03}" \
    --max-points "${B_ANCHOR_POINTS:-256}" \
    --min-shared "${B_MIN_SHARED:-32}" \
    --output "${B_ANALYSIS_OUTPUT:-$RUN_ROOT/b_analysis}"
  ;;
b-aggregate)
  : "${B_ANALYSES_INDEX:?JSON array of per-CAD B analysis output directories}"
  python -m diagnostics.depth_factorial.factorial_aggregate \
    --analyses-index "$B_ANALYSES_INDEX" \
    --output "${B_AGG_OUTPUT:-$RUN_ROOT/b_aggregate}" \
    --bootstrap "${BOOTSTRAP:-4000}"
  ;;
*)
  echo "Unknown depth factorial stage: $STAGE" >&2
  exit 2
  ;;
esac
