#!/usr/bin/env bash
# Strict frozen-depth camera/crop/distance study, no new training.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh check|smoke|run|analyze}"
: "${RUN_ROOT:?Set a NEW result root; never overwrite accepted GC6D runs}"
mkdir -p "$RUN_ROOT"
case "$STAGE" in
check)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m unittest discover -s tests -p 'test_frozen_depth_response.py' -v
  python -m compileall -q diagnostics/frozen_depth_response tests/test_frozen_depth_response.py
  bash -n scripts/frozen_depth_response/run.sh
  ;;
smoke|run)
  : "${BOUND_CONFIG:?Verified historical MIX50 GVAR e15 bound.json required}"
  : "${GC6D_ROOT:?GC6D dataset root}"
  : "${GC6D_SCENES:?Signed/verified JSON list of the 78 selected D435 test scenes}"
  args=()
  if [[ "$STAGE" == smoke ]]; then
    args+=(--limit "${SMOKE_FRAMES:-2}" --allow-partial)
    output="${SMOKE_OUTPUT:-$RUN_ROOT/smoke}"
  else
    output="${FULL_OUTPUT:-$RUN_ROOT/full}"
  fi
  if [[ "${SAVE_MAPS:-0}" == 1 ]]; then args+=(--save-maps); fi
  if [[ "${ALLOW_PARTIAL:-0}" == 1 ]]; then args+=(--allow-partial); fi
  # Preregistered digital pixel interventions; custom variants must use
  # geometry.INTERVENTIONS and are recorded in the immutable protocol.
  read -r -a variants <<< "${VARIANTS:-zoom_in_115 zoom_out_085 shift_right_08 shift_down_08}"
  if [[ "${#variants[@]}" -lt 1 ]]; then
    echo "At least one digital crop/zoom intervention is required" >&2;exit 2
  fi
  python -m diagnostics.frozen_depth_response.run \
    --bound-config "$BOUND_CONFIG" \
    --dataset-root "$GC6D_ROOT" \
    --scenes-json "$GC6D_SCENES" \
    --device "${DEVICE:-cuda:0}" \
    --output "$output" \
    --expected-scenes "${EXPECTED_SCENES:-78}" \
    --expected-frames "${EXPECTED_FRAMES:-1014}" \
    --max-ann "${MAX_ANN:-13}" \
    --split "${SPLIT:-test}" \
    --mask-mode "${GC6D_MASK_MODE:-workspace_depth}" \
    --workspace-pose-mode "${GC6D_POSE_MODE:-json}" \
    --workspace-outlier "${GC6D_WORKSPACE_OUTLIER:-0.02}" \
    --workspace-depth-trunc "${GC6D_WORKSPACE_TRUNC:-0}" \
    --factor-depth-mode "${GC6D_DEPTH_FACTOR_MODE:-bop}" \
    --intervention-views-per-scene "${PERTURB_VIEWS_PER_SCENE:-2}" \
    --min-foreground-pixels "${MIN_FOREGROUND_PIXELS:-128}" \
    --variants "${variants[@]}" "${args[@]}"
  ;;
analyze)
  root="${FULL_OUTPUT:-$RUN_ROOT/full}"
  [[ -f "$root/manifest.json" ]] || { echo "Missing full manifest" >&2; exit 2; }
  python -m diagnostics.frozen_depth_response.analyze \
    --run "$root" \
    --output "${ANALYSIS_OUTPUT:-$RUN_ROOT/analysis}" \
    --bootstrap "${BOOTSTRAP:-5000}" \
    --min-distance-span "${MIN_SCENE_DISTANCE_SPAN:-0.03}"
  ;;
*)
  echo "Unknown frozen-depth stage: $STAGE" >&2
  exit 2
  ;;
esac
