#!/usr/bin/env bash
# P0: train Original/GNTrans vs GC6D distance/crop/effective-focal audit.
# P1: occlusion-validated SAME 3D surface across GC6D's 13 D435 views.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh check|p1-cache-smoke|p1-cache|p0-original|p0-gntrans|p0-gc6d|p0-analyze|p1-track-smoke|p1-track|p1-analyze}"
: "${RUN_ROOT:?Use a NEW dedicated directory; never overwrite accepted GC6D runs}"
mkdir -p "$RUN_ROOT"
DEVICE="${DEVICE:-cuda:0}"
CACHE_ROOT="${P1_CACHE_ROOT:-$RUN_ROOT/p1_cache}"
TRACK_ROOT="${P1_TRACK_ROOT:-$RUN_ROOT/p1_tracks}"
case "$STAGE" in
check)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m unittest discover -s tests -p 'test_depth_followup.py' -v
  python -m compileall -q diagnostics/depth_followup tests/test_depth_followup.py
  bash -n scripts/depth_followup/run.sh
  ;;
p1-cache-smoke|p1-cache)
  : "${BOUND_CONFIG:?Verified historic MIX50 GVAR e15 bound.json required}"
  : "${GRASPNET_ROOT:?GraspNet root for strict frozen runtime}"
  : "${GC6D_ROOT:?GC6D root}"
  : "${GC6D_SCENES:?EXACT accepted 78 GC6D D435 scene IDs JSON, never guess}"
  args=()
  if [[ "$STAGE" == p1-cache-smoke ]]; then
    args+=(--smoke --limit 13)
    out="${P1_SMOKE_CACHE_ROOT:-$RUN_ROOT/p1_smoke_cache}"
  else
    out="$CACHE_ROOT"
  fi
  if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
  python -m diagnostics.depth_followup.p1_cache \
    --bound-config "$BOUND_CONFIG" \
    --graspnet-root "$GRASPNET_ROOT" \
    --gc6d-root "$GC6D_ROOT" --gc6d-scenes "$GC6D_SCENES" \
    --device "$DEVICE" --output "$out" "${args[@]}"
  ;;
p0-original|p0-gntrans|p0-gc6d)
  : "${BOUND_CONFIG:?}" "${GRASPNET_ROOT:?}"
  domain="${STAGE#p0-}"
  args=(--domain "$domain" --bound-config "$BOUND_CONFIG" \
        --graspnet-root "$GRASPNET_ROOT" \
        --device "$DEVICE" \
        --train-annos "${TRAIN_ANNOS:-0,64,128,192}")
  if [[ "$domain" == gntrans ]]; then
    : "${GNTRANS_ROOT:?GNTrans RGB root required}"
    args+=(--gntrans-root "$GNTRANS_ROOT")
  fi
  if [[ "$domain" == gc6d ]]; then
    : "${GC6D_ROOT:?}" "${GC6D_SCENES:?}"
    args+=(--gc6d-root "$GC6D_ROOT" --gc6d-scenes "$GC6D_SCENES")
    if [[ -f "$CACHE_ROOT/manifest.json" && "${REUSE_P1_CACHE:-1}" == 1 ]]; then
      args+=(--reuse-gc6d-cache "$CACHE_ROOT/manifest.json")
    fi
  fi
  if [[ "${SMOKE_FRAMES:-0}" != 0 ]]; then
    args+=(--smoke --max-frames "$SMOKE_FRAMES")
    out="$RUN_ROOT/p0_smoke_$domain"
  else
    out="$RUN_ROOT/p0_$domain"
  fi
  python -m diagnostics.depth_followup.p0_collect "\${args[@]}" --output "$out"
  ;;
p0-analyze)
  python -m diagnostics.depth_followup.p0_analyze \
    --original "$RUN_ROOT/p0_original" \
    --gntrans "$RUN_ROOT/p0_gntrans" \
    --gc6d "$RUN_ROOT/p0_gc6d" \
    --output "$RUN_ROOT/p0_report" \
    --bootstrap "${BOOTSTRAP:-3000}"
  ;;
p1-track-smoke|p1-track)
  args=()
  if [[ "$STAGE" == p1-track-smoke ]]; then
    args+=(--smoke --limit-scenes 1)
    root="${P1_SMOKE_TRACK_ROOT:-$RUN_ROOT/p1_smoke_tracks}"
    cache="${P1_SMOKE_CACHE_ROOT:-$RUN_ROOT/p1_smoke_cache}"
  else
    root="$TRACK_ROOT"
    cache="$CACHE_ROOT"
  fi
  if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
  python -m diagnostics.depth_followup.p1_track \
    --cache-manifest "$cache/manifest.json" \
    --points-per-object "${POINTS_PER_OBJECT:-64}" \
    --max-points "${MAX_ANCHOR_POINTS:-512}" \
    --visibility-abs-tol "${VISIBILITY_ABS_TOL:-0.008}" \
    --visibility-rel-tol "${VISIBILITY_REL_TOL:-0.012}" \
    --xyz-roundtrip-tol "${XYZ_ROUNDTRIP_TOL:-0.018}" \
    --output "$root" "${args[@]}"
  ;;
p1-analyze)
  python -m diagnostics.depth_followup.p1_analyze \
    --tracks "$TRACK_ROOT" \
    --output "$RUN_ROOT/p1_report" \
    --min-motion "${MIN_TRACK_Z_SHIFT:-0.03}" \
    --min-scene-pairs "${MIN_SCENE_PAIRS:-40}" \
    --bootstrap "${BOOTSTRAP:-4000}"
  ;;
*)
  echo "Unknown stage $STAGE" >&2
  exit 2
  ;;
esac
