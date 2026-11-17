#!/usr/bin/env bash
# P0: constant vs constrained affine, same held-out pixels.
# P1: corrected B scale cues + optional preregistered near-only/near+far
# head-only training with disjoint CAD evaluation objects.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: followup_p0_p1.sh check|p0-smoke|p0-run|p1-cues|p1-register|p1-train-near|p1-train-wide|p1-eval-native|p1-eval-near|p1-eval-wide|p1-compare}"
: "${RUN_ROOT:?Use a NEW output root, never prior GC6D/CAD results}"
mkdir -p "$RUN_ROOT"
DEVICE="${DEVICE:-cuda:0}"
case "$STAGE" in
check)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m unittest discover -s tests -p 'test_depth_factorial_followup.py' -v
  python -m compileall -q diagnostics/depth_factorial tests/test_depth_factorial_followup.py
  bash -n scripts/depth_factorial/followup_p0_p1.sh
  ;;
p0-smoke|p0-run)
  : "${GC6D_CACHE_MANIFEST:?Use accepted 78x13 P1 GC6D frozen depth cache}"
  args=()
  if [[ "$STAGE" == p0-smoke ]]; then
    args+=(--smoke --limit "${P0_SMOKE_FRAMES:-3}")
    out="${P0_SMOKE_OUTPUT:-$RUN_ROOT/p0_smoke}"
  else
    out="${P0_OUTPUT:-$RUN_ROOT/p0_oracle}"
  fi
  python -m diagnostics.depth_factorial.oracle_followup_p0 \
    --cache-manifest "$GC6D_CACHE_MANIFEST" \
    --output "$out" \
    --guard 8 --min-train 512 --min-test 128 \
    --affine-min-std 0.003 \
    --smooth-max 0.002 --edge-min 0.006 \
    --bootstrap "${BOOTSTRAP:-3000}" "${args[@]}"
  ;;
p1-cues)
  : "${CORRECTED_B_INDEX:?Explicit SHA and acceptance-checked CAD corrected B inventory}"
  python -m diagnostics.depth_factorial.scale_cues_p1 \
    --index "$CORRECTED_B_INDEX" \
    --output "${P1_CUES_OUTPUT:-$RUN_ROOT/p1_scale_cues}"
  ;;
p1-register)
  : "${TRAIN_CAD_INDEX:?Explicit array of signed CAD training render manifests}"
  : "${EVAL_CAD_INDEX:?Explicit array of disjoint CAD evaluation render manifests}"
  : "${CAD_OVERLAP_REVIEW:?Audited object duplication / CAD variant evidence}"
  python -m diagnostics.depth_factorial.distance_split_p1 \
    --train-sources "$TRAIN_CAD_INDEX" \
    --eval-sources "$EVAL_CAD_INDEX" \
    --object-overlap-review "$CAD_OVERLAP_REVIEW" \
    --output "${P1_REGISTRATION:-$RUN_ROOT/p1_cad_split.json}"
  ;;
p1-train-near|p1-train-wide)
  : "${P1_REGISTRATION:?Saved preregistered train/eval CAD split JSON}"
  : "${BOUND_CONFIG:?Strict MIX50 e15 checkpoint bound config}"
  : "${GRASPNET_ROOT:?GraspNet root for strict model initialization}"
  if [[ "$STAGE" == p1-train-near ]]; then
    arm=near_only
  else
    arm=near_mid_far
  fi
  python -m diagnostics.depth_factorial.distance_train_p1 \
    --registration "$P1_REGISTRATION" \
    --bound-config "$BOUND_CONFIG" \
    --graspnet-root "$GRASPNET_ROOT" \
    --arm "$arm" \
    --crop-policy "${TRAIN_CROP_POLICY:-fixed_roi}" \
    --steps "${STEPS:-200}" \
    --pixels-per-step "${PIXELS_PER_STEP:-256}" \
    --lr "${LR:-0.00002}" \
    --seed "${SEED:-20261117}" \
    --device "$DEVICE" \
    --output "$RUN_ROOT/p1_heads/$arm"
  ;;
p1-eval-native|p1-eval-near|p1-eval-wide)
  : "${P1_REGISTRATION:?}"
  : "${BOUND_CONFIG:?}"
  : "${GRASPNET_ROOT:?}"
  opts=()
  arm=native
  if [[ "$STAGE" == p1-eval-near ]]; then
    arm=near_only
  elif [[ "$STAGE" == p1-eval-wide ]]; then
    arm=near_mid_far
  fi
  if [[ "$arm" != native ]]; then
    checkpoint="$RUN_ROOT/p1_heads/$arm/last.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing trained last.pt: $checkpoint" >&2; exit 2; }
    # Hash is resolved from the actual saved bytes, never predicted in advance.
    sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
    opts+=(--head-checkpoint "$checkpoint" --head-sha256 "$sha")
  fi
  python -m diagnostics.depth_factorial.distance_eval_p1 \
    --registration "$P1_REGISTRATION" \
    --bound-config "$BOUND_CONFIG" --graspnet-root "$GRASPNET_ROOT" \
    --arm "$arm" \
    --train-crop-policy "${TRAIN_CROP_POLICY:-fixed_roi}" \
    --device "$DEVICE" \
    --output "$RUN_ROOT/p1_eval/$arm" "${opts[@]}"
  ;;
p1-compare)
  python -m diagnostics.depth_factorial.distance_compare_p1 \
    --native "$RUN_ROOT/p1_eval/native" \
    --near "$RUN_ROOT/p1_eval/near_only" \
    --wide "$RUN_ROOT/p1_eval/near_mid_far" \
    --bootstrap "${BOOTSTRAP:-4000}" \
    --output "$RUN_ROOT/p1_comparison"
  ;;
*)
  echo "Unknown follow-up stage: $STAGE" >&2
  exit 2
  ;;
esac
