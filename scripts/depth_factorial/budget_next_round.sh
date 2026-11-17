#!/usr/bin/env bash
# P0: no-training shared physical point crop-gradient conflicts on CAD2/5.
# P1: accepted ROI/Multi200 vs NEW ROI/Multi600; optionally entirely new CAD.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: budget_next_round.sh check|preflight|gradient|train-roi600|train-multi600|eval-roi600-old|eval-multi600-old|compare-old|register-new|eval-new-roi200|eval-new-multi200|eval-new-roi600|eval-new-multi600|compare-new}"
: "${RUN_ROOT:?Use NEW result directory, do not overwrite accepted 200step results}"
mkdir -p "$RUN_ROOT"
REG="${P1_REGISTRATION:-}"
BASE="${BOUND_CONFIG:-}"
ROOT="${GRASPNET_ROOT:-}"
DEVICE="${DEVICE:-cuda:0}"
OLD_ROI_NEAR="${ROI_NEAR_EVAL:-}"
OLD_ROI_WIDE="${ROI_WIDE_EVAL:-}"
OLD_MULTI_NEAR="${MULTI_NEAR_EVAL:-}"
OLD_MULTI_WIDE="${MULTI_WIDE_EVAL:-}"
OLD_ROI_HEAD="${ROI_WIDE_LAST:-}"
OLD_MULTI_HEAD="${MULTI_WIDE_LAST:-}"
PREFLIGHT="${BUDGET_PREFLIGHT:-$RUN_ROOT/budget_preflight.json}"
HOLDOUT="${NEW_CAD_HOLDOUT:-$RUN_ROOT/new_cad_holdout.json}"

need_registration() {
  : "${P1_REGISTRATION:?Original historical CAD2/5 train, CAD8/9 eval registration}"
  [[ -f "$REG" ]] || { echo "Missing historical registration $REG" >&2;exit 2; }
}

need_old() {
  need_registration
  : "${ROI_NEAR_EVAL:?Original ROI near eval root}"
  : "${ROI_WIDE_EVAL:?Original ROI wide eval root}"
  : "${MULTI_NEAR_EVAL:?Original Multi near eval root}"
  : "${MULTI_WIDE_EVAL:?Original Multi wide eval root}"
  : "${ROI_WIDE_LAST:?Historical ROI-wide accepted LAST checkpoint}"
  : "${MULTI_WIDE_LAST:?Historical Multi-wide accepted LAST checkpoint}"
  for d in "$OLD_ROI_NEAR" "$OLD_ROI_WIDE" "$OLD_MULTI_NEAR" "$OLD_MULTI_WIDE"; do
    [[ -f "$d/summary.json" ]] || { echo "Missing accepted evaluation $d" >&2;exit 2; }
  done
  for f in "$OLD_ROI_HEAD" "$OLD_MULTI_HEAD";do
    [[ -f "$f" ]] || { echo "Missing accepted training LAST $f" >&2;exit 2; }
  done
}

need_model() {
  : "${BOUND_CONFIG:?Strict original MIX50 GVAR36 e15 bound.json}"
  : "${GRASPNET_ROOT:?GraspNet root for frozen original base}"
}

sha_file() { sha256sum "$1" | awk '{print $1}'; }

eval_old_600() {
  local design="$1"
  local model_name;local out
  if [[ "$design" == roi ]];then
    model_name=fixed_roi;out="$RUN_ROOT/eval_old/roi600"
  else
    model_name=multi3;out="$RUN_ROOT/eval_old/multi600"
  fi
  local head="$RUN_ROOT/heads600/$design/last.pt"
  [[ -f "$head" ]] || { echo "Missing 600step LAST $head" >&2;exit 2;}
  python -m diagnostics.depth_factorial.distance_eval_p1 \
    --registration "$REG" --bound-config "$BASE" --graspnet-root "$ROOT" \
    --arm near_mid_far --train-crop-policy "$model_name" \
    --head-checkpoint "$head" --head-sha256 "$(sha_file "$head")" \
    --device "$DEVICE" --output "$out"
}

case "$STAGE" in
check)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
    -s tests -p 'test_depth_budget_conflict.py' -v
  python -m compileall -q diagnostics/depth_factorial \
    tests/test_depth_budget_conflict.py
  bash -n scripts/depth_factorial/budget_next_round.sh
  ;;
preflight)
  need_old
  python -m diagnostics.depth_factorial.distance_budget_preflight \
    --registration "$REG" \
    --roi-near-eval "$OLD_ROI_NEAR" \
    --roi-wide-eval "$OLD_ROI_WIDE" \
    --multi-near-eval "$OLD_MULTI_NEAR" \
    --multi-wide-eval "$OLD_MULTI_WIDE" \
    --roi200-last "$OLD_ROI_HEAD" --roi200-sha "$(sha_file "$OLD_ROI_HEAD")" \
    --multi200-last "$OLD_MULTI_HEAD" --multi200-sha "$(sha_file "$OLD_MULTI_HEAD")" \
    --output "$PREFLIGHT"
  ;;
gradient)
  need_old;need_model
  python -m diagnostics.depth_factorial.gradient_conflict_p0 \
    --registration "$REG" --bound-config "$BASE" --graspnet-root "$ROOT" \
    --roi-wide-last "$OLD_ROI_HEAD" --roi-wide-sha "$(sha_file "$OLD_ROI_HEAD")" \
    --multi-wide-last "$OLD_MULTI_HEAD" --multi-wide-sha "$(sha_file "$OLD_MULTI_HEAD")" \
    --points-per-case 128 --device "$DEVICE" \
    --output "$RUN_ROOT/gradient_p0"
  ;;
train-roi600|train-multi600)
  need_old;need_model
  [[ -f "$PREFLIGHT" ]] || { echo "Run preflight FIRST: $PREFLIGHT" >&2;exit 2; }
  if [[ "$STAGE" == train-roi600 ]];then design=roi;else design=multi3;fi
  python -m diagnostics.depth_factorial.distance_budget_train \
    --registration "$REG" --preflight "$PREFLIGHT" \
    --bound-config "$BASE" --graspnet-root "$ROOT" \
    --design "$design" --device "$DEVICE" \
    --steps 600 --pixels-per-step 256 --lr 0.00002 --seed 20261117 \
    --weight-decay 0.0001 --beta 0.03 --grad-clip 1.0 \
    --output "$RUN_ROOT/heads600/$design"
  ;;
eval-roi600-old|eval-multi600-old)
  need_registration;need_model
  if [[ "$STAGE" == eval-roi600-old ]];then eval_old_600 roi;else eval_old_600 multi3;fi
  ;;
compare-old)
  need_old
  python -m diagnostics.depth_factorial.distance_budget_compare \
    --registration "$REG" --preflight "$PREFLIGHT" \
    --roi200 "$OLD_ROI_WIDE" --multi200 "$OLD_MULTI_WIDE" \
    --roi600 "$RUN_ROOT/eval_old/roi600" \
    --multi600 "$RUN_ROOT/eval_old/multi600" \
    --output "$RUN_ROOT/budget_old_CAD_comparison"
  ;;
register-new)
  need_registration
  : "${NEW_CAD_SOURCES_INDEX:?Real SHA-verified new CAD render sources with category and prior-use fields}"
  : "${NEW_CAD_LINEAGE_REVIEW:?Actual CAD identity/semantic duplication review}"
  python -m diagnostics.depth_factorial.distance_budget_holdout_register \
    --registration "$REG" --sources-index "$NEW_CAD_SOURCES_INDEX" \
    --lineage-review "$NEW_CAD_LINEAGE_REVIEW" \
    --output "$HOLDOUT"
  ;;
eval-new-*)
  need_registration;need_model
  [[ -f "$HOLDOUT" ]] || { echo "Register truly new CAD sources FIRST" >&2;exit 2; }
  mode="${STAGE#eval-new-}"
  opts=()
  case "$mode" in
    roi200) head="$OLD_ROI_HEAD" ;;
    multi200) head="$OLD_MULTI_HEAD" ;;
    roi600) head="$RUN_ROOT/heads600/roi/last.pt" ;;
    multi600) head="$RUN_ROOT/heads600/multi3/last.pt" ;;
    *) echo "Unsupported new-CAD model $mode" >&2;exit 2;;
  esac
  [[ -f "$head" ]] || { echo "Missing new-CAD model checkpoint $head" >&2;exit 2;}
  python -m diagnostics.depth_factorial.distance_budget_holdout_eval \
    --registration "$REG" --holdout "$HOLDOUT" \
    --model "$mode" --bound-config "$BASE" --graspnet-root "$ROOT" \
    --head-checkpoint "$head" --head-sha "$(sha_file "$head")" \
    --device "$DEVICE" --output "$RUN_ROOT/eval_new_CAD/$mode"
  ;;
compare-new)
  need_registration
  [[ -f "$HOLDOUT" ]] || { echo "Register real new CAD FIRST" >&2;exit 2; }
  python -m diagnostics.depth_factorial.distance_budget_compare \
    --registration "$REG" --preflight "$PREFLIGHT" \
    --roi200 "$RUN_ROOT/eval_new_CAD/roi200" \
    --multi200 "$RUN_ROOT/eval_new_CAD/multi200" \
    --roi600 "$RUN_ROOT/eval_new_CAD/roi600" \
    --multi600 "$RUN_ROOT/eval_new_CAD/multi600" \
    --external-holdout --output "$RUN_ROOT/budget_new_CAD_comparison"
  ;;
*)
  echo "Unknown budget stage $STAGE" >&2
  exit 2
  ;;
esac
