#!/usr/bin/env bash
# Same branch GC6D physical depth follow-up: P0 matched-ray crop audit +
# P1 2×2 physical-distance coverage × training-crop source experiment.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: crosscrop_next_round.sh check|preflight|p0-before|train-near-multi|train-wide-multi|eval-near-multi|eval-wide-multi|p0-after|compare}"
: "${RUN_ROOT:?Set a NEW output directory; never overwrite historical ROI heads/AP}"
ROOT="$RUN_ROOT"
mkdir -p "$ROOT"
DEVICE="${DEVICE:-cuda:0}"
PREFLIGHT="${CROSSCROP_PREFLIGHT:-$ROOT/roi_preflight.json}"
REG="${P1_REGISTRATION:-}"
ROI_NEAR="${ROI_EVAL_NEAR:-}"
ROI_WIDE="${ROI_EVAL_WIDE:-}"
NATIVE="${NATIVE_EVAL:-}"
MIX_NEAR="$ROOT/crosscrop_heads/near_only"
MIX_WIDE="$ROOT/crosscrop_heads/near_mid_far"
EVAL_MIX_NEAR="$ROOT/crosscrop_eval/near_only"
EVAL_MIX_WIDE="$ROOT/crosscrop_eval/near_mid_far"

need_inputs() {
  : "${P1_REGISTRATION:?Use EXACT accepted CAD registration from historical ROI study}"
  : "${ROI_EVAL_NEAR:?Original accepted fixed_roi near_only evaluation root}"
  : "${ROI_EVAL_WIDE:?Original accepted fixed_roi near_mid_far evaluation root}"
  : "${NATIVE_EVAL:?Original accepted native evaluation root}"
  for d in "$ROI_NEAR" "$ROI_WIDE" "$NATIVE";do
    [[ -f "$d/summary.json" ]] || { echo "Missing accepted eval summary: $d" >&2;exit 2; }
  done
}

write_index() {
  local path="$1"
  local include_mix="$2"
  if [[ -e "$path" ]];then
    echo "Refuse overwrite eval manifest: $path" >&2
    exit 2
  fi
  python - "$path" "$include_mix" "$NATIVE" "$ROI_NEAR" "$ROI_WIDE" \
    "$EVAL_MIX_NEAR" "$EVAL_MIX_WIDE" <<'PY'
import json,sys
from pathlib import Path
path,with_mix,native,near,wide,mnear,mwide=sys.argv[1:]
index={"native":native,"roi_near":near,"roi_wide":wide}
if with_mix=="yes":
    index.update(multi_near=mnear,multi_wide=mwide)
for label,folder in index.items():
    if not (Path(folder)/"summary.json").is_file():
        raise SystemExit(f"Missing accepted {label} evaluation: {folder}")
Path(path).write_text(json.dumps(index,sort_keys=True,indent=2)+"\n")
PY
}

case "$STAGE" in
check)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
    -s tests -p 'test_distance_crosscrop.py' -v
  python -m compileall -q diagnostics/depth_factorial \
    tests/test_distance_crosscrop.py
  bash -n scripts/depth_factorial/crosscrop_next_round.sh
  ;;
preflight)
  need_inputs
  python -m diagnostics.depth_factorial.distance_crosscrop_preflight \
    --registration "$REG" --roi-near "$ROI_NEAR" \
    --roi-wide "$ROI_WIDE" --output "$PREFLIGHT"
  ;;
p0-before|p0-after)
  need_inputs
  stage_index="$ROOT/${STAGE}_run_index.json"
  if [[ "$STAGE" == p0-after ]];then
    write_index "$stage_index" yes
    out="$ROOT/p0_post_multi"
  else
    write_index "$stage_index" no
    out="$ROOT/p0_existing_roi"
  fi
  python -m diagnostics.depth_factorial.crop_consistency_audit \
    --registration "$REG" --eval-index "$stage_index" --output "$out" \
    --points-per-frame 256 --min-points 32
  ;;
train-near-multi|train-wide-multi)
  need_inputs
  : "${BOUND_CONFIG:?Strict historical MIX50 GVAR e15 bound checkpoint configuration}"
  : "${GRASPNET_ROOT:?GraspNet initialization dataset root}"
  [[ -f "$PREFLIGHT" ]] || {
    echo "Run crosscrop_next_round.sh preflight successfully FIRST" >&2;exit 2;}
  if [[ "$STAGE" == train-near-multi ]];then
    arm=near_only;out="$MIX_NEAR"
  else
    arm=near_mid_far;out="$MIX_WIDE"
  fi
  python -m diagnostics.depth_factorial.distance_crosscrop_train \
    --registration "$REG" --preflight "$PREFLIGHT" \
    --bound-config "$BOUND_CONFIG" --graspnet-root "$GRASPNET_ROOT" \
    --arm "$arm" --steps 200 --pixels-per-step 256 \
    --seed 20261117 --lr 0.00002 --weight-decay 0.0001 \
    --beta 0.03 --grad-clip 1.0 --device "$DEVICE" --output "$out"
  ;;
eval-near-multi|eval-wide-multi)
  need_inputs
  : "${BOUND_CONFIG:?}" "${GRASPNET_ROOT:?}"
  if [[ "$STAGE" == eval-near-multi ]];then
    arm=near_only;head="$MIX_NEAR/last.pt";out="$EVAL_MIX_NEAR"
  else
    arm=near_mid_far;head="$MIX_WIDE/last.pt";out="$EVAL_MIX_WIDE"
  fi
  [[ -f "$head" ]] || { echo "Missing multi3 LAST checkpoint: $head" >&2;exit 2; }
  sha="$(sha256sum "$head" | awk '{print $1}')"
  python -m diagnostics.depth_factorial.distance_eval_p1 \
    --registration "$REG" --bound-config "$BOUND_CONFIG" \
    --graspnet-root "$GRASPNET_ROOT" \
    --arm "$arm" --head-checkpoint "$head" --head-sha256 "$sha" \
    --train-crop-policy multi3 --device "$DEVICE" --output "$out"
  ;;
compare)
  need_inputs
  python -m diagnostics.depth_factorial.distance_crosscrop_compare \
    --registration "$REG" \
    --roi-near "$ROI_NEAR" --roi-wide "$ROI_WIDE" \
    --multi-near "$EVAL_MIX_NEAR" --multi-wide "$EVAL_MIX_WIDE" \
    --output "$ROOT/p1_factorial_comparison" --bootstrap 4000
  ;;
*)
  echo "Unknown next-round stage $STAGE" >&2;exit 2
  ;;
esac
