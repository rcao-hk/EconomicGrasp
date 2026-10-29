#!/usr/bin/env bash
set -euo pipefail
# P1-0 / P1-1 / P1-2 and fresh GC6D collision-on native AP.
# No trained weights are modified or overwritten.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
THREE_HIST_ROOT="${THREE_HIST_ROOT:-/data/robotarm/EconomicGrasp-triple-historical-20261027}"
MIX_ROOT="${MIX_ROOT:-/data/robotarm/result/grasp/rgbgrasp/triple_historical_20261027}"
POINT_CKPT="${POINT_CKPT:-${MIX_ROOT}/rayrope_point/train/checkpoint_latest.pt}"
GVAR_CKPT="${GVAR_CKPT:-${MIX_ROOT}/rayrope_point_gvar/train/checkpoint_latest.pt}"
TWO_HIST_ROOT="${TWO_HIST_ROOT:-}"
TWO_POINT_CKPT="${TWO_POINT_CKPT:-}"

GC6D_ROOT="${GC6D_ROOT:-/data/robotarm/dataset/GraspClutter6D}"
GRASPNET_ROOT="${GRASPNET_ROOT:-/data/robotarm/dataset/graspnet}"
GNTRANS_ROOT="${GNTRANS_ROOT:-/data/robotarm/dataset/GN-Trans}"
FILTERED_SCENES="${FILTERED_SCENES:-${GC6D_ROOT}/split_info/no_gn_test_overlap/grasp_train_scene_ids_no_graspnet_test_overlap.json}"
P0_RESULTS="${P0_RESULTS:-${MIX_ROOT}/p0_abcd_20261028}"
POINT_OFF_AP="${POINT_OFF_AP:-${P0_RESULTS}/p0_d_gc6d_point/gc6d_official_raw_accuracy.npy}"
GVAR_OFF_AP="${GVAR_OFF_AP:-${P0_RESULTS}/p0_d_gc6d_gvar/gc6d_official_raw_accuracy.npy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MIX_ROOT}/p1_012_collision_20261029}"

GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-3}"
OFFICIAL_EVAL_WORKERS="${OFFICIAL_EVAL_WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
P11_RAW_MAX_SCENES="${P11_RAW_MAX_SCENES:-24}"
P11_FORWARD_FRAMES="${P11_FORWARD_FRAMES:-24}"
P12_TRIPLETS="${P12_TRIPLETS:-3}"
P12_MODELS="${P12_MODELS:-point}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
# Stages: collision (both three-source models), baseline (historical two-source
# checkpoint, both collision on/off), p11_raw, p11_geometry, p12, compare.
STAGES="${STAGES:-collision,p11_raw,p11_geometry,p12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for flag in RESUME DRY_RUN; do
  [[ "${!flag}" == 0 || "${!flag}" == 1 ]] || {
    echo "${flag} must be 0/1" >&2; exit 2; }
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "Provide one GPU id in GPU" >&2;exit 2; }
IFS=, read -r -a REQUESTED <<< "$STAGES"
want() { local v; for v in "${REQUESTED[@]}";do [[ "$v" == "$1" ]] && return 0;done;return 1; }
for v in "${REQUESTED[@]}";do
  case "$v" in collision|baseline|p11_raw|p11_geometry|p12|compare) ;; *)
    echo "Unknown stage $v" >&2; exit 2;;
  esac
done

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'STAGES=%s GPU=%s\n3src root=%s\npoint=%s\ngvar=%s\n2src root=%s\n2src ckpt=%s\nGC6D=%s\nOUT=%s\n' \
     "$STAGES" "$GPU" "$THREE_HIST_ROOT" "$POINT_CKPT" "$GVAR_CKPT" \
     "$TWO_HIST_ROOT" "$TWO_POINT_CKPT" "$GC6D_ROOT" "$OUTPUT_ROOT"
  exit 0
fi
mkdir -p "$OUTPUT_ROOT"
resume_args=()
[[ "$RESUME" == 1 ]] && resume_args=(--resume)

infer_eval() {
  local tag="$1" hist="$2" ckpt="$3" mode="$4"
  [[ -f "$ckpt" && -d "$hist" ]] || { echo "Missing source/ckpt for $tag: $hist $ckpt" >&2;return 2; }
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/mixed_p1_gc6d_collision_eval.py \
    --historical-root "$hist" --checkpoint "$ckpt" \
    --gc6d-root "$GC6D_ROOT" --output-root "$OUTPUT_ROOT/$tag" \
    --mode both --collision "$mode" \
    --collision-thresh 0.01 --collision-voxel-size 0.01 \
    --collision-approach-dist 0.05 \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --eval-workers "$OFFICIAL_EVAL_WORKERS" \
    --accept-preprocessing-privilege \
    "${resume_args[@]}"
}

if want collision;then
  infer_eval three_point "$THREE_HIST_ROOT" "$POINT_CKPT" on
  infer_eval three_gvar "$THREE_HIST_ROOT" "$GVAR_CKPT" on
fi

if want baseline;then
  : "${TWO_HIST_ROOT:?Specify TWO_HIST_ROOT for the original 10%+10% Point checkpoint source tree}"
  : "${TWO_POINT_CKPT:?Specify TWO_POINT_CKPT (not the three-source checkpoint)}"
  infer_eval two_point "$TWO_HIST_ROOT" "$TWO_POINT_CKPT" both
fi

if want p11_raw;then
  "$PYTHON_BIN" tools/mixed_p1_gc6d_coverage.py \
    --mode offline --gc6d-root "$GC6D_ROOT" \
    --filtered-scenes "$FILTERED_SCENES" \
    --output-dir "$OUTPUT_ROOT/p11_raw" \
    --raw-max-scenes "$P11_RAW_MAX_SCENES" \
    --point-filter-mu .4 --max-width .1
fi

if want p11_geometry;then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/mixed_p1_gc6d_coverage.py \
    --mode counterfactual --gc6d-root "$GC6D_ROOT" \
    --filtered-scenes "$FILTERED_SCENES" \
    --graspnet-root "$GRASPNET_ROOT" --gntrans-root "$GNTRANS_ROOT" \
    --historical-root "$THREE_HIST_ROOT" --checkpoint "$POINT_CKPT" \
    --output-dir "$OUTPUT_ROOT/p11_geometry_point" \
    --forward-frames "$P11_FORWARD_FRAMES" --invalid-gt-policy keep_pred
fi

if want p12;then
  IFS=, read -r -a MODELS <<< "$P12_MODELS"
  for variant in "${MODELS[@]}";do
    case "$variant" in
      point) checkpoint="$POINT_CKPT" ;;
      gvar) checkpoint="$GVAR_CKPT" ;;
      *) echo "P12_MODELS must be point,gvar,or both (comma-separated)" >&2;exit 2;;
    esac
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/mixed_p1_source_gradients.py \
      --historical-root "$THREE_HIST_ROOT" --checkpoint "$checkpoint" \
      --gc6d-root "$GC6D_ROOT" --filtered-scenes "$FILTERED_SCENES" \
      --graspnet-root "$GRASPNET_ROOT" --gntrans-root "$GNTRANS_ROOT" \
      --output-dir "$OUTPUT_ROOT/p12_$variant" \
      --triplets "$P12_TRIPLETS" --seed 0 \
      --losses depth,objectness,graspness,view,cdf,width,total \
      --run-train-mode 1
  done
fi

if want compare;then
  : "${POINT_OFF_AP:?Missing P0 Point collision-off official AP path}"
  : "${GVAR_OFF_AP:?Missing P0 GVAR collision-off official AP path}"
  RUNS=(--run "three_point_off=$POINT_OFF_AP"
        --run "three_gvar_off=$GVAR_OFF_AP"
        --run "three_point_on=$OUTPUT_ROOT/three_point/collision_on/official_accuracy.npy"
        --run "three_gvar_on=$OUTPUT_ROOT/three_gvar/collision_on/official_accuracy.npy")
  PAIRS=(--compare three_point_off:three_point_on
         --compare three_gvar_off:three_gvar_on
         --compare three_point_on:three_gvar_on)
  if [[ -f "$OUTPUT_ROOT/two_point/collision_off/official_accuracy.npy" &&
        -f "$OUTPUT_ROOT/two_point/collision_on/official_accuracy.npy" ]];then
    RUNS+=(--run "two_point_off=$OUTPUT_ROOT/two_point/collision_off/official_accuracy.npy"
           --run "two_point_on=$OUTPUT_ROOT/two_point/collision_on/official_accuracy.npy")
    PAIRS+=(--compare two_point_off:three_point_off
           --compare two_point_on:three_point_on
           --compare two_point_off:two_point_on)
  else
    echo "[P1-0] Two-source baseline missing: three-source/filters comparison only; P1-0 data contribution remains unresolved." >&2
  fi
  "$PYTHON_BIN" tools/mixed_p1_gc6d_compare.py \
     "${RUNS[@]}" "${PAIRS[@]}" \
     --gc6d-root "$GC6D_ROOT" \
     --output-dir "$OUTPUT_ROOT/p10_comparisons" --bootstrap 10000 --seed 0
fi

echo "[P1] Requested stages finished. Output: $OUTPUT_ROOT"
