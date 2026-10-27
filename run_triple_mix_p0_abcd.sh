#!/usr/bin/env bash
set -euo pipefail
# P0-A/B/C/D. No training. A/B-offline/C CPU. D one CUDA GPU and official GC6D API.
# Default STAGES=A,B,C; enable D explicitly when ready.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MIX_ROOT="${MIX_ROOT:-/data/robotarm/result/grasp/rgbgrasp/triple_historical_20261027}"
POINT_RUN="${POINT_RUN:-${MIX_ROOT}/rayrope_point}"
GVAR_RUN="${GVAR_RUN:-${MIX_ROOT}/rayrope_point_gvar}"
GRASPNET_ROOT="${GRASPNET_ROOT:-/data/robotarm/dataset/graspnet}"
GNTRANS_RGB_ROOT="${GNTRANS_RGB_ROOT:-/data/robotarm/dataset/GN-Trans}"
GC6D_ROOT="${GC6D_ROOT:-/data/robotarm/dataset/GraspClutter6D}"
GC6D_FILTERED_JSON="${GC6D_FILTERED_JSON:-${GC6D_ROOT}/split_info/no_gn_test_overlap/grasp_train_scene_ids_no_graspnet_test_overlap.json}"
POINT_CKPT="${POINT_CKPT:-${POINT_RUN}/train/checkpoint_latest.pt}"
GVAR_CKPT="${GVAR_CKPT:-${GVAR_RUN}/train/checkpoint_latest.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MIX_ROOT}/p0_abcd_20261027}"
STAGES="${STAGES:-A,B,C}"
GPU="${GPU:-0}"
P0_A_BOOTSTRAP="${P0_A_BOOTSTRAP:-5000}"
VERIFY_DUMP_IDENTITIES="${VERIFY_DUMP_IDENTITIES:-1}"
P0_C_FRAME_STRIDE="${P0_C_FRAME_STRIDE:-20}"
P0_B_MAX_FORWARD_FRAMES="${P0_B_MAX_FORWARD_FRAMES:-24}"
ENABLE_CANDIDATE_EVAL="${ENABLE_CANDIDATE_EVAL:-0}"
ENABLE_FORWARD_PROBE="${ENABLE_FORWARD_PROBE:-0}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-3}"
OFFICIAL_EVAL_WORKERS="${OFFICIAL_EVAL_WORKERS:-12}"
POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_ROOT"
IFS=',' read -ra REQUESTED <<< "$STAGES"
want() { local x; for x in "${REQUESTED[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }
for x in "${REQUESTED[@]}"; do
  [[ "$x" == A || "$x" == B || "$x" == C || "$x" == D ]] || {
    echo "Unknown stage: $x; allowed A,B,C,D" >&2; exit 2;
  }
done
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY] stages=$STAGES"; echo "[DRY] point=$POINT_RUN"; echo "[DRY] gvar=$GVAR_RUN"
  echo "[DRY] GN=$GRASPNET_ROOT; GNTrans=$GNTRANS_RGB_ROOT; GC6D=$GC6D_ROOT"
  echo "[DRY] filtered=$GC6D_FILTERED_JSON; output=$OUTPUT_ROOT"
  exit 0
fi

if want A; then
  A_DUMP_VERIFY=()
  if [[ "$VERIFY_DUMP_IDENTITIES" == "1" ]]; then A_DUMP_VERIFY+=(--verify-dump-identities); fi
  python tools/mixed_p0_a_rank_stats.py \
    --run "point=$POINT_RUN" --run "gvar=$GVAR_RUN" \
    --output-dir "$OUTPUT_ROOT/p0_a_official_ap" \
    --bootstrap "$P0_A_BOOTSTRAP" --seed 0 \
    --ranks 1,5,10,50 "${A_DUMP_VERIFY[@]}"
  if [[ "$ENABLE_CANDIDATE_EVAL" == "1" ]]; then
    python tools/mixed_p0_a_candidate_eval.py \
      --graspnet-root "$GRASPNET_ROOT" \
      --run "point=$POINT_RUN" --run "gvar=$GVAR_RUN" \
      --out-csv "$OUTPUT_ROOT/p0_a_candidate/candidate_rows.csv" \
      --max-scenes-per-split 3 --sample-every 50 --topk 50
    python tools/mixed_p0_a_rank_stats.py \
      --run "point=$POINT_RUN" --run "gvar=$GVAR_RUN" \
      --output-dir "$OUTPUT_ROOT/p0_a_with_candidate_oracle" \
      --bootstrap "$P0_A_BOOTSTRAP" --seed 0 \
      --ranks 1,5,10,50 \
      --candidate-csv "$OUTPUT_ROOT/p0_a_candidate/candidate_rows.csv"
  fi
fi

if want B; then
  python tools/mixed_p0_b_source_supervision.py \
    --p0_gc6d_root "$GC6D_ROOT" \
    --p0_gc6d_split_json "$GC6D_FILTERED_JSON" \
    --p0_output_dir "$OUTPUT_ROOT/p0_b_cache" \
    --p0_mode offline \
    --dataset_root "$GRASPNET_ROOT"
  if [[ "$ENABLE_FORWARD_PROBE" == "1" ]]; then
    for pair in "point:$POINT_CKPT" "gvar:$GVAR_CKPT"; do
      name="${pair%%:*}"
      ckpt="${pair#*:}"
      CUDA_VISIBLE_DEVICES="$GPU" python tools/mixed_p0_b_source_supervision.py \
        --p0_gc6d_root "$GC6D_ROOT" \
        --p0_gntrans_root "$GNTRANS_RGB_ROOT" \
        --p0_gc6d_split_json "$GC6D_FILTERED_JSON" \
        --p0_output_dir "$OUTPUT_ROOT/p0_b_forward_$name" \
        --p0_mode forward --p0_max_forward_frames "$P0_B_MAX_FORWARD_FRAMES" \
        --dataset_root "$GRASPNET_ROOT" --camera realsense \
        --checkpoint_path "$ckpt" \
        --multi_modal --use_cdf --extend_angle --kview_mode A1 \
        --pose_depth_mode "$POSE_DEPTH_MODE"
    done
  fi
fi

if want C; then
  python tools/mixed_p0_c_gc6d_geometry_audit.py \
    --gc6d-root "$GC6D_ROOT" \
    --filtered-scenes "$GC6D_FILTERED_JSON" \
    --out-dir "$OUTPUT_ROOT/p0_c_gc6d_depth_pose" \
    --camera realsense-d435 \
    --frame-stride "$P0_C_FRAME_STRIDE" \
    --workspace-pose-mode json --depth-min .2 --depth-max 1.0
fi

if want D; then
  for pair in "point:$POINT_CKPT" "gvar:$GVAR_CKPT"; do
    name="${pair%%:*}"
    ckpt="${pair#*:}"
    CUDA_VISIBLE_DEVICES="$GPU" python tools/mixed_p0_d_gc6d_cva_eval.py \
      --gc6d-root "$GC6D_ROOT" --checkpoint "$ckpt" \
      --output-dir "$OUTPUT_ROOT/p0_d_gc6d_$name" \
      --camera realsense-d435 --split test \
      --pose-depth-mode "$POSE_DEPTH_MODE" \
      --sample-fraction 1.0 --batch-size "$INFER_BATCH_SIZE" \
      --workers "$NUM_WORKERS" --eval-workers "$OFFICIAL_EVAL_WORKERS" \
      --collision-thresh 0 \
      --do-infer --do-eval --skip-existing
  done
fi

echo "[P0-ABCD] requested stages complete. Results: $OUTPUT_ROOT"
