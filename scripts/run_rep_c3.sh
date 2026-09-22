#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

REP_A_WORK_ROOT=${REP_A_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness}
CACHE_ROOT=${CACHE_ROOT:-$REP_A_WORK_ROOT/cache}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_c3_real_errors}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}
GPUS=${GPUS:-0}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
SCORERS=${SCORERS:-A0,A1}
RESIDUAL_CASES=${RESIDUAL_CASES:-residual_full:0.5,residual_full:1.0,residual_local:0.5,residual_local:1.0}
# Optional comma-separated NAME=ROOT pairs, e.g.:
# MATERIALS=mataug=/data/robotarm/graspnet_material_aug
MATERIALS=${MATERIALS:-}
QUERY_LIMIT=${QUERY_LIMIT:-64}
MAX_FRAMES=${MAX_FRAMES:-0}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
SCORE_QUERY_CHUNK=${SCORE_QUERY_CHUNK:-64}
EVAL_SHARDS=${EVAL_SHARDS:-1}
EVAL_CHUNK=${EVAL_CHUNK:-128}
LABEL_SCOPE=${LABEL_SCOPE:-selected}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
FRAME_STRIDE=${FRAME_STRIDE:-10}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-4}
PHASES=${PHASES:-infer,evaluate,summary}
RESUME=${RESUME:-1}
REPAIR_CORRUPT=${REPAIR_CORRUPT:-0}
MIN_HOST_FREE_GIB=${MIN_HOST_FREE_GIB:-4}

source "$ROOT_DIR/scripts/rep_followup_jobs.sh"
parse_gpus
IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a TESTS <<< "$SPLITS"
IFS=',' read -r -a MODELS <<< "$SCORERS"
IFS=',' read -r -a MATERIAL_ARRAY <<< "$MATERIALS"

extra=()
[[ "$RESUME" == 1 ]] && extra+=(--resume)
[[ "$REPAIR_CORRUPT" == 1 ]] && extra+=(--repair-corrupt)

for phase in "${STEPS[@]}"; do
  case "$phase" in
    infer)
      scorer_args=()
      for name in "${MODELS[@]}"; do
        case "$name" in
          A0|A1|A2|A3) ck="$REP_A_WORK_ROOT/train/$name/checkpoint_best.pt" ;;
          *) echo "Rep-C3 currently supports Rep-A scorers; unknown $name" >&2; exit 2 ;;
        esac
        [[ -f "$ck" ]] || { echo "Missing scorer $ck" >&2; exit 2; }
        scorer_args+=(--scorer "$name=$ck")
      done
      material_args=()
      if [[ -n "$MATERIALS" ]]; then
        for item in "${MATERIAL_ARRAY[@]}"; do material_args+=(--material "$item"); done
      fi
      for split in "${TESTS[@]}"; do
        for shard in "${!GPU_IDS[@]}"; do
          launch "rep-c3/infer/$split/$shard" "${GPU_IDS[$shard]}" "$WORK_ROOT/logs/infer_${split}_${shard}.log"             "$ROOT_DIR/infer_rep_c3.py"             --dataset-root "$DATASET_ROOT" --stage1-checkpoint "$STAGE1_CKPT"             --cache-root "$CACHE_ROOT" --output-root "$WORK_ROOT" --split "$split"             --pose-depth-mode "$POSE_DEPTH_MODE" --residual-cases "$RESIDUAL_CASES"             "--offsets-mm=$OFFSETS_MM" --query-limit "$QUERY_LIMIT"             --score-query-chunk "$SCORE_QUERY_CHUNK" --max-frames "$MAX_FRAMES"             --shard-id "$shard" --num-shards "${#GPU_IDS[@]}"             --min-host-free-gib "$MIN_HOST_FREE_GIB"             "${scorer_args[@]}" "${material_args[@]}" "${extra[@]}"
        done
        wait_wave
      done
      ;;
    evaluate)
      for split in "${TESTS[@]}"; do
        for ((s=0;s<EVAL_SHARDS;s++)); do
          launch "rep-c3/eval/$split/$s" "" "$WORK_ROOT/logs/eval_${split}_${s}.log"             "$ROOT_DIR/evaluate_rep_fullpath.py"             --dataset-root "$DATASET_ROOT" --work-root "$WORK_ROOT" --split "$split"             --shard-id "$s" --num-shards "$EVAL_SHARDS" --eval-chunk "$EVAL_CHUNK"             --fc-mode "$FC_MODE" --verify-n "$VERIFY_N" --label-scope "$LABEL_SCOPE"             --min-host-free-gib "$MIN_HOST_FREE_GIB" "${extra[@]}"
        done
        wait_wave
      done
      ;;
    summary)
      "$PYTHON_BIN" "$ROOT_DIR/summarize_rep_c3.py"         --root "$WORK_ROOT" --splits "$SPLITS"
      ;;
    official)
      off=(); [[ "$RESUME" == 1 ]] && off+=(--resume)
      for split in "${TESTS[@]}"; do
        "$PYTHON_BIN" "$ROOT_DIR/eval_rep_fullpath_official.py"           --dataset-root "$DATASET_ROOT" --work-root "$WORK_ROOT" --split "$split"           --frame-stride "$FRAME_STRIDE" --workers "$OFFICIAL_WORKERS" "${off[@]}"
      done
      ;;
    *)
      echo "Unknown Rep-C3 phase $phase" >&2; exit 2 ;;
  esac
done
