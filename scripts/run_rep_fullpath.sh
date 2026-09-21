#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
REP_A_WORK_ROOT=${REP_A_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness}
REP_B_WORK_ROOT=${REP_B_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_b_hypothesis_image}
ATTR_WORK_ROOT=${ATTR_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_b0_attribution}
CACHE_ROOT=${CACHE_ROOT:-$REP_A_WORK_ROOT/cache}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}
GPUS=${GPUS:-0}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
PHASES=${PHASES:-infer,evaluate,summary}
SCORERS=${SCORERS:-A0,A1,B0}
MODES=${MODES:-reader_only,anchor_only,joint}
TEST_CASES=${TEST_CASES:-nominal,bias:-10,bias:10,bias:-20,bias:20,bias:-40,bias:40,scale:-0.05,scale:0.05,smooth:20}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
QUERY_LIMIT=${QUERY_LIMIT:-64}
SCORE_QUERY_CHUNK=${SCORE_QUERY_CHUNK:-64}
MAX_FRAMES=${MAX_FRAMES:-0}
TEST_SEED=${TEST_SEED:-2026}
EVAL_SHARDS=${EVAL_SHARDS:-1}
EVAL_CHUNK=${EVAL_CHUNK:-128}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
LABEL_SCOPE=${LABEL_SCOPE:-selected}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-1}
FRAME_STRIDE=${FRAME_STRIDE:-10}
RESUME=${RESUME:-1}
REPAIR_CORRUPT=${REPAIR_CORRUPT:-0}
MIN_HOST_FREE_GIB=${MIN_HOST_FREE_GIB:-4}
source "$ROOT_DIR/scripts/rep_followup_jobs.sh"
parse_gpus
[[ "$EVAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EVAL_SHARDS' >&2; exit 2; }
IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a TESTS <<< "$SPLITS"
IFS=',' read -r -a MODELS <<< "$SCORERS"
extra=(); [[ "$RESUME" == 1 ]] && extra+=(--resume)
[[ "$REPAIR_CORRUPT" == 1 ]] && extra+=(--repair-corrupt)
for phase in "${STEPS[@]}"; do
  case "$phase" in
    infer)
      model_args=()
      for name in "${MODELS[@]}"; do
        case "$name" in
          A0|A1|A2|A3) ck="$REP_A_WORK_ROOT/train/$name/checkpoint_best.pt" ;;
          B0|B1|B2) ck="$REP_B_WORK_ROOT/train/$name/checkpoint_best.pt" ;;
          no_image|independent_rgb|full) ck="$ATTR_WORK_ROOT/train/$name/checkpoint_best.pt" ;;
          stage1_native|'') continue ;;
          *) echo "Unknown scorer $name" >&2; exit 2 ;;
        esac
        [[ -f "$ck" ]] || { echo "Missing scorer $ck" >&2; exit 2; }
        model_args+=(--scorer "$name=$ck")
      done
      for split in "${TESTS[@]}"; do
        for shard in "${!GPU_IDS[@]}"; do
          launch "fullpath/infer/$split/$shard" "${GPU_IDS[$shard]}" "$WORK_ROOT/logs/infer_${split}_${shard}.log" \
            "$ROOT_DIR/infer_rep_fullpath.py" --dataset-root "$DATASET_ROOT" --stage1-checkpoint "$STAGE1_CKPT" \
            --cache-root "$CACHE_ROOT" --output-root "$WORK_ROOT" --split "$split" --pose-depth-mode "$POSE_DEPTH_MODE" \
            "--cases=$TEST_CASES" --modes "$MODES" "--offsets-mm=$OFFSETS_MM" --query-limit "$QUERY_LIMIT" \
            --score-query-chunk "$SCORE_QUERY_CHUNK" --seed "$TEST_SEED" --max-frames "$MAX_FRAMES" \
            --shard-id "$shard" --num-shards "${#GPU_IDS[@]}" --min-host-free-gib "$MIN_HOST_FREE_GIB" \
            "${model_args[@]}" "${extra[@]}"
        done
        wait_wave
      done ;;
    evaluate)
      for split in "${TESTS[@]}"; do
        for ((s=0;s<EVAL_SHARDS;s++)); do
          # CPU-only exact evaluator; default one scene worker to protect RAM.
          launch "fullpath/evaluate/$split/$s" "" "$WORK_ROOT/logs/eval_${split}_${s}.log" \
            "$ROOT_DIR/evaluate_rep_fullpath.py" --dataset-root "$DATASET_ROOT" --work-root "$WORK_ROOT" \
            --split "$split" --shard-id "$s" --num-shards "$EVAL_SHARDS" --eval-chunk "$EVAL_CHUNK" \
            --max-frames 0 --fc-mode "$FC_MODE" --verify-n "$VERIFY_N" --label-scope "$LABEL_SCOPE" \
            --min-host-free-gib "$MIN_HOST_FREE_GIB" "${extra[@]}"
        done
        wait_wave
      done ;;
    summary) "$PYTHON_BIN" "$ROOT_DIR/summarize_rep_followup.py" --kind fullpath --root "$WORK_ROOT" --splits "$SPLITS" ;;
    official)
      off=(); [[ "$RESUME" == 1 ]] && off+=(--resume)
      for split in "${TESTS[@]}"; do
        "$PYTHON_BIN" "$ROOT_DIR/eval_rep_fullpath_official.py" --dataset-root "$DATASET_ROOT" --work-root "$WORK_ROOT" \
          --split "$split" --frame-stride "$FRAME_STRIDE" --workers "$OFFICIAL_WORKERS" "${off[@]}"
      done ;;
    *) echo "Unknown phase $phase" >&2; exit 2 ;;
  esac
done
