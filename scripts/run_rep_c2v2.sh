#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

REP_A_WORK_ROOT=${REP_A_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness}
CACHE_ROOT=${CACHE_ROOT:-$REP_A_WORK_ROOT/cache}
A1_CHECKPOINT=${A1_CHECKPOINT:-$REP_A_WORK_ROOT/train/A1/checkpoint_best.pt}
SOURCE_ROOT=${SOURCE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_formal_joint_ap}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_fullpath_verifier_v2}
TRAIN_CACHE_ROOT=${TRAIN_CACHE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_fullpath_verifier/train_cache}
TRAIN_DIR=${TRAIN_DIR:-$WORK_ROOT/train}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}

GPUS=${GPUS:-0,1,2}
PHASES=${PHASES:-train,test}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
CASES=${CASES:-nominal,bias:-20,bias:20}
EPOCHS=${EPOCHS:-12}
QUERY_LIMIT=${QUERY_LIMIT:-64}
MOVE_LIMIT=${MOVE_LIMIT:-16}
VAL_MOVE_LIMIT=${VAL_MOVE_LIMIT:-0}
VAL_QUERY_CHUNK=${VAL_QUERY_CHUNK:-128}
VAL_EVERY=${VAL_EVERY:-1}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FILES=${MAX_VAL_FILES:-0}
MAX_TEST_FILES=${MAX_TEST_FILES:-0}
MINE_MAX_FRAMES=${MINE_MAX_FRAMES:-0}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
EVAL_CHUNK=${EVAL_CHUNK:-128}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
MIN_HOST_FREE_GIB=${MIN_HOST_FREE_GIB:-6}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-16}
FRAME_STRIDE=${FRAME_STRIDE:-10}
RESUME=${RESUME:-1}
REPAIR_CORRUPT=${REPAIR_CORRUPT:-0}

source "$ROOT_DIR/scripts/rep_followup_jobs.sh"
parse_gpus

IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a TESTS <<< "$SPLITS"

extra_mine=()
[[ "$RESUME" == 1 ]] && extra_mine+=(--resume)
[[ "$REPAIR_CORRUPT" == 1 ]] && extra_mine+=(--repair-corrupt)

for phase in "${STEPS[@]}"; do
  case "$phase" in
    mine)
      for shard in "${!GPU_IDS[@]}"; do
        launch "rep-c2v2/mine/$shard" "${GPU_IDS[$shard]}" "$WORK_ROOT/logs/mine_${shard}.log"           "$ROOT_DIR/prepare_rep_c2v2_train.py"           --dataset-root "$DATASET_ROOT"           --stage1-checkpoint "$STAGE1_CKPT"           --cache-root "$CACHE_ROOT"           --a1-checkpoint "$A1_CHECKPOINT"           --output-root "$TRAIN_CACHE_ROOT"           --pose-depth-mode "$POSE_DEPTH_MODE"           "--offsets-mm=$OFFSETS_MM"           --query-limit "$QUERY_LIMIT"           --move-limit "$MOVE_LIMIT"           --eval-chunk "$EVAL_CHUNK"           --fc-mode "$FC_MODE"           --verify-n "$VERIFY_N"           --max-frames "$MINE_MAX_FRAMES"           --min-host-free-gib "$MIN_HOST_FREE_GIB"           --shard-id "$shard"           --num-shards "${#GPU_IDS[@]}"           "${extra_mine[@]}"
      done
      wait_wave
      ;;
    train)
      args=(
        "$ROOT_DIR/train_rep_c2v2.py"
        --cache-root "$CACHE_ROOT"
        --train-cache-root "$TRAIN_CACHE_ROOT"
        --validation-source-root "$SOURCE_ROOT"
        --output-dir "$TRAIN_DIR"
        --epochs "$EPOCHS"
        --val-cases "$CASES"
        --val-move-limit "$VAL_MOVE_LIMIT"
        --val-query-chunk "$VAL_QUERY_CHUNK"
        --val-every "$VAL_EVERY"
        --max-train-frames "$MAX_TRAIN_FRAMES"
        --max-val-files "$MAX_VAL_FILES"
      )
      [[ "$RESUME" == 1 ]] && args+=(--resume)
      launch "rep-c2v2/train" "${GPU_IDS[0]}" "$WORK_ROOT/logs/train.log" "${args[@]}"
      wait_wave
      ;;
    test)
      slot=0
      for split in "${TESTS[@]}"; do
        launch "rep-c2v2/test/$split" "${GPU_IDS[$slot]}" "$WORK_ROOT/logs/test_${split}.log"           "$ROOT_DIR/test_rep_c2v2.py"           --source-root "$SOURCE_ROOT"           --cache-root "$CACHE_ROOT"           --c2v2-dir "$TRAIN_DIR"           --output-root "$WORK_ROOT/eval"           --split "$split"           --cases "$CASES"           --max-files "$MAX_TEST_FILES"
        slot=$((slot+1))
        if ((slot==${#GPU_IDS[@]})); then wait_wave; slot=0; fi
      done
      wait_wave
      ;;
    official)
      [[ -f "$WORK_ROOT/eval/protocol.json" ]] || {
        echo "Run PHASES=test first: missing $WORK_ROOT/eval/protocol.json" >&2
        exit 2
      }
      off=()
      [[ "$RESUME" == 1 ]] && off+=(--resume)
      for split in "${TESTS[@]}"; do
        "$PYTHON_BIN" "$ROOT_DIR/eval_rep_fullpath_official.py"           --dataset-root "$DATASET_ROOT"           --work-root "$WORK_ROOT/eval"           --split "$split"           --frame-stride "$FRAME_STRIDE"           --workers "$OFFICIAL_WORKERS"           "${off[@]}"
      done
      ;;
    *)
      echo "Unknown Rep-C2-v2 phase: $phase" >&2
      exit 2
      ;;
  esac
done
