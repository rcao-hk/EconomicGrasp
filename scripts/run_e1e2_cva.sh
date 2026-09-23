#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct}
CACHE_ROOT=${CACHE_ROOT:-$WORK_ROOT/action_cache}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
GPUS=${GPUS:-0}
VARIANTS=${VARIANTS:-E1,E2}
PHASES=${PHASES:-prepare,train,infer,eval,summary}
PREP_SPLITS=${PREP_SPLITS:-train,test_seen}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
CACHE_QUERIES=${CACHE_QUERIES:-64}
INFER_QUERIES=${INFER_QUERIES:-0}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
VAL_CASES=${VAL_CASES:-nominal,bias:-20,bias:20}
TEST_CASES=${TEST_CASES:-nominal,bias:-20,bias:20}
VOXEL_SIZE=${VOXEL_SIZE:-0.005}
EVAL_CHUNK=${EVAL_CHUNK:-128}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-8}
MIN_HOST_FREE_GIB=${MIN_HOST_FREE_GIB:-6}
EPOCHS=${EPOCHS:-12}
LR=${LR:-0.0001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
RELATIVE_WEIGHT=${RELATIVE_WEIGHT:-1.0}
RELATIVE_BETA=${RELATIVE_BETA:-0.1}
QUERY_CHUNK=${QUERY_CHUNK:-64}
GROUP_CHUNK=${GROUP_CHUNK:-512}
VAL_EVERY=${VAL_EVERY:-1}
ERROR_TRAINING=${ERROR_TRAINING:-1}
PREP_MAX_FRAMES=${PREP_MAX_FRAMES:-0}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
INFER_MAX_FRAMES=${INFER_MAX_FRAMES:-0}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
SCORE_SOURCE=${SCORE_SOURCE:-model}
RESUME=${RESUME:-1}
REPAIR_CORRUPT=${REPAIR_CORRUPT:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1
command -v setsid >/dev/null || { echo 'setsid is required' >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
IFS=',' read -r -a MODES <<< "$VARIANTS"
IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a PREP <<< "$PREP_SPLITS"
IFS=',' read -r -a TESTS <<< "$SPLITS"
for v in "${MODES[@]}"; do
  [[ "$v" == E1 || "$v" == E2 ]] || { echo "Bad variant $v" >&2; exit 2; }
done
[[ "$ERROR_TRAINING" == 0 || "$ERROR_TRAINING" == 1 ]] || exit 2
PIDS=(); NAMES=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done
  if (( ${#PIDS[@]} )); then
    sleep 2
    for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
launch() {
  local name="$1" gpu="$2" logfile="$3"; shift 3
  mkdir -p "$(dirname "$logfile")"
  echo "[E1E2] $name GPU=$gpu log=$logfile"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$logfile" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}
wait_wave() {
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[E1E2 ERROR] ${NAMES[$i]} failed; stopping remaining jobs" >&2
      exit 1
    fi
  done
  PIDS=(); NAMES=()
}
resume=(); [[ "$RESUME" == 1 ]] && resume+=(--resume)
for phase in "${STEPS[@]}"; do
  case "$phase" in
    prepare)
      repair=(); [[ "$REPAIR_CORRUPT" == 1 ]] && repair+=(--repair-corrupt)
      for split in "${PREP[@]}"; do
        for s in "${!GPU_IDS[@]}"; do
          launch "prepare/$split/$s" "${GPU_IDS[$s]}" "$WORK_ROOT/logs/prepare_${split}_${s}.log" \
            "$ROOT_DIR/prepare_e1e2_cva.py" \
            --dataset-root "$DATASET_ROOT" --stage1-checkpoint "$STAGE1_CKPT" \
            --output-root "$CACHE_ROOT" --split "$split" --sample-interval "$SAMPLE_INTERVAL" \
            --query-limit "$CACHE_QUERIES" "--offsets-mm=$OFFSETS_MM" --val-cases "$VAL_CASES" \
            --voxel-size "$VOXEL_SIZE" --eval-chunk "$EVAL_CHUNK" --fc-mode "$FC_MODE" \
            --verify-n "$VERIFY_N" --min-host-free-gib "$MIN_HOST_FREE_GIB" \
            --max-frames "$PREP_MAX_FRAMES" --shard-id "$s" --num-shards "${#GPU_IDS[@]}" \
            "${resume[@]}" "${repair[@]}"
        done
        wait_wave
      done
      ;;
    train)
      slot=0; no_error=(); [[ "$ERROR_TRAINING" == 0 ]] && no_error+=(--no-error-training)
      for v in "${MODES[@]}"; do
        launch "train/$v" "${GPU_IDS[$slot]}" "$WORK_ROOT/logs/train_${v}.log" \
          "$ROOT_DIR/train_e1e2_cva.py" \
          --dataset-root "$DATASET_ROOT" --stage1-checkpoint "$STAGE1_CKPT" \
          --cache-root "$CACHE_ROOT" --output-root "$WORK_ROOT/train/$v" --variant "$v" \
          --epochs "$EPOCHS" --lr "$LR" --weight-decay "$WEIGHT_DECAY" \
          --grad-accum "$GRAD_ACCUM_STEPS" --relative-weight "$RELATIVE_WEIGHT" --relative-beta "$RELATIVE_BETA" \
          --query-chunk "$QUERY_CHUNK" --group-chunk "$GROUP_CHUNK" --val-every "$VAL_EVERY" \
          --max-train-frames "$MAX_TRAIN_FRAMES" --max-val-frames "$MAX_VAL_FRAMES" \
          "${resume[@]}" "${no_error[@]}"
        slot=$((slot+1))
        if ((slot==${#GPU_IDS[@]})); then wait_wave; slot=0; fi
      done
      wait_wave
      ;;
    infer)
      for v in "${MODES[@]}"; do
        for split in "${TESTS[@]}"; do
          for s in "${!GPU_IDS[@]}"; do
            launch "infer/$v/$split/$s" "${GPU_IDS[$s]}" "$WORK_ROOT/logs/infer_${v}_${split}_${s}.log" \
              "$ROOT_DIR/inference_e1e2_cva.py" \
              --dataset-root "$DATASET_ROOT" --stage1-checkpoint "$STAGE1_CKPT" \
              --checkpoint "$WORK_ROOT/train/$v/checkpoint_best.pt" --output-root "$WORK_ROOT/test/$v" \
              --split "$split" --cases "$TEST_CASES" --sample-interval "$SAMPLE_INTERVAL" \
              --query-limit "$INFER_QUERIES" --score-source "$SCORE_SOURCE" --max-frames "$INFER_MAX_FRAMES" \
              --shard-id "$s" --num-shards "${#GPU_IDS[@]}" "${resume[@]}"
          done
          wait_wave
        done
      done
      ;;
    eval|official)
      methods=native,model
      for v in "${MODES[@]}"; do
        for split in "${TESTS[@]}"; do
          launch "official/$v/$split" "" "$WORK_ROOT/logs/official_${v}_${split}.log" \
            "$ROOT_DIR/eval_e1e2_cva.py" --dataset-root "$DATASET_ROOT" \
            --inference-root "$WORK_ROOT/test/$v" --split "$split" \
            --methods "$methods" --workers "$OFFICIAL_WORKERS" "${resume[@]}"
          wait_wave
        done
        # Native generator is shared. Avoid repeating its expensive official AP.
        methods=model
      done
      ;;
    summary)
      "$PYTHON_BIN" "$ROOT_DIR/summarize_e1e2_cva.py" --work-root "$WORK_ROOT"
      ;;
    *) echo "Unknown phase $phase" >&2; exit 2 ;;
  esac
done
