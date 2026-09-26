#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

E1_BASE_ROOT=${E1_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct}
DCR_BASE_ROOT=${DCR_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dcr_cva_10pct}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dcr_air_10pct}
CACHE_ROOT=${CACHE_ROOT:-$E1_BASE_ROOT/action_cache}
DCR_CHECKPOINT=${DCR_CHECKPOINT:-$DCR_BASE_ROOT/train/cdf/checkpoint_best.pt}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}

GPUS=${GPUS:-0,1,2,3}
TRAIN_GPU=${TRAIN_GPU:-}
PHASES=${PHASES:-train,infer,eval,summary}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
TEST_CASES=${TEST_CASES:-nominal,bias:-15,bias:15,bias:-25,bias:25,scale:-0.03,scale:0.03,smooth:5,smooth:10}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
EPOCHS=${EPOCHS:-12}
LR=${LR:-0.0001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-0.01}
AIR_HIDDEN=${AIR_HIDDEN:-128}
AIR_BOUND=${AIR_BOUND:-1.0}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
QUERY_CHUNK=${QUERY_CHUNK:-64}
SEED=${SEED:-2041}
VAL_EVERY=${VAL_EVERY:-1}
NO_ERROR_TRAINING=${NO_ERROR_TRAINING:-0}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
INFER_MAX_FRAMES=${INFER_MAX_FRAMES:-0}
INFER_QUERIES=${INFER_QUERIES:-0}
EVAL_METHODS=${EVAL_METHODS:-native,dcr_stage1,air_stage1}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-16}
RESUME=${RESUME:-1}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1

command -v setsid >/dev/null || { echo "setsid is required" >&2; exit 2; }
[[ -f "$CACHE_ROOT/protocol.json" ]] || { echo "Missing cache protocol" >&2; exit 2; }
[[ -f "$DCR_CHECKPOINT" ]] || { echo "Missing DCR checkpoint: $DCR_CHECKPOINT" >&2; exit 2; }
[[ -f "$STAGE1_CKPT" ]] || { echo "Missing Stage-1 checkpoint: $STAGE1_CKPT" >&2; exit 2; }

IFS="," read -r -a GPU_IDS <<< "$GPUS"
IFS="," read -r -a STEPS <<< "$PHASES"
IFS="," read -r -a TESTS <<< "$SPLITS"
[[ ${#GPU_IDS[@]} -gt 0 && "$OFFICIAL_WORKERS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -n "$TRAIN_GPU" ]] || TRAIN_GPU=${GPU_IDS[0]}
for split in "${TESTS[@]}"; do
  [[ "$split" == test_seen || "$split" == test_similar || "$split" == test_novel ]] || {
    echo "Bad split: $split" >&2; exit 2; }
done

PIDS=(); NAMES=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done
  if (( ${#PIDS[@]} )); then
    sleep 2
    for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
  fi
}
trap cleanup EXIT
trap "exit 130" INT
trap "exit 143" TERM

launch() {
  local name="$1" gpu="$2" logfile="$3"; shift 3
  mkdir -p "$(dirname "$logfile")"
  echo "[DCR-AIR] $name GPU=$gpu log=$logfile"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$logfile" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}
wait_wave() {
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[DCR-AIR ERROR] ${NAMES[$i]} failed; stopping other jobs" >&2
      exit 1
    fi
  done
  PIDS=(); NAMES=()
}

resume=(); [[ "$RESUME" == 1 ]] && resume+=(--resume)
noerr=(); [[ "$NO_ERROR_TRAINING" == 1 ]] && noerr+=(--no-error-training)

for phase in "${STEPS[@]}"; do
  case "$phase" in
    train)
      launch "train" "$TRAIN_GPU" "$WORK_ROOT/logs/train.log" \
        "$ROOT_DIR/train_dcr_air.py" \
        --dataset-root "$DATASET_ROOT" \
        --stage1-checkpoint "$STAGE1_CKPT" \
        --dcr-checkpoint "$DCR_CHECKPOINT" \
        --cache-root "$CACHE_ROOT" \
        --output-root "$WORK_ROOT/train" \
        --epochs "$EPOCHS" --lr "$LR" --weight-decay "$WEIGHT_DECAY" \
        --anchor-weight "$ANCHOR_WEIGHT" --hidden "$AIR_HIDDEN" \
        --residual-bound "$AIR_BOUND" --grad-accum "$GRAD_ACCUM_STEPS" \
        --query-chunk "$QUERY_CHUNK" --seed "$SEED" --val-every "$VAL_EVERY" \
        --max-train-frames "$MAX_TRAIN_FRAMES" --max-val-frames "$MAX_VAL_FRAMES" \
        "${noerr[@]}" "${resume[@]}"
      wait_wave
      ;;
    infer)
      checkpoint="$WORK_ROOT/train/checkpoint_best.pt"
      [[ -f "$checkpoint" ]] || { echo "Missing AIR checkpoint: $checkpoint" >&2; exit 2; }
      for split in "${TESTS[@]}"; do
        for s in "${!GPU_IDS[@]}"; do
          launch "infer/$split/$s" "${GPU_IDS[$s]}" \
            "$WORK_ROOT/logs/infer_${split}_${s}.log" \
            "$ROOT_DIR/inference_dcr_air.py" \
            --dataset-root "$DATASET_ROOT" --stage1-checkpoint "$STAGE1_CKPT" \
            --dcr-checkpoint "$DCR_CHECKPOINT" --checkpoint "$checkpoint" \
            --output-root "$WORK_ROOT/test" --split "$split" \
            --cases "$TEST_CASES" --sample-interval "$SAMPLE_INTERVAL" \
            --query-limit "$INFER_QUERIES" --max-frames "$INFER_MAX_FRAMES" \
            --shard-id "$s" --num-shards "${#GPU_IDS[@]}" "${resume[@]}"
        done
        wait_wave
      done
      ;;
    eval|official)
      slot=0
      for split in "${TESTS[@]}"; do
        launch "official/$split" "${GPU_IDS[$slot]}" \
          "$WORK_ROOT/logs/official_${split}.log" \
          "$ROOT_DIR/eval_dcr_air.py" \
          --dataset-root "$DATASET_ROOT" --inference-root "$WORK_ROOT/test" \
          --split "$split" --methods "$EVAL_METHODS" \
          --workers "$OFFICIAL_WORKERS" "${resume[@]}"
        slot=$((slot+1))
        if (( slot == ${#GPU_IDS[@]} )); then wait_wave; slot=0; fi
      done
      wait_wave
      ;;
    summary)
      "$PYTHON_BIN" "$ROOT_DIR/summarize_dcr_air.py" --work-root "$WORK_ROOT"
      ;;
    *) echo "Unknown phase: $phase" >&2; exit 2 ;;
  esac
done
