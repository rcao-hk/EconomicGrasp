#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
E1_BASE_ROOT=${E1_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dcr_cva_10pct}
CACHE_ROOT=${CACHE_ROOT:-$E1_BASE_ROOT/action_cache}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-$E1_BASE_ROOT/train/E1/checkpoint_best.pt}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
GPUS=${GPUS:-0}
LOSS_MODES=${LOSS_MODES:-cdf}
PHASES=${PHASES:-train,infer,eval,summary}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
TEST_CASES=${TEST_CASES:-nominal,bias:-20,bias:20}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
EPOCHS=${EPOCHS:-6}
LR=${LR:-0.00001}
RANK_LR=${RANK_LR:-0.0001}
RANK_BOUND=${RANK_BOUND:-0.5}
RANK_HIDDEN=${RANK_HIDDEN:-128}
RANK_ANCHOR_WEIGHT=${RANK_ANCHOR_WEIGHT:-0.1}
RANK_STRENGTH=${RANK_STRENGTH:-1.0}
RELATIVE_WEIGHT=${RELATIVE_WEIGHT:-1.0}
RELATIVE_BETA=${RELATIVE_BETA:-0.1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
QUERY_CHUNK=${QUERY_CHUNK:-64}
GROUP_CHUNK=${GROUP_CHUNK:-512}
VAL_EVERY=${VAL_EVERY:-1}
SEED=${SEED:-2032}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
INFER_MAX_FRAMES=${INFER_MAX_FRAMES:-0}
INFER_QUERIES=${INFER_QUERIES:-0}
# Optional E1 checkpoint for a no-training, zero-residual equivalence test.
INFER_CHECKPOINT=${INFER_CHECKPOINT:-}
EVAL_METHODS=${EVAL_METHODS:-local,stage1,anchored}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
RESUME=${RESUME:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1
command -v setsid >/dev/null || { echo 'setsid is required' >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
IFS=',' read -r -a MODES <<< "$LOSS_MODES"
IFS=',' read -r -a STEPS <<< "$PHASES"
IFS=',' read -r -a TESTS <<< "$SPLITS"
[[ ${#GPU_IDS[@]} -gt 0 && "$OFFICIAL_WORKERS" =~ ^[1-9][0-9]*$ ]] || exit 2
for x in "${MODES[@]}"; do
  [[ "$x" == cdf || "$x" == cdf_relative ]] || { echo "Bad loss mode: $x" >&2; exit 2; }
done
for x in "${TESTS[@]}"; do
  [[ "$x" == test_seen || "$x" == test_similar || "$x" == test_novel ]] || exit 2
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
trap 'exit 130' INT
trap 'exit 143' TERM
launch() {
  local name="$1" gpu="$2" logfile="$3"; shift 3
  mkdir -p "$(dirname "$logfile")"
  echo "[DCR] $name GPU=$gpu log=$logfile"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$logfile" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}
wait_wave() {
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[DCR ERROR] ${NAMES[$i]} failed; stopping other jobs" >&2; exit 1
    fi
  done
  PIDS=(); NAMES=()
}
resume=(); [[ "$RESUME" == 1 ]] && resume+=(--resume)
for phase in "${STEPS[@]}"; do
  case "$phase" in
    train)
      [[ -f "$CACHE_ROOT/protocol.json" && -f "$INIT_CHECKPOINT" ]] || {
        echo 'Existing E1 action cache and warm-start checkpoint required' >&2; exit 2; }
      slot=0
      for mode in "${MODES[@]}"; do
        launch "train/$mode" "${GPU_IDS[$slot]}" "$WORK_ROOT/logs/train_${mode}.log" \
          "$ROOT_DIR/train_dcr_cva.py" --dataset-root "$DATASET_ROOT" \
          --stage1-checkpoint "$STAGE1_CKPT" --cache-root "$CACHE_ROOT" \
          --init-checkpoint "$INIT_CHECKPOINT" --output-root "$WORK_ROOT/train/$mode" \
          --selection-loss "$mode" --epochs "$EPOCHS" --lr "$LR" --rank-lr "$RANK_LR" \
          --rank-bound "$RANK_BOUND" --rank-hidden "$RANK_HIDDEN" --rank-anchor-weight "$RANK_ANCHOR_WEIGHT" \
          --relative-weight "$RELATIVE_WEIGHT" --relative-beta "$RELATIVE_BETA" --seed "$SEED" \
          --grad-accum "$GRAD_ACCUM_STEPS" --query-chunk "$QUERY_CHUNK" --group-chunk "$GROUP_CHUNK" \
          --val-every "$VAL_EVERY" --max-train-frames "$MAX_TRAIN_FRAMES" --max-val-frames "$MAX_VAL_FRAMES" \
          "${resume[@]}"
        slot=$((slot+1))
        if ((slot==${#GPU_IDS[@]})); then wait_wave; slot=0; fi
      done
      wait_wave ;;
    infer)
      for mode in "${MODES[@]}"; do
        checkpoint=${INFER_CHECKPOINT:-$WORK_ROOT/train/$mode/checkpoint_best.pt}
        [[ -f "$checkpoint" ]] || { echo "Missing $checkpoint" >&2; exit 2; }
        for split in "${TESTS[@]}"; do
          for s in "${!GPU_IDS[@]}"; do
            launch "infer/$mode/$split/$s" "${GPU_IDS[$s]}" "$WORK_ROOT/logs/infer_${mode}_${split}_${s}.log" \
              "$ROOT_DIR/inference_dcr_cva.py" --dataset-root "$DATASET_ROOT" --stage1-checkpoint "$STAGE1_CKPT" \
              --checkpoint "$checkpoint" --output-root "$WORK_ROOT/test/$mode" --split "$split" \
              --cases "$TEST_CASES" --sample-interval "$SAMPLE_INTERVAL" --rank-strength "$RANK_STRENGTH" \
              --query-limit "$INFER_QUERIES" --max-frames "$INFER_MAX_FRAMES" \
              --shard-id "$s" --num-shards "${#GPU_IDS[@]}" "${resume[@]}"
          done
          wait_wave
        done
      done ;;
    eval|official)
      for mode in "${MODES[@]}"; do
        slot=0
        for split in "${TESTS[@]}"; do
          launch "official/$mode/$split" "${GPU_IDS[$slot]}" "$WORK_ROOT/logs/official_${mode}_${split}.log" \
            "$ROOT_DIR/eval_dcr_cva.py" --dataset-root "$DATASET_ROOT" --inference-root "$WORK_ROOT/test/$mode" \
            --split "$split" --methods "$EVAL_METHODS" --workers "$OFFICIAL_WORKERS" "${resume[@]}"
          slot=$((slot+1))
          if ((slot==${#GPU_IDS[@]})); then wait_wave; slot=0; fi
        done
        wait_wave
      done ;;
    summary)
      "$PYTHON_BIN" "$ROOT_DIR/summarize_dcr_cva.py" --work-root "$WORK_ROOT" ;;
    *) echo "Unknown phase: $phase" >&2; exit 2 ;;
  esac
done
