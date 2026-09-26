#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
P0_WORK_ROOT=${P0_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources_5mm}
P0_CACHE_ROOT=${P0_CACHE_ROOT:-${P0_WORK_ROOT}/cache}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p1_action_evidence}
IMAGE_CACHE_ROOT=${IMAGE_CACHE_ROOT:-${WORK_ROOT}/image_cache}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}

SPLITS=${SPLITS:-train,test_seen,test_similar,test_novel}
CACHE_GPUS=${CACHE_GPUS:-0,1,2,3,4,5}
MAX_SAMPLES_PER_SHARD=${MAX_SAMPLES_PER_SHARD:-0}
RESUME=${RESUME:-1}
REPAIR_INVALID_CACHE=${REPAIR_INVALID_CACHE:-1}
OVERWRITE=${OVERWRITE:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-50}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1

[[ -f "$STAGE1_CKPT" ]] || { echo "Missing Stage-1 checkpoint: $STAGE1_CKPT" >&2; exit 2; }
[[ -d "$P0_CACHE_ROOT/train" ]] || { echo "Missing Rep-P0 cache: $P0_CACHE_ROOT" >&2; exit 2; }

IFS=',' read -r -a GPU_IDS <<< "$CACHE_GPUS"
IFS=',' read -r -a SPLIT_IDS <<< "$SPLITS"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "No CACHE_GPUS" >&2; exit 2; }

PIDS=()
NAMES=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
}
trap cleanup INT TERM

wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-P1-CACHE][ERROR] ${NAMES[$i]} failed" >&2
      failed=1
    fi
  done
  PIDS=(); NAMES=()
  [[ $failed -eq 0 ]] || exit 1
}

for raw_split in "${SPLIT_IDS[@]}"; do
  split="$(echo "$raw_split" | xargs)"
  [[ -d "$P0_CACHE_ROOT/$split" ]] || {
    echo "Missing Rep-P0 split cache: $P0_CACHE_ROOT/$split" >&2
    exit 2
  }
  for shard in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$shard]}"
    log="$WORK_ROOT/logs/cache_${split}_${shard}.log"
    mkdir -p "$(dirname "$log")"
    args=(
      "$ROOT_DIR/mine_rep_p1_image_features.py"
      --dataset_root "$DATASET_ROOT"
      --stage1_checkpoint "$STAGE1_CKPT"
      --p0_cache_root "$P0_CACHE_ROOT"
      --output_root "$IMAGE_CACHE_ROOT"
      --split "$split"
      --shard_id "$shard"
      --num_shards "${#GPU_IDS[@]}"
      --max_samples "$MAX_SAMPLES_PER_SHARD"
      --progress_every "$PROGRESS_EVERY"
    )
    [[ "$RESUME" == 1 ]] && args+=(--resume)
    [[ "$REPAIR_INVALID_CACHE" == 1 ]] && args+=(--repair_invalid_cache)
    [[ "$OVERWRITE" == 1 ]] && args+=(--overwrite)
    echo "[REP-P1-CACHE] split=$split shard=$shard gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "${args[@]}" >"$log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$split/$shard")
  done
  wait_wave
done

echo "[REP-P1-CACHE] completed: $IMAGE_CACHE_ROOT"
