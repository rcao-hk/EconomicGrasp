#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

P0_WORK_ROOT=${P0_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources_5mm}
P0_CACHE_ROOT=${P0_CACHE_ROOT:-${P0_WORK_ROOT}/cache}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p1_action_evidence}
IMAGE_CACHE_ROOT=${IMAGE_CACHE_ROOT:-${WORK_ROOT}/image_cache}
TRAIN_ROOT=${TRAIN_ROOT:-${WORK_ROOT}/train}

VARIANTS=${VARIANTS:-geo_pred,img_point,img_region}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-256}
DROPOUT=${DROPOUT:-0.1}
CDF_WEIGHT=${CDF_WEIGHT:-1.0}
RANK_WEIGHT=${RANK_WEIGHT:-0.5}
RANK_TEMPERATURE=${RANK_TEMPERATURE:-0.1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-5}
SEED=${SEED:-2101}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

[[ -d "$P0_CACHE_ROOT/train" ]] || { echo "Missing P0 train cache" >&2; exit 2; }
[[ -d "$P0_CACHE_ROOT/test_seen" ]] || { echo "Missing P0 seen cache" >&2; exit 2; }
[[ -d "$IMAGE_CACHE_ROOT/train" ]] || { echo "Missing Rep-P1 image cache; run cache phase first" >&2; exit 2; }
[[ -d "$IMAGE_CACHE_ROOT/test_seen" ]] || { echo "Missing Rep-P1 seen image cache" >&2; exit 2; }

IFS=',' read -r -a VARIANT_IDS <<< "$VARIANTS"
IFS=',' read -r -a GPU_IDS <<< "$TRAIN_GPUS"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "No TRAIN_GPUS" >&2; exit 2; }
mkdir -p "$TRAIN_ROOT"

PIDS=()
NAMES=()
wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-P1-TRAIN][ERROR] ${NAMES[$i]} failed" >&2
      failed=1
    fi
  done
  PIDS=(); NAMES=()
  [[ $failed -eq 0 ]] || exit 1
}

slot=0
for raw_variant in "${VARIANT_IDS[@]}"; do
  variant="$(echo "$raw_variant" | xargs)"
  [[ -n "$variant" ]] || continue
  gpu="${GPU_IDS[$slot]}"
  out="$TRAIN_ROOT/$variant"
  mkdir -p "$out"
  args=(
    "$ROOT_DIR/train_rep_p1_probe.py"
    --train_p0_root "$P0_CACHE_ROOT/train"
    --val_p0_root "$P0_CACHE_ROOT/test_seen"
    --train_image_root "$IMAGE_CACHE_ROOT/train"
    --val_image_root "$IMAGE_CACHE_ROOT/test_seen"
    --output_dir "$out"
    --variant "$variant"
    --epochs "$EPOCHS"
    --learning_rate "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --hidden_dim "$HIDDEN_DIM"
    --dropout "$DROPOUT"
    --cdf_weight "$CDF_WEIGHT"
    --rank_weight "$RANK_WEIGHT"
    --rank_temperature "$RANK_TEMPERATURE"
    --grad_accum_steps "$GRAD_ACCUM_STEPS"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
    --seed "$SEED"
    --device cuda:0
    --max_train_frames "$MAX_TRAIN_FRAMES"
    --max_val_frames "$MAX_VAL_FRAMES"
    --progress_every "$PROGRESS_EVERY"
  )
  echo "[REP-P1-TRAIN] variant=$variant gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "${args[@]}" >"$out/train.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$variant")
  slot=$((slot+1))
  if [[ $slot -ge ${#GPU_IDS[@]} ]]; then
    wait_wave
    slot=0
  fi
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

echo "[REP-P1-TRAIN] completed: $TRAIN_ROOT"
