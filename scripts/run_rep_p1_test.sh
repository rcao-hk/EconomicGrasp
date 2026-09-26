#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

P0_WORK_ROOT=${P0_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources_5mm}
P0_CACHE_ROOT=${P0_CACHE_ROOT:-${P0_WORK_ROOT}/cache}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p1_action_evidence}
IMAGE_CACHE_ROOT=${IMAGE_CACHE_ROOT:-${WORK_ROOT}/image_cache}
TRAIN_ROOT=${TRAIN_ROOT:-${WORK_ROOT}/train}

# Which training checkpoint to evaluate.  Keep "best" as the backwards-
# compatible default; use CHECKPOINT_KIND=latest for representation-dynamics
# diagnosis without overwriting the formal best-checkpoint test results.
CHECKPOINT_KIND=${CHECKPOINT_KIND:-best}
case "$CHECKPOINT_KIND" in
  best|latest) ;;
  *)
    echo "CHECKPOINT_KIND must be best or latest, got: $CHECKPOINT_KIND" >&2
    exit 2
    ;;
esac

if [[ -z "${TEST_ROOT:-}" ]]; then
  if [[ "$CHECKPOINT_KIND" == "latest" ]]; then
    TEST_ROOT="${WORK_ROOT}/test_latest"
  else
    TEST_ROOT="${WORK_ROOT}/test"
  fi
fi

VARIANTS=${VARIANTS:-action_only,geo_pred,img_point,img_region}
SPLITS=${SPLITS:-test_similar,test_novel}
TEST_GPUS=${TEST_GPUS:-0,1,2,3,4,5}
MAX_FRAMES=${MAX_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
SAVE_PER_QUERY=${SAVE_PER_QUERY:-0}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

IFS=',' read -r -a VARIANT_IDS <<< "$VARIANTS"
IFS=',' read -r -a SPLIT_IDS <<< "$SPLITS"
IFS=',' read -r -a GPU_IDS <<< "$TEST_GPUS"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "No TEST_GPUS" >&2; exit 2; }
mkdir -p "$TEST_ROOT"

PIDS=()
NAMES=()
wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-P1-TEST][ERROR] ${NAMES[$i]} failed" >&2
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
  ckpt="$TRAIN_ROOT/$variant/checkpoint_${CHECKPOINT_KIND}.tar"
  [[ -f "$ckpt" ]] || { echo "Missing checkpoint: $ckpt" >&2; exit 2; }
  for raw_split in "${SPLIT_IDS[@]}"; do
    split="$(echo "$raw_split" | xargs)"
    [[ -d "$P0_CACHE_ROOT/$split" ]] || {
      echo "Missing P0 split cache: $P0_CACHE_ROOT/$split" >&2; exit 2;
    }
    if [[ "$variant" == "img_point" || "$variant" == "img_region" ]]; then
      [[ -d "$IMAGE_CACHE_ROOT/$split" ]] || {
        echo "Missing Rep-P1 image cache: $IMAGE_CACHE_ROOT/$split" >&2; exit 2;
      }
    fi
    gpu="${GPU_IDS[$slot]}"
    out="$TEST_ROOT/$variant/$split"
    mkdir -p "$out"
    args=(
      "$ROOT_DIR/test_rep_p1_probe.py"
      --p0_cache_root "$P0_CACHE_ROOT/$split"
      --image_cache_root "$IMAGE_CACHE_ROOT/$split"
      --checkpoint "$ckpt"
      --output_dir "$out"
      --device cuda:0
      --max_frames "$MAX_FRAMES"
      --progress_every "$PROGRESS_EVERY"
    )
    [[ "$SAVE_PER_QUERY" == 1 ]] && args+=(--save_per_query)
    echo "[REP-P1-TEST] checkpoint=$CHECKPOINT_KIND variant=$variant split=$split gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "${args[@]}" >"$out/test.log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$variant/$split")
    slot=$((slot+1))
    if [[ $slot -ge ${#GPU_IDS[@]} ]]; then
      wait_wave
      slot=0
    fi
  done
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

"$PYTHON_BIN" "$ROOT_DIR/summarize_rep_p1.py"   --test_root "$TEST_ROOT"   --output_dir "$TEST_ROOT"   --variants "$VARIANTS"   --splits "$SPLITS"

echo "[REP-P1-TEST] checkpoint=$CHECKPOINT_KIND completed: $TEST_ROOT"
