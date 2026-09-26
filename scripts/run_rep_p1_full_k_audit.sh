#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

P0_WORK_ROOT=${P0_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources}
P0_CACHE_ROOT=${P0_CACHE_ROOT:-${P0_WORK_ROOT}/cache}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p1_action_evidence}
IMAGE_CACHE_ROOT=${IMAGE_CACHE_ROOT:-${WORK_ROOT}/image_cache}
TRAIN_ROOT=${TRAIN_ROOT:-${WORK_ROOT}/train}

CHECKPOINT_KIND=${CHECKPOINT_KIND:-latest}
case "$CHECKPOINT_KIND" in
  best|latest) ;;
  *)
    echo "CHECKPOINT_KIND must be best or latest, got: $CHECKPOINT_KIND" >&2
    exit 2
    ;;
esac

AUDIT_ROOT=${AUDIT_ROOT:-${WORK_ROOT}/full_k_audit_${CHECKPOINT_KIND}}
VARIANTS=${VARIANTS:-action_only,geo_pred,img_point,img_region}
SPLITS=${SPLITS:-test_similar,test_novel}
AUDIT_GPUS=${AUDIT_GPUS:-0,1}
MAX_FRAMES=${MAX_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
FOCUS_TOP_N=${FOCUS_TOP_N:-200}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

IFS=',' read -r -a SPLIT_IDS <<< "$SPLITS"
IFS=',' read -r -a GPU_IDS <<< "$AUDIT_GPUS"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "No AUDIT_GPUS" >&2; exit 2; }

for variant in ${VARIANTS//,/ }; do
  ckpt="$TRAIN_ROOT/$variant/checkpoint_${CHECKPOINT_KIND}.tar"
  [[ -f "$ckpt" ]] || {
    echo "Missing Rep-P1 checkpoint: $ckpt" >&2
    exit 2
  }
done

mkdir -p "$AUDIT_ROOT/logs"
PIDS=()
NAMES=()

wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-P1-FULL-K][ERROR] ${NAMES[$i]} failed; inspect logs" >&2
      failed=1
    fi
  done
  PIDS=(); NAMES=()
  [[ $failed -eq 0 ]] || exit 1
}

slot=0
for raw_split in "${SPLIT_IDS[@]}"; do
  split="$(echo "$raw_split" | xargs)"
  [[ -d "$P0_CACHE_ROOT/$split" ]] || {
    echo "Missing Rep-P0 split cache: $P0_CACHE_ROOT/$split" >&2
    exit 2
  }
  if [[ "$VARIANTS" == *"img_point"* || "$VARIANTS" == *"img_region"* ]]; then
    [[ -d "$IMAGE_CACHE_ROOT/$split" ]] || {
      echo "Missing Rep-P1 image cache: $IMAGE_CACHE_ROOT/$split" >&2
      exit 2
    }
  fi

  gpu="${GPU_IDS[$slot]}"
  out="$AUDIT_ROOT/$split"
  log="$AUDIT_ROOT/logs/${split}.log"
  mkdir -p "$out"

  echo "[REP-P1-FULL-K] split=$split checkpoint=$CHECKPOINT_KIND gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u     "$ROOT_DIR/audit_rep_p1_full_k_curves.py"     --p0_cache_root "$P0_CACHE_ROOT/$split"     --image_cache_root "$IMAGE_CACHE_ROOT/$split"     --train_root "$TRAIN_ROOT"     --output_dir "$out"     --split "$split"     --variants "$VARIANTS"     --checkpoint_kind "$CHECKPOINT_KIND"     --device cuda:0     --max_frames "$MAX_FRAMES"     --progress_every "$PROGRESS_EVERY"     --focus_top_n "$FOCUS_TOP_N"     >"$log" 2>&1 &

  PIDS+=("$!")
  NAMES+=("$split")
  slot=$((slot+1))
  if [[ $slot -ge ${#GPU_IDS[@]} ]]; then
    wait_wave
    slot=0
  fi
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

"$PYTHON_BIN" "$ROOT_DIR/summarize_rep_p1_full_k_audit.py"   --audit_root "$AUDIT_ROOT"   --output_dir "$AUDIT_ROOT"   --splits "$SPLITS"

echo "[REP-P1-FULL-K] completed: $AUDIT_ROOT"
