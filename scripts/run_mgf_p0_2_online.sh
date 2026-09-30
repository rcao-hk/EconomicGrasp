#!/usr/bin/env bash
# P0-2: live exact evaluator; no feature/action label mining or training cache.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_mixed_matched/train/checkpoint_latest.pt}
CONTROLS_ROOT=${CONTROLS_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/mgf_p0_1_frozen_online}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/mgf_p0_2_online_audit}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
GPUS=${GPUS:-0,1,2}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
VARIANTS=${VARIANTS:-base,feature_only,full,cva}
MODE=${MODE:-both}
RESUME=${RESUME:-0}
GNTRANS_RGB_ROOT=${GNTRANS_RGB_ROOT:-}
[[ "$MODE" == action || -d "$GNTRANS_RGB_ROOT" ]] || { echo 'Set GNTRANS_RGB_ROOT to paired RGB root (containing scenes/00000/0000_color.png).' >&2; exit 2; }
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} PYTHONUNBUFFERED=1
command -v setsid >/dev/null
IFS=, read -ra IDS <<< "$GPUS"; IFS=, read -ra SS <<< "$SPLITS"
declare -A used=()
for id in "${IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ && -z "${used[$id]:-}" ]] || { echo "Invalid/repeated GPU $id" >&2; exit 2; }; used[$id]=1
done
resume=(); [[ "$RESUME" == 1 ]] && resume=(--resume)
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
mkdir -p "$WORK_ROOT/logs"
for split in "${SS[@]}"; do
  for shard in "${!IDS[@]}"; do
    setsid env CUDA_VISIBLE_DEVICES="${IDS[$shard]}" "$PYTHON_BIN" -u "$ROOT/mgf_p0_audit.py" \
      --source-checkpoint "$SOURCE_CHECKPOINT" --controls-root "$CONTROLS_ROOT" --variants "$VARIANTS" \
      --dataset-root "$DATASET_ROOT" --gntrans-rgb-root "$GNTRANS_RGB_ROOT" --output-root "$WORK_ROOT" \
      --split "$split" --mode "$MODE" --frames-per-scene "${FRAMES_PER_SCENE:-1}" --queries "${QUERIES:-8}" \
      --seed "${SEED:-0}" --ray-offsets="${RAY_OFFSETS:--0.02,-0.01,0.01,0.02}" \
      --roll-degrees="${ROLL_DEGREES:--15,15}" --width-offsets="${WIDTH_OFFSETS:--0.01,0.01}" \
      --fc-mode "${FC_MODE:-official}" --verify-n "${VERIFY_N:-8}" \
      --shard-id "$shard" --num-shards "${#IDS[@]}" "${resume[@]}" \
      >>"$WORK_ROOT/logs/${split}_${shard}.log" 2>&1 &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || { cleanup; exit 1; }; done
  PIDS=()
  "$PYTHON_BIN" "$ROOT/mgf_p0_audit.py" --merge --output-root "$WORK_ROOT" --split "$split"
done
