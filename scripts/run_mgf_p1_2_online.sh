#!/usr/bin/env bash
# P1-2: live canonical-label vs exact-action audit; no label/action cache.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_mixed_matched/train/checkpoint_latest.pt}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/mgf_p1_2_label_audit}
GPUS=${GPUS:-3,5,6}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
FRAMES_PER_SCENE=${FRAMES_PER_SCENE:-1}
QUERIES=${QUERIES:-8}
TOP_CANDIDATES=${TOP_CANDIDATES:-4}
RANDOM_CANDIDATES=${RANDOM_CANDIDATES:-4}
SEED=${SEED:-0}
FC_MODE=${FC_MODE:-official}
VERIFY_N=${VERIFY_N:-8}
RESUME=${RESUME:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} PYTHONUNBUFFERED=1
command -v setsid >/dev/null
IFS=, read -ra IDS <<< "$GPUS"; IFS=, read -ra SS <<< "$SPLITS"
declare -A used=()
for id in "${IDS[@]}"; do [[ "$id" =~ ^[0-9]+$ && -z "${used[$id]:-}" ]] || exit 2; used[$id]=1; done
resume=(); [[ "$RESUME" == 1 ]] && resume=(--resume)
PIDS=()
cleanup(){ for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
mkdir -p "$WORK_ROOT/logs"
for split in "${SS[@]}"; do
  for shard in "${!IDS[@]}"; do
    setsid env CUDA_VISIBLE_DEVICES="${IDS[$shard]}" "$PYTHON_BIN" -u "$ROOT/mgf_p1_label_audit.py"       --source-checkpoint "$SOURCE_CHECKPOINT" --dataset-root "$DATASET_ROOT"       --output-root "$WORK_ROOT" --split "$split" --frames-per-scene "$FRAMES_PER_SCENE"       --queries "$QUERIES" --top-candidates "$TOP_CANDIDATES" --random-candidates "$RANDOM_CANDIDATES"       --seed "$SEED" --fc-mode "$FC_MODE" --verify-n "$VERIFY_N"       --shard-id "$shard" --num-shards "${#IDS[@]}" "${resume[@]}"       >>"$WORK_ROOT/logs/${split}_${shard}.log" 2>&1 &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || { cleanup; exit 1; }; done
  PIDS=()
  "$PYTHON_BIN" "$ROOT/mgf_p1_label_audit.py" --merge --output-root "$WORK_ROOT" --split "$split"
done
