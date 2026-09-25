#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
DCR_BASE_ROOT=${DCR_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dcr_cva_10pct}
DCR_CHECKPOINT=${DCR_CHECKPOINT:-$DCR_BASE_ROOT/train/cdf/checkpoint_best.pt}
VIS_ROOT=${VIS_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dcr_behavior_vis}
AIR_CHECKPOINT=${AIR_CHECKPOINT:-}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-}

GPUS=${GPUS:-0,1,2}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
FRAMES=${FRAMES:-0,128,255}
SCENES=${SCENES:-}
CASES=${CASES:-nominal,bias:-25,bias:25,scale:-0.03,scale:0.03,smooth:10}
VIS_ITEMS=${VIS_ITEMS:-all}
TOPK=${TOPK:-20}
MAX_POINTS=${MAX_POINTS:-50000}
QUERY_LIMIT=${QUERY_LIMIT:-0}
QUERY_CHUNK=${QUERY_CHUNK:-64}
RANK_STRENGTH=${RANK_STRENGTH:-0}
OVERWRITE=${OVERWRITE:-0}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

IFS="," read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "No GPUs configured" >&2; exit 2; }
[[ -f "$STAGE1_CKPT" ]] || { echo "Missing Stage-1 checkpoint: $STAGE1_CKPT" >&2; exit 2; }
[[ -f "$DCR_CHECKPOINT" ]] || { echo "Missing DCR checkpoint: $DCR_CHECKPOINT" >&2; exit 2; }

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

mkdir -p "$VIS_ROOT/logs"
for s in "${!GPU_IDS[@]}"; do
  args=(
    "$ROOT_DIR/visualize_dcr_behavior.py"
    --dataset-root "$DATASET_ROOT"
    --stage1-checkpoint "$STAGE1_CKPT"
    --dcr-checkpoint "$DCR_CHECKPOINT"
    --output-root "$VIS_ROOT"
    --splits "$SPLITS"
    --frames "$FRAMES"
    --cases "$CASES"
    --items "$VIS_ITEMS"
    --topk "$TOPK"
    --max-points "$MAX_POINTS"
    --query-limit "$QUERY_LIMIT"
    --query-chunk "$QUERY_CHUNK"
    --rank-strength "$RANK_STRENGTH"
    --shard-id "$s"
    --num-shards "${#GPU_IDS[@]}"
  )
  [[ -n "$SCENES" ]] && args+=(--scenes "$SCENES")
  [[ -n "$AIR_CHECKPOINT" ]] && args+=(--air-checkpoint "$AIR_CHECKPOINT")
  [[ -n "$OFFICIAL_ROOT" ]] && args+=(--official-root "$OFFICIAL_ROOT")
  [[ "$OVERWRITE" == 1 ]] && args+=(--overwrite)
  echo "[VIS] shard=$s gpu=${GPU_IDS[$s]}"
  setsid env CUDA_VISIBLE_DEVICES="${GPU_IDS[$s]}" "$PYTHON_BIN" -u "${args[@]}" \
    >"$VIS_ROOT/logs/shard_${s}.log" 2>&1 &
  PIDS+=("$!"); NAMES+=("vis-shard-$s")
done

for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[VIS ERROR] ${NAMES[$i]} failed; see $VIS_ROOT/logs" >&2
    exit 1
  fi
done
PIDS=()
echo "[VIS] complete: $VIS_ROOT"
