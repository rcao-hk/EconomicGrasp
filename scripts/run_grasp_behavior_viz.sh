#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
DCR_BASE_ROOT=${DCR_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dcr_cva_10pct}
DCR_CHECKPOINT=${DCR_CHECKPOINT:-$DCR_BASE_ROOT/train/cdf/checkpoint_best.pt}
E1_CHECKPOINT=${E1_CHECKPOINT:-}
AIR_CHECKPOINT=${AIR_CHECKPOINT:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/behavior_viz}

GPUS=${GPUS:-1,2,3}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
FRAMES=${FRAMES:-0,128,255}
CASES=${CASES:-nominal,bias:-25,bias:25,smooth:10}
ITEMS=${ITEMS:-core}
TOPK=${TOPK:-50}
LOCAL_QUERIES=${LOCAL_QUERIES:-8}
QUERY_LIMIT=${QUERY_LIMIT:-0}
QUERY_CHUNK=${QUERY_CHUNK:-64}
POINT_STRIDE=${POINT_STRIDE:-3}
EVAL_METHODS=${EVAL_METHODS:-dcr}
EVAL_CASES=${EVAL_CASES:-nominal}
EVAL_TOPK=${EVAL_TOPK:-50}
EVAL_VOXEL_SIZE=${EVAL_VOXEL_SIZE:-0.008}
NMS_TRANS_TH=${NMS_TRANS_TH:-0.03}
NMS_ROT_DEG=${NMS_ROT_DEG:-30}
MAX_WIDTH=${MAX_WIDTH:-0.1}
SCENE_IDS=${SCENE_IDS:-100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185}
MAX_SCENES=${MAX_SCENES:-0}
RESUME=${RESUME:-1}
SEED=${SEED:-2051}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

[[ -f "$STAGE1_CKPT" ]] || { echo "Missing Stage-1 checkpoint: $STAGE1_CKPT" >&2; exit 2; }
[[ -f "$DCR_CHECKPOINT" ]] || { echo "Missing DCR checkpoint: $DCR_CHECKPOINT" >&2; exit 2; }
IFS="," read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -gt 0 ]] || exit 2

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

mkdir -p "$OUTPUT_ROOT/logs"
for s in "${!GPU_IDS[@]}"; do
  args=(
    "$ROOT_DIR/visualize_grasp_behavior.py"
    --dataset-root "$DATASET_ROOT"
    --stage1-checkpoint "$STAGE1_CKPT"
    --dcr-checkpoint "$DCR_CHECKPOINT"
    --output-root "$OUTPUT_ROOT"
    --camera realsense
    --splits "$SPLITS"
    --frames "$FRAMES"
    --cases "$CASES"
    --items "$ITEMS"
    --topk "$TOPK"
    --local-queries "$LOCAL_QUERIES"
    --query-limit "$QUERY_LIMIT"
    --query-chunk "$QUERY_CHUNK"
    --point-stride "$POINT_STRIDE"
    --eval-methods "$EVAL_METHODS"
    --eval-cases "$EVAL_CASES"
    --eval-topk "$EVAL_TOPK"
    --eval-voxel-size "$EVAL_VOXEL_SIZE"
    --nms-trans-th "$NMS_TRANS_TH"
    --nms-rot-deg "$NMS_ROT_DEG"
    --max-width "$MAX_WIDTH"
    --max-scenes "$MAX_SCENES"
    --shard-id "$s"
    --num-shards "${#GPU_IDS[@]}"
    --device cuda:0
    --seed "$SEED"
  )
  [[ -n "$E1_CHECKPOINT" ]] && args+=(--e1-checkpoint "$E1_CHECKPOINT")
  [[ -n "$AIR_CHECKPOINT" ]] && args+=(--air-checkpoint "$AIR_CHECKPOINT")
  [[ -n "$SCENE_IDS" ]] && args+=(--scene-ids "$SCENE_IDS")
  [[ "$RESUME" == 1 ]] && args+=(--resume)

  echo "[GRASP-VIZ] shard=$s GPU=${GPU_IDS[$s]} log=$OUTPUT_ROOT/logs/shard_${s}.log"
  setsid env CUDA_VISIBLE_DEVICES="${GPU_IDS[$s]}" "$PYTHON_BIN" -u "${args[@]}" \
    >>"$OUTPUT_ROOT/logs/shard_${s}.log" 2>&1 &
  PIDS+=("$!"); NAMES+=("viz/$s")
done

for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[GRASP-VIZ ERROR] ${NAMES[$i]} failed" >&2
    exit 1
  fi
done
PIDS=(); NAMES=()

(
  cd "$ROOT_DIR"
  "$PYTHON_BIN" -m tools.merge_grasp_behavior_viz --root "$OUTPUT_ROOT"
)
echo "[GRASP-VIZ] complete: $OUTPUT_ROOT/index.html"
