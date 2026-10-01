#!/usr/bin/env bash
# P0 diagnostic: local (A x D) action selection x global cross-query scoring 2x2.
#
# Four conditions from the SAME forward:
#   lbase_gbase : Base local, Base global
#   lfull_gbase : Full local, Base global
#   lbase_gfull : Base local, Full global
#   lfull_gfull : Full local, Full global
#
# Collision-on is the primary benchmark result; collision-off is retained as a
# secondary scoring/mechanism diagnostic. No training and no cache generation.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_mixed_matched/train/checkpoint_latest.pt}
P0_ROOT=${P0_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/mgf_p0_1_frozen_online}
FULL_CONTROL_CHECKPOINT=${FULL_CONTROL_CHECKPOINT:-$P0_ROOT/full/train/checkpoint_latest.pt}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/mgf_p0_local_global_2x2}

GPUS=${GPUS:-0,1,2}
PHASES=${PHASES:-infer,eval}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
EVAL_FRACTION=${EVAL_FRACTION:-0.1}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-1}
WORKERS=${WORKERS:-2}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-10}
COLLISION=${COLLISION:-both}
COLLISION_THRESH=${COLLISION_THRESH:-0.01}
COLLISION_VOXEL_SIZE=${COLLISION_VOXEL_SIZE:-0.01}
COLLISION_APPROACH_DIST=${COLLISION_APPROACH_DIST:-0.05}
INFER_MAX_FRAMES=${INFER_MAX_FRAMES:-0}
RESUME=${RESUME:-0}
BOOTSTRAP=${BOOTSTRAP:-20000}

CONDITIONS=(lbase_gbase lfull_gbase lbase_gfull lfull_gfull)

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

command -v setsid >/dev/null
[[ -f "$SOURCE_CHECKPOINT" ]] || { echo "[ERROR] Missing source: $SOURCE_CHECKPOINT" >&2; exit 2; }
[[ -f "$FULL_CONTROL_CHECKPOINT" ]] || { echo "[ERROR] Missing Full control: $FULL_CONTROL_CHECKPOINT" >&2; exit 2; }
[[ "$COLLISION" == off || "$COLLISION" == on || "$COLLISION" == both ]] || {
  echo "[ERROR] COLLISION=off|on|both" >&2; exit 2;
}

IFS=, read -ra IDS <<< "$GPUS"
IFS=, read -ra SS <<< "$SPLITS"
IFS=, read -ra PP <<< "$PHASES"
[[ "${#IDS[@]}" -gt 0 ]] || { echo "[ERROR] No GPUs" >&2; exit 2; }
declare -A used=()
for id in "${IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ && -z "${used[$id]:-}" ]] || {
    echo "[ERROR] Invalid/repeated GPU: $id" >&2; exit 2;
  }
  used[$id]=1
done

resume=()
[[ "$RESUME" == 1 ]] && resume=(--resume)
PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch() {
  local gpu="$1" log="$2"
  shift 2
  mkdir -p "$(dirname "$log")"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$log" 2>&1 &
  PIDS+=("$!")
}
wait_all() {
  local pid
  for pid in "${PIDS[@]}"; do
    wait "$pid" || { cleanup; return 1; }
  done
  PIDS=()
}

echo "============================================================"
echo "[MGF LOCAL x GLOBAL 2x2]"
echo "  source       : $SOURCE_CHECKPOINT"
echo "  full control : $FULL_CONTROL_CHECKPOINT"
echo "  output       : $WORK_ROOT"
echo "  GPUs         : $GPUS"
echo "  splits       : $SPLITS"
echo "  collision    : $COLLISION (ON is primary)"
echo "============================================================"

for phase in "${PP[@]}"; do
  case "$phase" in
    infer)
      for split in "${SS[@]}"; do
        echo "[2x2] inference split=$split"
        for shard in "${!IDS[@]}"; do
          launch "${IDS[$shard]}" "$WORK_ROOT/logs/infer_${split}_${shard}.log"             "$ROOT/mgf_p0_local_global_2x2.py"             --source-checkpoint "$SOURCE_CHECKPOINT"             --full-control-checkpoint "$FULL_CONTROL_CHECKPOINT"             --dataset-root "$DATASET_ROOT"             --output-root "$WORK_ROOT"             --split "$split"             --eval-fraction "$EVAL_FRACTION"             --shard-id "$shard" --num-shards "${#IDS[@]}"             --batch-size "$INFER_BATCH_SIZE" --workers "$WORKERS"             --collision "$COLLISION"             --collision-thresh "$COLLISION_THRESH"             --voxel-size "$COLLISION_VOXEL_SIZE"             --approach-dist "$COLLISION_APPROACH_DIST"             --max-frames "$INFER_MAX_FRAMES"             "${resume[@]}"
        done
        wait_all
      done
      ;;

    eval)
      modes=("$COLLISION")
      [[ "$COLLISION" != both ]] || modes=(on off)
      for mode in "${modes[@]}"; do
        for cond in "${CONDITIONS[@]}"; do
          for split in "${SS[@]}"; do
            echo "[2x2] official eval collision=$mode cond=$cond split=$split"
            launch "${IDS[0]}" "$WORK_ROOT/logs/eval_${mode}_${cond}_${split}.log"               "$ROOT/eval_metric_grasp_field.py"               --dataset-root "$DATASET_ROOT"               --inference-root "$WORK_ROOT/$cond/test_collision_$mode"               --split "$split" --workers "$OFFICIAL_WORKERS" "${resume[@]}"
            wait_all
          done
        done
      done
      ;;

    *)
      echo "[ERROR] Unknown phase: $phase" >&2
      exit 2
      ;;
  esac
done

# Formal comparison requires complete official results. Skip automatically for a
# smoke run or infer-only invocation.
if [[ "$INFER_MAX_FRAMES" == 0 && ",$PHASES," == *",eval,"* ]]; then
  "$PYTHON_BIN" "$ROOT/mgf_p0_local_global_2x2_compare.py"     --root "$WORK_ROOT" --collision "$COLLISION" --bootstrap "$BOOTSTRAP"
fi

echo
echo "============================================================"
echo "[MGF LOCAL x GLOBAL 2x2] complete"
echo "Primary comparison:"
echo "  $WORK_ROOT/comparison/comparison.md"
echo "============================================================"
