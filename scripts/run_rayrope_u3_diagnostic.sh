#!/usr/bin/env bash
# Frozen U3 halfwidth: no training, same inference/evaluation protocol as native U3.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-/home/robotarm/miniconda3/envs/grasp/bin/python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
U3_ROOT=${U3_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_renderfg_tsdfbg_v2/formal/U3}
U3_CHECKPOINT=${U3_CHECKPOINT:-$U3_ROOT/train/checkpoint_latest.pt}
DIAG_ROOT=${DIAG_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_renderfg_tsdfbg_v2/diagnostics/u3_halfwidth}
NATIVE_ROOT=${NATIVE_ROOT:-$U3_ROOT/test_native}
GPUS=${GPUS:-0,1,2}
MODES=${MODES:-"learned point constant_train fixed20 shuffle"}
SPLITS=${SPLITS:-"test_seen test_similar test_novel"}
PHASE=${PHASE:-all}
RESUME=${RESUME:-0}
REQUIRE_NATIVE_PARITY=${REQUIRE_NATIVE_PARITY:-1}
SHUFFLE_SEED=${SHUFFLE_SEED:-20261013}
MAX_FRAMES=${MAX_FRAMES:-0}
[[ -f "$U3_CHECKPOINT" && -f "$(dirname "$U3_CHECKPOINT")/metrics.json" ]] || {
  echo "Missing U3 checkpoint or training metrics" >&2; exit 2;
}
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || exit 2
[[ "$PHASE" == all || "$PHASE" == infer || "$PHASE" == eval || "$PHASE" == compare ]] || exit 2
[[ "$(realpath -m "$DIAG_ROOT")" != "$(realpath -m "$U3_ROOT")" ]] || exit 2
IFS=, read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 3 ]] || { echo "Use 3 available GPUs" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && "$gpu" != 4 ]] || { echo "Invalid/prohibited GPU $gpu" >&2; exit 2; }
done
if [[ "$REQUIRE_NATIVE_PARITY" == 1 && ! -d "$NATIVE_ROOT" ]]; then
  echo "Original U3 inference missing at $NATIVE_ROOT" >&2; exit 2
fi
mkdir -p "$DIAG_ROOT/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
resume=(); [[ "$RESUME" == 1 ]] && resume=(--resume)
echo "[U3-DIAG] checkpoint=$U3_CHECKPOINT, output=$DIAG_ROOT"

if [[ "$PHASE" == all || "$PHASE" == infer ]]; then
  for mode in $MODES; do
    for split in $SPLITS; do
      extra=()
      if [[ "$mode" == learned && "$REQUIRE_NATIVE_PARITY" == 1 ]]; then
        extra+=(--native-reference-root "$NATIVE_ROOT")
      fi
      PIDS=()
      for shard in 0 1 2; do
        log="$DIAG_ROOT/logs/infer_$(printf '%s_%s_%s' "$mode" "$split" "$shard").log"
        env CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" "$PYTHON_BIN" -u \
          scripts/diagnose_rayrope_u3.py --mode "$mode" "${extra[@]}" \
          --shuffle-seed "$SHUFFLE_SEED" --checkpoint "$U3_CHECKPOINT" \
          --dataset-root "$DATASET_ROOT" --output-root "$DIAG_ROOT/$mode" \
          --split "$split" --batch-size 3 --workers 1 \
          --shard-id "$shard" --num-shards 3 --collision both \
          --collision-thresh .01 --voxel-size .01 --approach-dist .05 \
          --max-frames "$MAX_FRAMES" "${resume[@]}" >"$log" 2>&1 &
        PIDS+=("$!")
      done
      for pid in "${PIDS[@]}"; do
        wait "$pid" || { for q in "${PIDS[@]}"; do kill "$q" 2>/dev/null || true; done; exit 1; }
      done
      echo "[U3-DIAG] infer complete: $mode / $split"
    done
  done
fi

if [[ "$PHASE" == all || "$PHASE" == eval ]]; then
  for mode in $MODES; do
    for collision in on off; do
      for split in $SPLITS; do
        log="$DIAG_ROOT/logs/eval_$(printf '%s_%s_%s' "$mode" "$collision" "$split").log"
        env CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PYTHON_BIN" -u eval_moge_rayrope.py \
          --dataset-root "$DATASET_ROOT" \
          --inference-root "$DIAG_ROOT/$mode/test_collision_$collision" \
          --split "$split" --workers 2 "${resume[@]}" >"$log" 2>&1
        echo "[U3-DIAG] eval complete: $mode / $collision / $split"
      done
    done
  done
fi

if [[ "$PHASE" == all || "$PHASE" == compare ]]; then
  "$PYTHON_BIN" scripts/compare_rayrope_u3_halfwidth.py \
    --root "$DIAG_ROOT" --modes "$MODES" --splits "$SPLITS"
fi
echo "[U3-DIAG] complete: $DIAG_ROOT"
