#!/usr/bin/env bash
# Single controlled run. Existing dataset annotations only; no feature/action cache.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
VARIANT=${VARIANT:-moge_rayrope}
case "$VARIANT" in
  baseline) DEFAULT_M=0; DEFAULT_R=0; DEFAULT_ENCODING=expected; DEFAULT_UNCERTAINTY=fixed ;;
  moge) DEFAULT_M=1; DEFAULT_R=0; DEFAULT_ENCODING=expected; DEFAULT_UNCERTAINTY=fixed ;;
  rayrope) DEFAULT_M=0; DEFAULT_R=1; DEFAULT_ENCODING=expected; DEFAULT_UNCERTAINTY=fixed ;;
  moge_rayrope) DEFAULT_M=1; DEFAULT_R=1; DEFAULT_ENCODING=expected; DEFAULT_UNCERTAINTY=fixed ;;
  attention_none) DEFAULT_M=0; DEFAULT_R=1; DEFAULT_ENCODING=none; DEFAULT_UNCERTAINTY=fixed ;;
  rayrope_point) DEFAULT_M=0; DEFAULT_R=1; DEFAULT_ENCODING=point; DEFAULT_UNCERTAINTY=fixed ;;
  moge_rayrope_point) DEFAULT_M=1; DEFAULT_R=1; DEFAULT_ENCODING=point; DEFAULT_UNCERTAINTY=fixed ;;
  moge_rayrope_learned) DEFAULT_M=1; DEFAULT_R=1; DEFAULT_ENCODING=expected; DEFAULT_UNCERTAINTY=learned ;;
  rayrope_learned) DEFAULT_M=0; DEFAULT_R=1; DEFAULT_ENCODING=expected; DEFAULT_UNCERTAINTY=learned ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac
USE_MOGE=${USE_MOGE:-$DEFAULT_M}
USE_RAYROPE=${USE_RAYROPE:-$DEFAULT_R}
RAY_ENCODING=${RAY_ENCODING:-$DEFAULT_ENCODING}
UNCERTAINTY=${UNCERTAINTY:-$DEFAULT_UNCERTAINTY}
UNCERTAINTY_LOSS=${UNCERTAINTY_LOSS:-interval}
GNTRANS_RGB_ROOT=${GNTRANS_RGB_ROOT:-}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
LABEL_FOLDER=${LABEL_FOLDER:-economic_grasp_label_300views_extend_angle_cdf_depth}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20/$VARIANT}
TRAIN_ROOT=${TRAIN_ROOT:-$WORK_ROOT/train}
TEST_ROOT=${TEST_ROOT:-$WORK_ROOT/test_native}
GPUS=${GPUS:-0,1,2}
PHASES=${PHASES:-train,infer,eval}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
ENCODER=${ENCODER:-vitb}
POSE_MODE=${POSE_MODE:-global_film}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-3}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-1}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.2}
EVAL_FRACTION=${EVAL_FRACTION:-0.1}
USE_FUSE_DEPTH=${USE_FUSE_DEPTH:-1}
LR=${LR:-0.0003}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
GRAD_CLIP=${GRAD_CLIP:-1.0}
GROUP_CHUNK=${GROUP_CHUNK:-64}
SEEDS=${SEEDS:-1024}
WORKERS=${WORKERS:-2}
EVAL_WORKERS=${EVAL_WORKERS:-1}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
RESUME=${RESUME:-0}
COLLISION=${COLLISION:-both}
SEED=${SEED:-0}
DEPTH_BIAS_MM=${DEPTH_BIAS_MM:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  echo 'INIT_CHECKPOINT is intentionally unsupported: this suite uses fresh task heads, or RESUME=1 of its own run.' >&2; exit 2
fi
if [[ -n "${GRAD_ACCUM:-}" ]]; then
  echo 'No gradient accumulation is implemented. Change BATCH_SIZE only.' >&2; exit 2
fi
IFS=, read -ra IDS <<< "$GPUS"
IFS=, read -ra PP <<< "$PHASES"
IFS=, read -ra SS <<< "$SPLITS"
declare -A seen=()
[[ ${#IDS[@]} -gt 0 ]] || exit 2
for gpu in "${IDS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -z "${seen[$gpu]:-}" ]] || { echo "Invalid/repeated GPU $gpu" >&2; exit 2; }
  seen[$gpu]=1
done
for split in "${SS[@]}"; do
  [[ "$split" == test_seen || "$split" == test_similar || "$split" == test_novel ]] || exit 2
done
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || exit 2
[[ "$COLLISION" == on || "$COLLISION" == off || "$COLLISION" == both ]] || exit 2
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || exit 2
if (( ${#IDS[@]}*BATCH_SIZE != 9 )); then echo '[WARN] Effective batch differs from matched default 9.' >&2; fi
command -v setsid >/dev/null
mkdir -p "$WORK_ROOT/logs"
resume=(); [[ "$RESUME" == 1 ]] && resume=(--resume)
PIDS=()
cleanup(){ for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
launch(){ local gpu="$1" log="$2"; shift 2; setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$log" 2>&1 & PIDS+=("$!"); }
wait_all(){ local pid; for pid in "${PIDS[@]}"; do wait "$pid" || { cleanup; return 1; }; done; PIDS=(); }
printf '[MR20] variant=%s MoGe=%s RayRoPE=%s encoding=%s uncertainty=%s GPUs=%s batch/GPU=%s epochs=%s\n' \
 "$VARIANT" "$USE_MOGE" "$USE_RAYROPE" "$RAY_ENCODING" "$UNCERTAINTY" "$GPUS" "$BATCH_SIZE" "$EPOCHS"
MIX_ARGS=()
if [[ -n "$GNTRANS_RGB_ROOT" ]]; then MIX_ARGS+=(--gntrans-rgb-root "$GNTRANS_RGB_ROOT"); fi
for phase in "${PP[@]}"; do
  case "$phase" in
    train)
      launch "$GPUS" "$WORK_ROOT/logs/train.log" -m torch.distributed.run --standalone \
        --nproc_per_node="${#IDS[@]}" "$ROOT/train_moge_rayrope.py" \
        --dataset-root "$DATASET_ROOT" --label-folder "$LABEL_FOLDER" --output-root "$TRAIN_ROOT" \
        --encoder "$ENCODER" --pose-mode "$POSE_MODE" --use-moge "$USE_MOGE" --use-rayrope "$USE_RAYROPE" \
        --ray-encoding "$RAY_ENCODING" --uncertainty "$UNCERTAINTY" \
        --uncertainty-loss "$UNCERTAINTY_LOSS" "${MIX_ARGS[@]}" \
        --ray-apply-vo "${RAY_APPLY_VO:-1}" --shape-tokens "${SHAPE_TOKENS:-0}" \
        --fixed-halfwidth "${FIXED_HALFWIDTH:-0.02}" --ray-radius-px "${RAY_RADIUS_PX:-40}" \
        --ray-grid "${RAY_GRID:-7}" --group-chunk "$GROUP_CHUNK" --seeds "$SEEDS" \
        --shape-global-weight "${SHAPE_GLOBAL_WEIGHT:-1}" --shape-local-weight "${SHAPE_LOCAL_WEIGHT:-0.5}" \
        --reprojection-weight "${REPROJECTION_WEIGHT:-0.05}" --interval-weight "${INTERVAL_WEIGHT:-0.1}" \
        --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --train-fraction "$TRAIN_FRACTION" \
        --eval-fraction "$EVAL_FRACTION" --use-fuse-depth "$USE_FUSE_DEPTH" \
        --lr "$LR" --weight-decay "$WEIGHT_DECAY" --grad-clip "$GRAD_CLIP" \
        --workers "$WORKERS" --eval-workers "$EVAL_WORKERS" --seed "$SEED" --log-every "${LOG_EVERY:-20}" \
        --max-train-frames "${MAX_TRAIN_FRAMES:-0}" --max-val-frames "${MAX_VAL_FRAMES:-0}" \
        --max-steps "${MAX_STEPS:-0}" "${resume[@]}"
      wait_all ;;
    infer)
      [[ -f "$TRAIN_ROOT/checkpoint_latest.pt" ]] || { echo 'Missing latest checkpoint' >&2; exit 2; }
      if [[ "$DEPTH_BIAS_MM" != 0 && "$TEST_ROOT" == "$WORK_ROOT/test_native" ]]; then
        echo 'Nonzero DEPTH_BIAS_MM requires an explicit, separate TEST_ROOT.' >&2; exit 2
      fi
      for split in "${SS[@]}"; do
        for shard in "${!IDS[@]}"; do
          launch "${IDS[$shard]}" "$WORK_ROOT/logs/infer_${split}_${shard}.log" "$ROOT/inference_moge_rayrope.py" \
            --checkpoint "$TRAIN_ROOT/checkpoint_latest.pt" --dataset-root "$DATASET_ROOT" \
            --output-root "$TEST_ROOT" --split "$split" --batch-size "$INFER_BATCH_SIZE" \
            --workers "$EVAL_WORKERS" --shard-id "$shard" --num-shards "${#IDS[@]}" \
            --collision "$COLLISION" --collision-thresh "${COLLISION_THRESH:-0.01}" \
            --voxel-size "${COLLISION_VOXEL_SIZE:-0.01}" --approach-dist "${COLLISION_APPROACH_DIST:-0.05}" \
            --depth-bias-mm "$DEPTH_BIAS_MM" --max-frames "${INFER_MAX_FRAMES:-0}" "${resume[@]}"
        done
        wait_all
      done ;;
    eval)
      modes=("$COLLISION"); [[ "$COLLISION" != both ]] || modes=(on off)
      for mode in "${modes[@]}"; do
        for split in "${SS[@]}"; do
          launch "${IDS[0]}" "$WORK_ROOT/logs/eval_${mode}_${split}.log" "$ROOT/eval_moge_rayrope.py" \
            --dataset-root "$DATASET_ROOT" --inference-root "$TEST_ROOT/test_collision_$mode" \
            --split "$split" --workers "$OFFICIAL_WORKERS" "${resume[@]}"
          wait_all
        done
      done ;;
    *) echo "Unknown phase $phase" >&2; exit 2 ;;
  esac
done
echo "[MR20] Completed requested phases. Outputs: $WORK_ROOT"
