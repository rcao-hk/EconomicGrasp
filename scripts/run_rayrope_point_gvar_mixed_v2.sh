#!/usr/bin/env bash
# Point RayRoPE (control) / Point RayRoPE + GVAR (combined); Mixed-v2 only.
# Never modifies the archived U0/U3 or GVAR experiment entrypoints.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
VARIANT="${VARIANT:-rayrope_point}"
case "$VARIANT" in
  rayrope_point) USE_GVAR=0 ;;
  rayrope_point_gvar) USE_GVAR=1 ;;
  *) echo "VARIANT must be rayrope_point or rayrope_point_gvar" >&2; exit 2 ;;
esac
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT to paired GN-Trans rendered RGB dataset}"
if [[ "${USE_MOGE:-0}" != 0 || "${USE_RAYROPE:-1}" != 1 ||
      "${RAY_ENCODING:-point}" != point || "${UNCERTAINTY:-fixed}" != fixed ||
      "${UNCERTAINTY_LOSS:-interval}" != interval || "${SHAPE_TOKENS:-0}" != 0 ||
      "${RAY_APPLY_VO:-1}" != 1 ]]; then
  echo 'Feasibility study fixes MoGe-off, Point RayRoPE, no uncertainty head, VO on' >&2
  exit 2
fi
USE_MOGE=0; USE_RAYROPE=1; RAY_ENCODING=point
UNCERTAINTY=fixed; UNCERTAINTY_LOSS=interval
GVAR_READER_DIM="${GVAR_READER_DIM:-64}"
GVAR_READER_HEADS="${GVAR_READER_HEADS:-4}"
GVAR_ACTION_CHUNK="${GVAR_ACTION_CHUNK:-256}"
GVAR_ACTIVATION_CHECKPOINT="${GVAR_ACTIVATION_CHECKPOINT:-1}"
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
LABEL_FOLDER=${LABEL_FOLDER:-economic_grasp_label_300views_extend_angle_cdf_depth}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rayrope_point_gvar_mixed_v2/$VARIANT}
TRAIN_ROOT=${TRAIN_ROOT:-$WORK_ROOT/train}
TEST_ROOT=${TEST_ROOT:-$WORK_ROOT/test_native}
GPUS=${GPUS:-0,1,2}
PHASES=${PHASES:-train,infer,eval}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
ENCODER=${ENCODER:-vitb}
POSE_MODE=${POSE_MODE:-global_film}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-3}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-3}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.1}
EVAL_FRACTION=${EVAL_FRACTION:-0.1}
USE_FUSE_DEPTH=${USE_FUSE_DEPTH:-1}
if [[ "$TRAIN_FRACTION" != "0.1" || "$EVAL_FRACTION" != "0.1" || "$USE_FUSE_DEPTH" != "1" ]]; then
  echo 'Only 10% RS + 10% GN-Trans, 10% RS validation, Mixed-v2 fused RS GT allowed' >&2; exit 2
fi
LR=${LR:-0.0003}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
GRAD_CLIP=${GRAD_CLIP:-1.0}
GROUP_CHUNK=${GROUP_CHUNK:-32}
SEEDS=${SEEDS:-1024}
WORKERS=${WORKERS:-3}
EVAL_WORKERS=${EVAL_WORKERS:-4}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-16}
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
  [[ "$gpu" != 4 ]] || { echo 'GPU4 is excluded by previous server allocation policy' >&2; exit 2; }
  seen[$gpu]=1
done
for split in "${SS[@]}"; do
  [[ "$split" == test_seen || "$split" == test_similar || "$split" == test_novel ]] || exit 2
done
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || exit 2
[[ "$COLLISION" == on || "$COLLISION" == off || "$COLLISION" == both ]] || exit 2
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || exit 2
if (( ${#IDS[@]}*BATCH_SIZE != 9 )) && [[ "${MAX_STEPS:-0}" == 0 && "${MAX_TRAIN_FRAMES:-0}" == 0 ]]; then
  echo 'Formal comparison requires world_size×batch_per_gpu=9 (3×3).' >&2; exit 2
fi
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
printf '[RPG] variant=%s Point RayRoPE=1 GVAR=%s Mixed-v2 10%%+10%% GPUs=%s batch/GPU=%s epochs=%s\n' \
 "$VARIANT" "$USE_GVAR" "$GPUS" "$BATCH_SIZE" "$EPOCHS"
MIX_ARGS=(--gntrans-rgb-root "$GNTRANS_RGB_ROOT")
for phase in "${PP[@]}"; do
  case "$phase" in
    train)
      launch "$GPUS" "$WORK_ROOT/logs/train.log" -m torch.distributed.run --standalone \
        --nproc_per_node="${#IDS[@]}" "$ROOT/train_rayrope_gvar.py" \
        --dataset-root "$DATASET_ROOT" --label-folder "$LABEL_FOLDER" --output-root "$TRAIN_ROOT" \
        --joint-variant "$VARIANT" --gvar-reader-dim "$GVAR_READER_DIM" \
        --gvar-reader-heads "$GVAR_READER_HEADS" --gvar-action-chunk "$GVAR_ACTION_CHUNK" \
        --gvar-activation-checkpoint "$GVAR_ACTIVATION_CHECKPOINT" \
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
          launch "${IDS[$shard]}" "$WORK_ROOT/logs/infer_${split}_${shard}.log" "$ROOT/inference_rayrope_gvar.py" \
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
          launch "${IDS[0]}" "$WORK_ROOT/logs/eval_${mode}_${split}.log" "$ROOT/eval_rayrope_gvar.py" \
            --dataset-root "$DATASET_ROOT" --inference-root "$TEST_ROOT/test_collision_$mode" \
            --split "$split" --workers "$OFFICIAL_WORKERS" "${resume[@]}"
          wait_all
        done
      done ;;
    *) echo "Unknown phase $phase" >&2; exit 2 ;;
  esac
done
echo "[RPG] Completed requested phases. Outputs: $WORK_ROOT"
