#!/usr/bin/env bash
# Online 10%-frame experiment. No P0/P1 cache/mining stage.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_metric_grasp_field_10pct}
# An explicitly empty INIT_CHECKPOINT requests training new task/metric heads.
INIT_CHECKPOINT=${INIT_CHECKPOINT-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
GPUS=${GPUS:-0,1,2,3,4,5}
INFER_GPUS=${INFER_GPUS:-$GPUS}
PHASES=${PHASES:-train,infer,eval}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
CHECKPOINT_KIND=${CHECKPOINT_KIND:-best}
RESUME=${RESUME:-0}
ENCODER=${ENCODER:-vitb}
POSE_MODE=${POSE_MODE:-global_film}
SEED_MODE=${SEED_MODE:-image_fps}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-0.0001}
GEOMETRY_LR=${GEOMETRY_LR:-0.00001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
SAMPLE_FRACTION=${SAMPLE_FRACTION:-0.1}
WORKERS=${WORKERS:-2}
EVAL_WORKERS=${EVAL_WORKERS:-1}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
M_POINT=${M_POINT:-1024}
GROUP_CHUNK=${GROUP_CHUNK:-256}
ACTION_CHUNK=${ACTION_CHUNK:-256}
FIELD_BINS=${FIELD_BINS:-160}
FIELD_HIDDEN=${FIELD_HIDDEN:-64}
FIELD_STRIDE=${FIELD_STRIDE:-4}
EVIDENCE_MODE=${EVIDENCE_MODE:-learned}
SURFACE_EPSILON=${SURFACE_EPSILON:-0.005}
PRIOR_SIGMA=${PRIOR_SIGMA:-0.03}
FIXED_SIGMA=${FIXED_SIGMA:-0.02}
PROFILE_WEIGHT=${PROFILE_WEIGHT:-1}
PROFILE_MEAN_WEIGHT=${PROFILE_MEAN_WEIGHT:-10}
BASE_CDF_WEIGHT=${BASE_CDF_WEIGHT:-0.25}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
MAX_STEPS=${MAX_STEPS:-0}
INFER_MAX_FRAMES=${INFER_MAX_FRAMES:-0}
SEED=${SEED:-2117}
AMP=${AMP:-0}
TOP4=${TOP4:-0}
TRAIN_ROOT=${TRAIN_ROOT:-$WORK_ROOT/train}
TEST_ROOT=${TEST_ROOT:-$WORK_ROOT/test_$CHECKPOINT_KIND}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1
command -v setsid >/dev/null || { echo 'setsid is required' >&2; exit 2; }
[[ "$CHECKPOINT_KIND" == best || "$CHECKPOINT_KIND" == latest ]] || { echo 'CHECKPOINT_KIND=best|latest' >&2; exit 2; }
[[ -d "$DATASET_ROOT" ]] || { echo "Missing dataset: $DATASET_ROOT" >&2; exit 2; }
[[ -f "checkpoints/depth_anything_v2_${ENCODER}.pth" ]] || { echo "Missing official DAV2 $ENCODER checkpoint" >&2; exit 2; }
IFS=',' read -r -a TRAIN_IDS <<< "$GPUS"
IFS=',' read -r -a INFER_IDS <<< "$INFER_GPUS"
IFS=',' read -r -a SPLIT_IDS <<< "$SPLITS"
IFS=',' read -r -a STEPS <<< "$PHASES"
for id in "${TRAIN_IDS[@]}" "${INFER_IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: $id" >&2; exit 2; }
done
for split in "${SPLIT_IDS[@]}"; do
  [[ "$split" == test_seen || "$split" == test_similar || "$split" == test_novel ]] || { echo "Invalid split: $split" >&2; exit 2; }
done
PIDS=(); NAMES=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
launch() {
  local name="$1" gpu="$2" logfile="$3"; shift 3
  mkdir -p "$(dirname "$logfile")"
  echo "[MGF] $name gpu=$gpu log=$logfile"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$logfile" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}
wait_wave() {
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[MGF ERROR] ${NAMES[$i]} failed; inspect logs" >&2
      exit 1
    fi
  done
  PIDS=(); NAMES=()
}
resume=(); [[ "$RESUME" == 1 ]] && resume+=(--resume)
amp=(); [[ "$AMP" == 1 ]] && amp+=(--amp)
top4=(); [[ "$TOP4" == 1 ]] && top4+=(--top4)

for phase in "${STEPS[@]}"; do
  case "$phase" in
    train)
      [[ -z "$INIT_CHECKPOINT" || -f "$INIT_CHECKPOINT" ]] || { echo "Missing INIT_CHECKPOINT: $INIT_CHECKPOINT" >&2; exit 2; }
      launch train "$GPUS" "$WORK_ROOT/logs/train.log" \
        -m torch.distributed.run --standalone --nproc_per_node="${#TRAIN_IDS[@]}" \
        "$ROOT_DIR/train_metric_grasp_field.py" \
        --dataset-root "$DATASET_ROOT" --output-root "$TRAIN_ROOT" \
        --init-checkpoint "$INIT_CHECKPOINT" --encoder "$ENCODER" \
        --pose-mode "$POSE_MODE" --seed-mode "$SEED_MODE" \
        --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
        --lr "$LR" --geometry-lr "$GEOMETRY_LR" --weight-decay "$WEIGHT_DECAY" \
        --sample-fraction "$SAMPLE_FRACTION" --workers "$WORKERS" --eval-workers "$EVAL_WORKERS" \
        --m-point "$M_POINT" --group-chunk "$GROUP_CHUNK" --action-chunk "$ACTION_CHUNK" \
        --field-bins "$FIELD_BINS" --field-hidden "$FIELD_HIDDEN" --field-stride "$FIELD_STRIDE" \
        --evidence-mode "$EVIDENCE_MODE" --surface-epsilon "$SURFACE_EPSILON" \
        --prior-sigma "$PRIOR_SIGMA" --fixed-sigma "$FIXED_SIGMA" \
        --profile-weight "$PROFILE_WEIGHT" --profile-mean-weight "$PROFILE_MEAN_WEIGHT" \
        --base-cdf-weight "$BASE_CDF_WEIGHT" --max-train-frames "$MAX_TRAIN_FRAMES" \
        --max-val-frames "$MAX_VAL_FRAMES" --max-steps "$MAX_STEPS" --seed "$SEED" \
        "${amp[@]}" "${resume[@]}"
      wait_wave
      ;;
    infer)
      ckpt="$TRAIN_ROOT/checkpoint_${CHECKPOINT_KIND}.pt"
      [[ -f "$ckpt" ]] || { echo "Missing checkpoint: $ckpt" >&2; exit 2; }
      for split in "${SPLIT_IDS[@]}"; do
        for shard in "${!INFER_IDS[@]}"; do
          launch "infer/$split/$shard" "${INFER_IDS[$shard]}" "$WORK_ROOT/logs/infer_${CHECKPOINT_KIND}_${split}_${shard}.log" \
            "$ROOT_DIR/inference_metric_grasp_field.py" --dataset-root "$DATASET_ROOT" \
            --checkpoint "$ckpt" --output-root "$TEST_ROOT" --split "$split" \
            --shard-id "$shard" --num-shards "${#INFER_IDS[@]}" \
            --max-frames "$INFER_MAX_FRAMES" --workers "$EVAL_WORKERS" \
            "${top4[@]}" "${resume[@]}"
        done
        wait_wave
      done
      ;;
    eval)
      for split in "${SPLIT_IDS[@]}"; do
        launch "eval/$split" "${INFER_IDS[0]}" "$WORK_ROOT/logs/eval_${CHECKPOINT_KIND}_${split}.log" \
          "$ROOT_DIR/eval_metric_grasp_field.py" --dataset-root "$DATASET_ROOT" \
          --inference-root "$TEST_ROOT" --split "$split" --workers "$OFFICIAL_WORKERS" "${resume[@]}"
        wait_wave
      done
      ;;
    *) echo "Unknown PHASES entry: $phase (train,infer,eval)" >&2; exit 2 ;;
  esac
done
