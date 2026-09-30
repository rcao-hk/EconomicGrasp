#!/usr/bin/env bash
# P1-1: frozen-source ray-profile and relative-prior ablations; no new cache.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_mixed_matched/train/checkpoint_latest.pt}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/mgf_p1_1_frozen_online}
GPUS=${GPUS:-3,5,6}
VARIANTS=${VARIANTS:-profile_hard,profile_fixed,profile_learned,relative_metric_only,relative_scalar,relative_feature}
PHASES=${PHASES:-train,infer,eval}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-3}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-1}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.2}
EVAL_FRACTION=${EVAL_FRACTION:-0.1}
LR=${LR:-0.0003}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
RANKING_WEIGHT=${RANKING_WEIGHT:-0.1}
TEMPERATURE=${TEMPERATURE:-0.1}
WORKERS=${WORKERS:-2}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
COLLISION=${COLLISION:-both}
SEED=${SEED:-0}
RESUME=${RESUME:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} PYTHONUNBUFFERED=1
command -v setsid >/dev/null
[[ "$COLLISION" == off || "$COLLISION" == on || "$COLLISION" == both ]] || exit 2
IFS=, read -ra IDS <<< "$GPUS"; IFS=, read -ra VS <<< "$VARIANTS"; IFS=, read -ra SS <<< "$SPLITS"; IFS=, read -ra PP <<< "$PHASES"
declare -A used=()
for id in "${IDS[@]}"; do [[ "$id" =~ ^[0-9]+$ && -z "${used[$id]:-}" ]] || exit 2; used[$id]=1; done
resume=(); [[ "$RESUME" == 1 ]] && resume=(--resume)
PIDS=()
cleanup(){ for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
launch(){ local gpu="$1" log="$2"; shift 2; mkdir -p "$(dirname "$log")"; setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$log" 2>&1 & PIDS+=("$!"); }
wait_all(){ local p; for p in "${PIDS[@]}"; do wait "$p" || { cleanup; return 1; }; done; PIDS=(); }

for phase in "${PP[@]}"; do
  case "$phase" in
    train)
      for v in "${VS[@]}"; do
        echo "[P1-1] train $v GPUs=$GPUS additional_epochs=$EPOCHS"
        launch "$GPUS" "$WORK_ROOT/$v/logs/train.log" -m torch.distributed.run --standalone --nproc_per_node="${#IDS[@]}"           "$ROOT/mgf_p1_train.py" --source-checkpoint "$SOURCE_CHECKPOINT" --dataset-root "$DATASET_ROOT"           --output-root "$WORK_ROOT/$v/train" --family p1_1 --variant "$v" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"           --train-fraction "$TRAIN_FRACTION" --eval-fraction "$EVAL_FRACTION" --lr "$LR" --weight-decay "$WEIGHT_DECAY"           --ranking-weight "$RANKING_WEIGHT" --temperature "$TEMPERATURE" --seed "$SEED" --workers "$WORKERS"           --max-train-frames "${MAX_TRAIN_FRAMES:-0}" --max-val-frames "${MAX_VAL_FRAMES:-0}"           --max-steps "${MAX_STEPS:-0}" "${resume[@]}"
        wait_all
      done ;;
    infer)
      for v in "${VS[@]}"; do
        ck="$WORK_ROOT/$v/train/checkpoint_latest.pt"; [[ -f "$ck" ]] || { echo "Missing $ck" >&2; exit 2; }
        for split in "${SS[@]}"; do
          for shard in "${!IDS[@]}"; do
            launch "${IDS[$shard]}" "$WORK_ROOT/$v/logs/infer_${split}_${shard}.log" "$ROOT/mgf_p1_infer.py"               --source-checkpoint "$SOURCE_CHECKPOINT" --control-checkpoint "$ck" --dataset-root "$DATASET_ROOT"               --output-root "$WORK_ROOT/$v" --split "$split" --eval-fraction "$EVAL_FRACTION"               --shard-id "$shard" --num-shards "${#IDS[@]}" --batch-size "$INFER_BATCH_SIZE" --workers "$WORKERS"               --collision "$COLLISION" --max-frames "${INFER_MAX_FRAMES:-0}" "${resume[@]}"
          done
          wait_all
        done
      done ;;
    eval)
      modes=("$COLLISION"); [[ "$COLLISION" != both ]] || modes=(off on)
      for v in "${VS[@]}"; do
        for mode in "${modes[@]}"; do
          for split in "${SS[@]}"; do
            launch "${IDS[0]}" "$WORK_ROOT/$v/logs/eval_${mode}_${split}.log" "$ROOT/eval_metric_grasp_field.py"               --dataset-root "$DATASET_ROOT" --inference-root "$WORK_ROOT/$v/test_collision_$mode"               --split "$split" --workers "$OFFICIAL_WORKERS" "${resume[@]}"
            wait_all
          done
        done
      done ;;
    *) echo "Unknown PHASES=$phase" >&2; exit 2;;
  esac
done
"$PYTHON_BIN" "$ROOT/mgf_p1_compare.py" --root "$WORK_ROOT" --family p1_1 --variants "$VARIANTS"
