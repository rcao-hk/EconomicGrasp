#!/usr/bin/env bash
set -euo pipefail
: "${DATASET_ROOT:?Set DATASET_ROOT to GraspNet root}"
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT to GN-Trans RGB root}"
: "${OUTPUT_ROOT:?Set a NEW OUTPUT_ROOT outside all existing experiments}"
GPUS="${GPUS:-0,1,2}"
BATCH_SIZE="${BATCH_SIZE:-3}"
MAX_EPOCH="${MAX_EPOCH:-21}"
LR="${LR:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-3}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
DEPTH_WEIGHT_DECAY="${DEPTH_WEIGHT_DECAY:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
USE_FUSE_DEPTH="${USE_FUSE_DEPTH:-1}"
GNTRANS_FUSED_BACKGROUND="${GNTRANS_FUSED_BACKGROUND:-0}"
SHARED_DEPTH_CKPT="${SHARED_DEPTH_CKPT:-}"
SEED="${SEED:-0}"
for name in USE_FUSE_DEPTH GNTRANS_FUSED_BACKGROUND; do
  if [[ "${!name}" != "0" && "${!name}" != "1" ]]; then
    echo "${name} must be 0 or 1" >&2; exit 2
  fi
done
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then echo 'No GPUs specified' >&2; exit 2; fi
if [[ -d "$OUTPUT_ROOT" && -n "$(ls -A "$OUTPUT_ROOT")" ]]; then
  echo "Refusing to overwrite a previous run in OUTPUT_ROOT=$OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPUS"
extra=()
if [[ "$USE_FUSE_DEPTH" == 1 ]]; then extra+=(--use_fuse_depth); fi
if [[ "$GNTRANS_FUSED_BACKGROUND" == 1 ]]; then extra+=(--gntrans_fused_background); fi
if [[ -n "$SHARED_DEPTH_CKPT" ]]; then
  if [[ ! -f "$SHARED_DEPTH_CKPT" ]]; then echo "Missing SHARED_DEPTH_CKPT=$SHARED_DEPTH_CKPT" >&2; exit 1; fi
  extra+=(--shared_depth_checkpoint "$SHARED_DEPTH_CKPT")
fi
echo "[MIXED-POINT] GPUS=$GPUS batch_per_gpu=$BATCH_SIZE epochs=$MAX_EPOCH"
echo "[MIXED-POINT] Original fused background=$USE_FUSE_DEPTH GN-Trans fused background=$GNTRANS_FUSED_BACKGROUND"
(set -o pipefail; torchrun --standalone --nproc_per_node="${#GPU_ARRAY[@]}" \
  train_point_depth_pose_mixed_ddp.py \
  --dataset_root "$DATASET_ROOT" \
  --gntrans_rgb_root "$GNTRANS_RGB_ROOT" \
  --log_dir "$OUTPUT_ROOT" \
  --camera realsense \
  --num_point 20000 \
  --batch_size "$BATCH_SIZE" \
  --max_epoch "$MAX_EPOCH" \
  --learning_rate "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --depth_weight_decay "$DEPTH_WEIGHT_DECAY" \
  --num_workers "$NUM_WORKERS" \
  --eval_num_workers "$EVAL_NUM_WORKERS" \
  --ckpt_save_interval 5 \
  --mix_train_fraction 0.1 \
  --mix_eval_fraction 0.1 \
  --min_depth 0.2 \
  --max_depth 1.0 \
  --depth_prob_loss_weight 10 \
  --graspness_mode scene \
  --pose_depth_mode global_film \
  --seed "$SEED" \
  "${extra[@]}" 2>&1 | tee "$OUTPUT_ROOT/train.log")
if [[ "$MAX_EPOCH" -ge 20 ]]; then
  echo "[MIXED-POINT] Finished; frozen test checkpoint $OUTPUT_ROOT/checkpoint_epoch_019.tar"
else
  echo "[MIXED-POINT] Smoke/short training finished; NO official e19 checkpoint expected"
fi
