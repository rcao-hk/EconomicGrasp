#!/usr/bin/env bash
# Same full-data original GraspNet 100% pose-aware depth training for two point heads.
set -euo pipefail
: "${DATASET_ROOT:?Set GraspNet-1B root}"
: "${OUTPUT_ROOT:?Set a NEW experiment output root}"
GPUS="${GPUS:-0,1,2}"
HEADS="${HEADS:-legacy,cdf}"
TRAIN_DOMAIN="${TRAIN_DOMAIN:-original}"
MIX_TRAIN_FRACTION="${MIX_TRAIN_FRACTION:-0.5}"
GNTRANS_RGB_ROOT="${GNTRANS_RGB_ROOT:-}"
GNTRANS_FUSED_BACKGROUND="${GNTRANS_FUSED_BACKGROUND:-0}"
MAX_EPOCH="${MAX_EPOCH:-21}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-3}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
LR="${LR:-0.0001}"
SEED="${SEED:-0}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
SHARED_DEPTH_CKPT="${SHARED_DEPTH_CKPT:-}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
IFS=',' read -r -a HEAD_ARRAY <<< "$HEADS"
if ((${#GPU_ARRAY[@]}<1)); then echo "No GPUs given" >&2; exit 2; fi
if [[ "$MAX_TRAIN_BATCHES" != 0 || "$MAX_VAL_BATCHES" != 0 ]]; then
  echo "[SMOKE] Limited batches: results must NOT be used as official checkpoints"
fi
case "$TRAIN_DOMAIN" in
  original) ;;
  mixed)
    : "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT for TRAIN_DOMAIN=mixed}"
    [[ -d "$GNTRANS_RGB_ROOT" ]] || { echo "Missing GN-Trans RGB root: $GNTRANS_RGB_ROOT" >&2; exit 2; }
    ;;
  *) echo "TRAIN_DOMAIN must be original or mixed, got: $TRAIN_DOMAIN" >&2; exit 2 ;;
esac
if [[ "$GNTRANS_FUSED_BACKGROUND" != 0 && "$GNTRANS_FUSED_BACKGROUND" != 1 ]]; then
  echo "GNTRANS_FUSED_BACKGROUND must be 0 or 1" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
extra=(--train_domain "$TRAIN_DOMAIN")
if [[ "$TRAIN_DOMAIN" == mixed ]]; then
  extra+=(--gntrans_rgb_root "$GNTRANS_RGB_ROOT"
          --mix_train_fraction "$MIX_TRAIN_FRACTION")
  if [[ "$GNTRANS_FUSED_BACKGROUND" == 1 ]]; then
    extra+=(--gntrans_fused_background)
  fi
fi
if [[ -n "$SHARED_DEPTH_CKPT" ]]; then
  [[ -f "$SHARED_DEPTH_CKPT" ]] || { echo "Missing shared depth checkpoint" >&2; exit 2; }
  extra+=(--shared_depth_checkpoint "$SHARED_DEPTH_CKPT")
fi
# Preflight all destinations before starting either model; never truncate evidence.
for head in "${HEAD_ARRAY[@]}"; do
  case "$head" in legacy|cdf) ;; *) echo "Invalid head=$head" >&2; exit 2;; esac
  if [[ -d "$OUTPUT_ROOT/$head" && -n "$(ls -A "$OUTPUT_ROOT/$head")" ]]; then
    echo "Destination already nonempty: $OUTPUT_ROOT/$head" >&2; exit 1
  fi
done
for head in "${HEAD_ARRAY[@]}"; do
  run="$OUTPUT_ROOT/$head"
  mkdir -p "$run"
  echo "[POINT-DEPTH] domain=$TRAIN_DOMAIN head=$head GPUS=$GPUS batch_per_gpu=$BATCH_SIZE epochs=$MAX_EPOCH fraction=$MIX_TRAIN_FRACTION"
  (set -o pipefail; torchrun --standalone --nproc_per_node="${#GPU_ARRAY[@]}" \
    train_point_depth_pose_graspnet100.py \
    --point_head "$head" \
    --dataset_root "$DATASET_ROOT" --camera realsense \
    --log_dir "$run" --num_point 20000 \
    --batch_size "$BATCH_SIZE" --max_epoch "$MAX_EPOCH" \
    --num_workers "$NUM_WORKERS" --eval_num_workers "$EVAL_NUM_WORKERS" \
    --ckpt_save_interval 5 \
    --learning_rate "$LR" --weight_decay 0 --depth_weight_decay 0 \
    --depth_prob_loss_weight 10 \
    --use_fuse_depth --graspness_mode scene --pose_depth_mode global_film \
    --val_fraction 0.1 --seed "$SEED" \
    --max_train_batches "$MAX_TRAIN_BATCHES" \
    --max_val_batches "$MAX_VAL_BATCHES" \
    "${extra[@]}" 2>&1 | tee "$run/train.log")
  if [[ "$MAX_EPOCH" -ge 20 && "$MAX_TRAIN_BATCHES" == 0 && "$MAX_VAL_BATCHES" == 0 ]]; then
    [[ -f "$run/checkpoint_epoch_019.tar" ]] || {
      echo "No required e19 checkpoint for $head" >&2; exit 1;
    }
  fi
  echo "[POINT100] finished $head -> $run"
done
