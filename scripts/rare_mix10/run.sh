#!/usr/bin/env bash
# Explicit stages, no implicit full-run or checkpoint overwrite.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh prepare|base|bind|mine|index|fit|test}"
: "${RUN_ROOT:?Set an isolated RUN_ROOT}"
ARCH="${ARCH:-point_gvar}"
[[ "$ARCH" == point || "$ARCH" == point_gvar ]] || { echo 'ARCH must be point or point_gvar' >&2; exit 2; }
ROOT="$RUN_ROOT/$ARCH"
mkdir -p "$ROOT"
PROTOCOL="$RUN_ROOT/frames_mix10.json"
CONFIG="$ROOT/bound.json"
DEVICE="${DEVICE:-cuda:0}"
case "$STAGE" in
prepare)
  [[ ! -e "$PROTOCOL" ]] || { echo 'Protocol exists: keep it; no overwrite' >&2; exit 2; }
  python -m rare_grasp.pipeline manifest --output "$PROTOCOL"
  ;;
base)
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}"
  [[ ! -e "$ROOT/base" ]] || { echo 'Base dir exists: refuse overwrite/resume' >&2; exit 2; }
  GPUS="${GPUS:-0,1,2}"; IFS=',' read -r -a devices <<< "$GPUS"
  extra=(); [[ "$ARCH" != point_gvar ]] || extra+=(--enable_gvar)
  CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nproc_per_node="${#devices[@]}" \
    -m rare_grasp.train_base --dataset_root "$DATASET_ROOT" --gntrans_rgb_root "$GNTRANS_RGB_ROOT" \
    --log_dir "$ROOT/base" --camera realsense --batch_size "${BATCH_SIZE:-3}" --max_epoch 20 \
    --learning_rate 0.0003 --weight_decay 0.001 --depth_weight_decay 0.001 \
    --num_workers "${NUM_WORKERS:-3}" --eval_num_workers 0 --ckpt_save_interval 1 \
    --mix_train_fraction 0.1 --mix_eval_fraction 0.1 --mix_use_fused_background \
    --multi_modal --use_cdf --extend_angle --enable_point_rayrope "${extra[@]}" \
    --enable_eval --eval_start_epoch 0 --graspness_mode scene --kview_mode A1 \
    --pose_depth_mode global_film --use_fuse_depth --seed 0 \
    --rayrope_group_chunk 2048 --gvar_action_chunk 2048 \
    --rayrope_checkpoint_chunks 0 --gvar_activation_checkpoint 0
  ;;
bind)
  [[ ! -e "$CONFIG" ]] || { echo 'Bound config exists; do not replace silently' >&2; exit 2; }
  python -m rare_grasp.pipeline bind --template "configs/rare_mix10/$ARCH.json" \
    --checkpoint "${BASE_CHECKPOINT:-$ROOT/base/checkpoint_19.tar}" --output "$CONFIG"
  ;;
mine)
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}"
  for part in train validation; do
    python -m rare_grasp.pipeline mine --config "$CONFIG" --protocol "$PROTOCOL" \
      --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
      --partition "$part" --output "$ROOT/cache/$part" --families 32 --seed 0 \
      --device "$DEVICE" --num-shards "${NUM_SHARDS:-1}" --shard-index "${SHARD_INDEX:-0}"
  done
  ;;
index)
  for part in train validation; do
    python -m rare_grasp.pipeline index --protocol "$PROTOCOL" --partition "$part" \
      --cache "$ROOT/cache/$part" --output "$ROOT/cache/${part}_index.json"
  done
  ;;
fit)
  for objective in point rare; do
    python -m rare_grasp.pipeline fit --train-index "$ROOT/cache/train_index.json" \
      --val-index "$ROOT/cache/validation_index.json" --output "$ROOT/heads/$objective" \
      --objective "$objective" --pair-scope image --point-weight 10 --pair-weight 0.5 \
      --max-pairs 2048 --epochs 20 --batch-frames 4 --hidden 64 \
      --lr 0.0001 --weight-decay 0.0001 --seed 0 --device "$DEVICE"
  done
  ;;
test)
  : "${DATASET_ROOT:?}"
  DOMAIN="${DOMAIN:-original}" # optionally gntrans; original-sensor collision for both
  for split in test_seen test_similar test_novel; do
    for method in native point rare; do
      extra=();policies=native
      if [[ "$method" != native ]]; then
        extra+=(--ranker "$ROOT/heads/$method/last.pt")
        policies=both,depth_only,rank_only
      fi
      out="$ROOT/test/$DOMAIN/$split/$method"
      python -m rare_grasp.pipeline infer --config "$CONFIG" --protocol "$PROTOCOL" \
        --dataset-root "$DATASET_ROOT" --gntrans-root "${GNTRANS_RGB_ROOT:-}" \
        --domain "$DOMAIN" --split "$split" --output "$out" --policies "$policies" \
        --device "$DEVICE" --seed 0 "${extra[@]}"
      python -m rare_grasp.pipeline eval --dataset-root "$DATASET_ROOT" --inference "$out" \
        --workers "${EVAL_WORKERS:-16}"
    done
  done
  ;;
*) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
