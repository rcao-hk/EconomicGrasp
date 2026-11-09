#!/usr/bin/env bash
# RARE 50% Original + 50% GN-Trans: opt-in stages, never overwrite a run.
# Branch: exp/rare-ranking-mix50 (forked from exp/rare-ranking-mix10).
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh check|prepare|base|bind|smoke|mine|index|fit|infer|eval|test|compare}"
: "${RUN_ROOT:?Set an isolated RUN_ROOT; never reuse rare_mix10 directories}"
ARCH="${ARCH:-point_gvar}"
[[ "$ARCH" == point || "$ARCH" == point_gvar ]] || { echo "ARCH must be point or point_gvar" >&2; exit 2; }
ROOT="$RUN_ROOT/$ARCH"
PROTOCOL="$RUN_ROOT/frames_mix50_test10.json"
CONFIG="$ROOT/bound.json"
DEVICE="${DEVICE:-cuda:0}"
mkdir -p "$ROOT"

check_space() {
  local min_gib="$1" free_kib
  free_kib="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || { echo "Unable to check free disk space" >&2; exit 2; }
  if ((free_kib < min_gib * 1024 * 1024)); then
    echo "Only $((free_kib / 1024 / 1024)) GiB free, require >=${min_gib} GiB" >&2
    echo "Choose a new storage volume; do NOT delete another research run." >&2
    exit 2
  fi
}
require_bound() {
  [[ -f "$CONFIG" && -f "$PROTOCOL" ]] ||
    { echo "Run prepare+bind first (no automatic training)" >&2; exit 2; }
}
check) 
  python -m unittest discover -s tests -p 'test_rare_mix10.py' -v
  python -m unittest discover -s tests -p 'test_rare_mix50.py' -v
  python -m compileall -q rare_grasp tests/test_rare_mix50.py
  bash -n scripts/rare_mix50/run.sh
  exit 0
  ;;
esac
case "$STAGE" in
prepare)
  [[ ! -e "$PROTOCOL" ]] || { echo "Protocol already exists; refuse overwrite" >&2; exit 2; }
  python -m rare_grasp.pipeline manifest --train-fraction 0.5 --output "$PROTOCOL"
  ;;
base)
  : "${DATASET_ROOT:?Set GraspNet root}" "${GNTRANS_RGB_ROOT:?Set GN-Trans RGB root}"
  check_space "${MIN_FREE_GIB_BASE:-12}"
  [[ ! -e "$ROOT/base" ]] || { echo "Base dir exists; refuse overwrite" >&2; exit 2; }
  GPUS="${GPUS:-0,1,2}"; IFS=',' read -r -a devices <<< "$GPUS"
  ((${#devices[@]} >= 1)) || { echo "No GPUs" >&2; exit 2; }
  extra=(); [[ "$ARCH" != point_gvar ]] || extra+=(--enable_gvar)
  # Save checkpoint.tar every epoch (atomic file replacement by trainer), but
  # only one interval snapshot at e0 to avoid ~20 full checkpoint copies.
  CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nproc_per_node="${#devices[@]}" \
    -m rare_grasp.train_base --dataset_root "$DATASET_ROOT" --gntrans_rgb_root "$GNTRANS_RGB_ROOT" \
    --log_dir "$ROOT/base" --camera realsense --batch_size "${BATCH_SIZE:-3}" --max_epoch 20 \
    --learning_rate 0.0003 --weight_decay 0.001 --depth_weight_decay 0.001 \
    --num_workers "${NUM_WORKERS:-3}" --eval_num_workers 0 --ckpt_save_interval 20 \
    --mix_train_fraction 0.5 --mix_eval_fraction 0.1 --mix_use_fused_background \
    --multi_modal --use_cdf --extend_angle --enable_point_rayrope "${extra[@]}" \
    --enable_eval --eval_start_epoch 0 --graspness_mode scene --kview_mode A1 \
    --pose_depth_mode global_film --use_fuse_depth --seed 0 \
    --rayrope_group_chunk 2048 --gvar_action_chunk 2048 \
    --rayrope_checkpoint_chunks 0 --gvar_activation_checkpoint 0
  ;;
bind)
  [[ ! -e "$CONFIG" ]] || { echo "Bound config exists; refuse overwrite" >&2; exit 2; }
  ck="${BASE_CHECKPOINT:-$ROOT/base/checkpoint.tar}"
  flags=()
  if [[ "${HISTORICAL_E15:-0}" == 1 ]]; then flags+=(--historical-e15); fi
  python -m rare_grasp.pipeline bind --template "configs/rare_mix50/$ARCH.json" \
    --checkpoint "$ck" --output "$CONFIG" "${flags[@]}"
  ;;
smoke)
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}"
  require_bound
  for domain in original gntrans; do
    out="$ROOT/smoke/$domain"
    [[ ! -e "$out" ]] || { echo "$out exists; do not overwrite smoke" >&2; exit 2; }
    python -m rare_grasp.pipeline mine --config "$CONFIG" --protocol "$PROTOCOL" \
      --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
      --partition train --domain "$domain" --families 2 --limit 1 \
      --output "$out" --device "$DEVICE" --seed 0
  done
  ;;
mine)
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}"
  require_bound
  check_space "${MIN_FREE_GIB_MINE:-6}"
  for part in train validation; do
    python -m rare_grasp.pipeline mine --config "$CONFIG" --protocol "$PROTOCOL" \
      --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
      --partition "$part" --output "$ROOT/cache/$part" --families 32 --seed 0 \
      --num-shards "${NUM_SHARDS:-1}" --shard-index "${SHARD_INDEX:-0}" \
      --device "$DEVICE"
  done
  ;;
index)
  require_bound
  for part in train validation; do
    out="$ROOT/cache/${part}_index.json"
    [[ ! -e "$out" ]] || { echo "$out exists; refuse overwrite" >&2; exit 2; }
    python -m rare_grasp.pipeline index --protocol "$PROTOCOL" --partition "$part" \
      --cache "$ROOT/cache/$part" --output "$out"
  done
  ;;
fit)
  require_bound
  for objective in point rare; do
    out="$ROOT/heads/$objective"
    [[ ! -e "$out" ]] || { echo "$out exists; refuse overwrite" >&2; exit 2; }
    python -m rare_grasp.pipeline fit --train-index "$ROOT/cache/train_index.json" \
      --val-index "$ROOT/cache/validation_index.json" --output "$out" \
      --objective "$objective" --pair-scope image --point-weight 10 --pair-weight 0.5 \
      --max-pairs 2048 --epochs 20 --batch-frames 4 --hidden 64 \
      --lr 0.0001 --weight-decay 0.0001 --seed 0 --device "$DEVICE"
  done
  ;;
infer|eval|test)
  : "${DATASET_ROOT:?}"
  require_bound
  DOMAIN="${DOMAIN:-original}"
  [[ "$DOMAIN" == original || "$DOMAIN" == gntrans ]] || { echo "DOMAIN must be original or gntrans" >&2; exit 2; }
  [[ "$DOMAIN" != gntrans || -n "${GNTRANS_RGB_ROOT:-}" ]] ||
    { echo "GNTRANS_RGB_ROOT is required for GNTrans evaluation" >&2; exit 2; }
  for split in ${SPLITS:-test_seen test_similar test_novel}; do
    [[ "$split" == test_seen || "$split" == test_similar || "$split" == test_novel ]] ||
      { echo "Unsupported split: $split" >&2; exit 2; }
    for method in ${METHODS:-native point rare}; do
      [[ "$method" == native || "$method" == point || "$method" == rare ]] ||
        { echo "Unsupported method: $method" >&2; exit 2; }
      out="$ROOT/test/$DOMAIN/$split/$method"
      if [[ "$STAGE" != eval ]]; then
        extra=();policies=native
        if [[ "$method" != native ]]; then
          extra+=(--ranker "$ROOT/heads/$method/last.pt")
          policies=both,depth_only,rank_only
        fi
        python -m rare_grasp.pipeline infer --config "$CONFIG" --protocol "$PROTOCOL" \
          --dataset-root "$DATASET_ROOT" --gntrans-root "${GNTRANS_RGB_ROOT:-}" \
          --domain "$DOMAIN" --split "$split" --output "$out" --policies "$policies" \
          --device "$DEVICE" --seed 0 "${extra[@]}"
      fi
      if [[ "$STAGE" != infer ]]; then
        python -m rare_grasp.pipeline eval --dataset-root "$DATASET_ROOT" \
          --inference "$out" --workers "${EVAL_WORKERS:-16}"
      fi
    done
  done
  ;;
compare)
  DOMAIN="${DOMAIN:-original}"
  POLICY="${POLICY:-both}"
  [[ "$POLICY" == both || "$POLICY" == depth_only || "$POLICY" == rank_only ]] ||
    { echo "Unexpected policy $POLICY" >&2; exit 2; }
  out="$ROOT/rare_minus_mse_${DOMAIN}_${POLICY}.json"
  [[ ! -e "$out" ]] || { echo "Comparison exists; refuse overwrite" >&2; exit 2; }
  args=()
  for split in test_seen test_similar test_novel; do
    args+=(--left "$ROOT/test/$DOMAIN/$split/point/$POLICY/ap_${split}_realsense.npy")
    args+=(--right "$ROOT/test/$DOMAIN/$split/rare/$POLICY/ap_${split}_realsense.npy")
  done
  python -m rare_grasp.pipeline compare "${args[@]}" \
    --output "$out" --seed 0 --bootstrap 5000
  ;;
*)
  echo "Unknown stage $STAGE" >&2
  exit 2
  ;;
esac
