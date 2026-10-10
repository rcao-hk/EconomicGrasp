#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT}"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1,2}"
VARIANT="${VARIANT:-volume}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/log/gvar_${VARIANT}_10pct}"
BATCH_SIZE="${BATCH_SIZE:-3}"
MAX_EPOCH="${MAX_EPOCH:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-16}"
ACTION_CHUNK="${ACTION_CHUNK:-512}"
READER_DIM="${READER_DIM:-64}"
READER_HEADS="${READER_HEADS:-4}"
ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-1}"
MAX_BATCHES="${MAX_BATCHES:-0}"
SEED="${SEED:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"
IFS=',' read -r -a GPUs <<< "${GPUS}"
[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPUS=${GPUS}" >&2; exit 2; }
case "${VARIANT}" in baseline|slot|volume_fixed|volume|volume_rel|volume_fixed_rel) ;; *) echo "Invalid VARIANT=${VARIANT}" >&2; exit 2;; esac
extra=(--use_fuse_depth --mix_use_fused_background)
if [[ -n "${RESUME_CKPT}" ]]; then
  extra+=(--resume --checkpoint_path "${RESUME_CKPT}")
fi
cmd=("${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${#GPUs[@]}" train_gvar.py
  --dataset_root "${DATASET_ROOT}" --gntrans_rgb_root "${GNTRANS_RGB_ROOT}" --log_dir "${OUTPUT_ROOT}"
  --camera realsense --batch_size "${BATCH_SIZE}" --max_epoch "${MAX_EPOCH}"
  --learning_rate "${LR:-0.0001}" --weight_decay 0 --depth_weight_decay 0
  --num_workers "${NUM_WORKERS}" --eval_num_workers "${EVAL_NUM_WORKERS}"
  --ckpt_save_interval 5 --mix_train_fraction 0.1 --mix_eval_fraction 0.1
  --multi_modal --use_cdf --extend_angle --enable_eval --eval_start_epoch 0
  --graspness_mode scene --kview_mode A1 --pose_depth_mode global_film --seed "${SEED}"
  --gvar_variant "${VARIANT}" --gvar_reader_dim "${READER_DIM}" --gvar_reader_heads "${READER_HEADS}"
  --gvar_action_chunk "${ACTION_CHUNK}" --gvar_activation_checkpoint "${ACTIVATION_CHECKPOINT}"
  --gvar_max_batches "${MAX_BATCHES}" "${extra[@]}" "$@")
printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPUS}"; printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
if [[ -z "${RESUME_CKPT}" && -d "${OUTPUT_ROOT}" && -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing nonempty OUTPUT_ROOT; choose a new directory or RESUME_CKPT" >&2; exit 2
fi
if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then echo "Missing resume checkpoint" >&2; exit 2; fi
mkdir -p "$(dirname "${OUTPUT_ROOT}")"
export CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
"${cmd[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}.console.log"
