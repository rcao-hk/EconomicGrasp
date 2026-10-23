#!/usr/bin/env bash
set -euo pipefail
# Experiment: 10% original GraspNet + paired 10% GN-Trans + all 232 filtered GC6D scenes.
# Entrypoint from the repository root. No RGB-D network input is enabled.

: "${DATASET_ROOT:?Set DATASET_ROOT to GraspNet-1Billion root}"
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT to GN-Trans RGB root}"
: "${GC6D_ROOT:?Set GC6D_ROOT to GraspClutter6D root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPUS="${GPUS:-0,1,2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/log/cva_cdf_gntrans_gc6d_mix_10pct}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.1}"
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
GC6D_CAMERA="${GC6D_CAMERA:-realsense-d435}"
GC6D_MAX_ANN="${GC6D_MAX_ANN:-13}"
GC6D_SCENE_IDS_FILE="${GC6D_SCENE_IDS_FILE:-${GC6D_ROOT}/split_info/no_gn_test_overlap/grasp_train_scene_ids_no_graspnet_test_overlap.json}"
GC6D_LABEL_FOLDER="${GC6D_LABEL_FOLDER:-economic_grasp_label_300views_extend_angle_cdf_depth}"
GC6D_EXPECTED_SCENES="${GC6D_EXPECTED_SCENES:-232}"
GC6D_EXPECTED_FRAMES="${GC6D_EXPECTED_FRAMES:-3016}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EPOCH="${MAX_EPOCH:-20}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
DEPTH_WEIGHT_DECAY="${DEPTH_WEIGHT_DECAY:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-5}"
POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
GC6D_WORKSPACE_POSE_MODE="${GC6D_WORKSPACE_POSE_MODE:-json}"
USE_FUSE_DEPTH="${USE_FUSE_DEPTH:-1}"
TRAIN_VALIDATE="${TRAIN_VALIDATE:-0}"
CHECK_DATA="${CHECK_DATA:-1}"
CHECK_ALL_CACHES="${CHECK_ALL_CACHES:-0}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-0}"

for flag in USE_FUSE_DEPTH TRAIN_VALIDATE CHECK_DATA CHECK_ALL_CACHES DRY_RUN; do
  [[ "${!flag}" =~ ^[01]$ ]] || { echo "Invalid $flag=${!flag}, expected 0/1" >&2; exit 2; }
done
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
[[ "${NPROC}" -ge 1 ]] || { echo "GPUS must specify at least one device" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

if [[ "${CHECK_DATA}" == "1" ]]; then
  CHECK_ARGS=()
  if [[ "${CHECK_ALL_CACHES}" == "1" ]]; then CHECK_ARGS+=(--check_all_caches); fi
  python check_gc6d_cdf_mix_data.py \
    --gc6d_root "${GC6D_ROOT}" \
    --gc6d_scene_ids_file "${GC6D_SCENE_IDS_FILE}" \
    --gc6d_camera "${GC6D_CAMERA}" \
    --gc6d_label_folder "${GC6D_LABEL_FOLDER}" \
    --max_ann "${GC6D_MAX_ANN}" \
    --expected_scenes "${GC6D_EXPECTED_SCENES}" \
    --expected_frames "${GC6D_EXPECTED_FRAMES}" \
    --workspace_pose_mode "${GC6D_WORKSPACE_POSE_MODE}" \
    --load_one \
    --output_json "${OUTPUT_ROOT}/gc6d_preflight.json" \
    "${CHECK_ARGS[@]}"
  # The original paired-domain checker validates RGB pair correspondence.
  python check_gntrans_mix_data.py \
    --dataset_root "${DATASET_ROOT}" \
    --gntrans_rgb_root "${GNTRANS_RGB_ROOT}" \
    --fraction "${TRAIN_FRACTION}" \
    --check_all_selected \
    --load_one \
    --output "${OUTPUT_ROOT}/gntrans_pair_preflight.json"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[TRIPLE-MIX] Dry-run preflight complete; training skipped."
  exit 0
fi

EXTRA=()
if [[ "${USE_FUSE_DEPTH}" == "1" ]]; then
  EXTRA+=(--use_fuse_depth --mix_use_fused_background)
fi
if [[ "${TRAIN_VALIDATE}" == "1" ]]; then
  EXTRA+=(--enable_eval --eval_start_epoch 0)
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
torchrun --standalone --nproc_per_node="${NPROC}" train_cva_gntrans_gc6d_mix_ddp.py \
  --dataset_root "${DATASET_ROOT}" \
  --gntrans_rgb_root "${GNTRANS_RGB_ROOT}" \
  --gc6d_root "${GC6D_ROOT}" \
  --gc6d_scene_ids_file "${GC6D_SCENE_IDS_FILE}" \
  --gc6d_camera "${GC6D_CAMERA}" \
  --gc6d_max_ann "${GC6D_MAX_ANN}" \
  --gc6d_cdf_label_folder "${GC6D_LABEL_FOLDER}" \
  --gc6d_expected_scenes "${GC6D_EXPECTED_SCENES}" \
  --gc6d_expected_frames "${GC6D_EXPECTED_FRAMES}" \
  --gc6d_workspace_pose_mode "${GC6D_WORKSPACE_POSE_MODE}" \
  --log_dir "${OUTPUT_ROOT}" \
  --camera realsense \
  --batch_size "${BATCH_SIZE}" \
  --max_epoch "${MAX_EPOCH}" \
  --learning_rate "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --depth_weight_decay "${DEPTH_WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_num_workers "${EVAL_NUM_WORKERS}" \
  --ckpt_save_interval "${CKPT_SAVE_INTERVAL}" \
  --mix_train_fraction "${TRAIN_FRACTION}" \
  --mix_eval_fraction "${EVAL_FRACTION}" \
  --multi_modal --use_cdf --extend_angle \
  --graspness_mode scene --kview_mode A1 \
  --pose_depth_mode "${POSE_DEPTH_MODE}" \
  --seed "${SEED}" \
  "${EXTRA[@]}"
