#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
CKPT=${CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/center_decoupling_diag}
CAMERA=${CAMERA:-realsense}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-none}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
SAVE_RAW_GRASPS=${SAVE_RAW_GRASPS:-0}

extra=()
if [[ "${SAVE_RAW_GRASPS}" == "1" ]]; then
  extra+=(--save_raw_grasps)
fi

for SPLIT in ${SPLITS:-test_seen test_similar test_novel}; do
  out="${OUTPUT_ROOT}/${SPLIT}"
  mkdir -p "${out}"
  python diagnose_cva_center_decoupling_imagefps.py \
    --dataset_root "${DATASET_ROOT}" \
    --checkpoint_path "${CKPT}" \
    --output_dir "${out}" \
    --split "${SPLIT}" \
    --camera "${CAMERA}" \
    --sample_interval "${SAMPLE_INTERVAL}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_workers "${NUM_WORKERS}" \
    --pose_depth_mode "${POSE_DEPTH_MODE}" \
    --fc_mode "${FC_MODE}" \
    --verify_n "${VERIFY_N}" \
    "${extra[@]}"
done
