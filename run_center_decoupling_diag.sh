#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
CKPT=${CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/center_decoupling_diag}
CAMERA=${CAMERA:-realsense}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_WORKERS=${NUM_WORKERS:-4}

# The canonical economicgrasp_dpt_cva_cdf_distill_stage1 checkpoint was trained
# with pose-conditioned metric depth using global FiLM.  Keep the diagnostic on
# the same depth architecture by default; overriding this is only appropriate for
# a checkpoint trained with a different pose-depth mode.
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}

FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
SAVE_RAW_GRASPS=${SAVE_RAW_GRASPS:-0}

# Fast diagnostic controls.  The defaults preserve the main causal question
# while avoiding unnecessary exact-action evaluation of all 1024 image-FPS
# queries in every frame.  QUERY_EVAL_NUM=0 restores exhaustive evaluation.
QUERY_EVAL_NUM=${QUERY_EVAL_NUM:-128}
QUERY_EVAL_MODE=${QUERY_EVAL_MODE:-topk_uniform}
EVAL_VALID_ONLY=${EVAL_VALID_ONLY:-1}
PROFILE_TIMING=${PROFILE_TIMING:-1}

# One process per split.  Splits are assigned to GPUs round-robin, with at most
# one active diagnostic process per listed GPU.  This matches the repository's
# multi-GPU inference launchers and avoids sharing one model process across GPUs.
GPUS=${GPUS:-0,1,2}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}

case "${POSE_DEPTH_MODE}" in
  none|global_film|ray_gravity_film) ;;
  *) echo "POSE_DEPTH_MODE must be one of: none, global_film, ray_gravity_film" >&2; exit 2 ;;
esac
case "${FC_MODE}" in
  reuse_contacts|official) ;;
  *) echo "FC_MODE must be reuse_contacts or official" >&2; exit 2 ;;
esac
case "${SAVE_RAW_GRASPS}" in
  0|1) ;;
  *) echo "SAVE_RAW_GRASPS must be 0 or 1" >&2; exit 2 ;;
esac
case "${EVAL_VALID_ONLY}" in
  0|1) ;;
  *) echo "EVAL_VALID_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
case "${PROFILE_TIMING}" in
  0|1) ;;
  *) echo "PROFILE_TIMING must be 0 or 1" >&2; exit 2 ;;
esac
case "${QUERY_EVAL_MODE}" in
  all|topk|uniform|topk_uniform) ;;
  *) echo "QUERY_EVAL_MODE must be one of: all, topk, uniform, topk_uniform" >&2; exit 2 ;;
esac

if [[ "${POSE_DEPTH_MODE}" != "global_film" ]]; then
  echo "[CENTER-DIAG][WARN] POSE_DEPTH_MODE=${POSE_DEPTH_MODE}; the default Stage-1 checkpoint expects global_film." >&2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "No GPUs specified in GPUS=${GPUS}" >&2
  exit 2
fi

# Accept either comma-separated or whitespace-separated SPLITS.
SPLITS_NORMALIZED="${SPLITS//,/ }"
read -r -a SPLIT_ARRAY <<< "${SPLITS_NORMALIZED}"
if [[ "${#SPLIT_ARRAY[@]}" -eq 0 ]]; then
  echo "No splits specified in SPLITS=${SPLITS}" >&2
  exit 2
fi

extra=()
if [[ "${SAVE_RAW_GRASPS}" == "1" ]]; then
  extra+=(--save_raw_grasps)
fi
if [[ "${EVAL_VALID_ONLY}" == "1" ]]; then
  extra+=(--eval_valid_only)
fi
if [[ "${PROFILE_TIMING}" == "1" ]]; then
  extra+=(--profile_timing)
fi

mkdir -p "${OUTPUT_ROOT}"

PIDS=()
ACTIVE_SPLITS=()

wait_active_workers() {
  local failed=0
  local i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[CENTER-DIAG][ERROR] split=${ACTIVE_SPLITS[$i]} failed; inspect its diagnostic.log" >&2
      failed=1
    fi
  done
  PIDS=()
  ACTIVE_SPLITS=()
  if [[ "${failed}" != "0" ]]; then
    exit 1
  fi
}

launch_split() {
  local split="$1"
  local gpu="$2"
  local out="${OUTPUT_ROOT}/${split}"
  mkdir -p "${out}"

  args=(
    "${SCRIPT_DIR}/diagnose_cva_center_decoupling_imagefps.py"
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${CKPT}"
    --output_dir "${out}"
    --split "${split}"
    --camera "${CAMERA}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --max_samples "${MAX_SAMPLES}"
    --num_workers "${NUM_WORKERS}"
    --pose_depth_mode "${POSE_DEPTH_MODE}"
    --fc_mode "${FC_MODE}"
    --verify_n "${VERIFY_N}"
    --query_eval_num "${QUERY_EVAL_NUM}"
    --query_eval_mode "${QUERY_EVAL_MODE}"
  )
  args+=("${extra[@]}")

  echo "[CENTER-DIAG][LAUNCH] split=${split} gpu=${gpu} pose_depth=${POSE_DEPTH_MODE} sample=${SAMPLE_INTERVAL} query_mode=${QUERY_EVAL_MODE} query_num=${QUERY_EVAL_NUM} valid_only=${EVAL_VALID_ONLY}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="${gpu}" python "${args[@]}" >"${out}/diagnostic.log" 2>&1 &
  PIDS+=("$!")
  ACTIVE_SPLITS+=("${split}")
}

slot=0
for raw_split in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw_split}" | xargs)"
  [[ -n "${split}" ]] || continue

  launch_split "${split}" "${GPU_ARRAY[$slot]}"
  slot=$((slot + 1))

  # Do not oversubscribe a GPU.  If there are more splits than GPUs, finish the
  # current wave before launching the next one.
  if [[ "${slot}" -ge "${#GPU_ARRAY[@]}" ]]; then
    wait_active_workers
    slot=0
  fi
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait_active_workers
fi

echo "[CENTER-DIAG] completed: ${OUTPUT_ROOT}"
for raw_split in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw_split}" | xargs)"
  [[ -n "${split}" ]] || continue
  if [[ -f "${OUTPUT_ROOT}/${split}/summary.json" ]]; then
    echo "[CENTER-DIAG] summary: ${OUTPUT_ROOT}/${split}/summary.json"
  fi
done
