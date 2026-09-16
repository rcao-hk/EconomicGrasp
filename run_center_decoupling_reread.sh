#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
CKPT=${CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/center_decoupling_reread_corrected}
CAMERA=${CAMERA:-realsense}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
SAVE_RAW_GRASPS=${SAVE_RAW_GRASPS:-0}
QUERY_EVAL_NUM=${QUERY_EVAL_NUM:-128}
QUERY_EVAL_MODE=${QUERY_EVAL_MODE:-topk_uniform}
EVAL_VALID_ONLY=${EVAL_VALID_ONLY:-1}
PROFILE_TIMING=${PROFILE_TIMING:-1}
NOOP_CHECK_SAMPLES=${NOOP_CHECK_SAMPLES:-2}
NOOP_ATOL=${NOOP_ATOL:-5e-5}
GPUS=${GPUS:-0,1,2}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}

# Avoid the NumPy huge-page madvise behavior that caused severe memory-management
# overhead on some of the evaluation servers. Override explicitly if desired.
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

case "${POSE_DEPTH_MODE}" in
  none|global_film|ray_gravity_film) ;;
  *) echo "POSE_DEPTH_MODE must be none/global_film/ray_gravity_film" >&2; exit 2 ;;
esac
case "${FC_MODE}" in
  reuse_contacts|official) ;;
  *) echo "FC_MODE must be reuse_contacts or official" >&2; exit 2 ;;
esac
case "${QUERY_EVAL_MODE}" in
  all|topk|uniform|topk_uniform) ;;
  *) echo "QUERY_EVAL_MODE must be all/topk/uniform/topk_uniform" >&2; exit 2 ;;
esac
case "${EVAL_VALID_ONLY}" in 0|1) ;; *) echo "EVAL_VALID_ONLY must be 0 or 1" >&2; exit 2;; esac
case "${PROFILE_TIMING}" in 0|1) ;; *) echo "PROFILE_TIMING must be 0 or 1" >&2; exit 2;; esac
case "${SAVE_RAW_GRASPS}" in 0|1) ;; *) echo "SAVE_RAW_GRASPS must be 0 or 1" >&2; exit 2;; esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
SPLITS_NORMALIZED="${SPLITS//,/ }"
read -r -a SPLIT_ARRAY <<< "${SPLITS_NORMALIZED}"
[[ "${#GPU_ARRAY[@]}" -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }
[[ "${#SPLIT_ARRAY[@]}" -gt 0 ]] || { echo "No splits specified" >&2; exit 2; }

extra=()
[[ "${SAVE_RAW_GRASPS}" == "1" ]] && extra+=(--save_raw_grasps)
[[ "${EVAL_VALID_ONLY}" == "1" ]] && extra+=(--eval_valid_only)
[[ "${PROFILE_TIMING}" == "1" ]] && extra+=(--profile_timing)

mkdir -p "${OUTPUT_ROOT}"
PIDS=()
ACTIVE_SPLITS=()

wait_active_workers() {
  local failed=0
  local i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REREAD-DIAG][ERROR] split=${ACTIVE_SPLITS[$i]} failed; inspect diagnostic.log" >&2
      failed=1
    fi
  done
  PIDS=()
  ACTIVE_SPLITS=()
  [[ "${failed}" == "0" ]] || exit 1
}

launch_split() {
  local split="$1"
  local gpu="$2"
  local out="${OUTPUT_ROOT}/${split}"
  mkdir -p "${out}"
  args=(
    "${SCRIPT_DIR}/diagnose_cva_center_decoupling_reread.py"
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
    --noop_check_samples "${NOOP_CHECK_SAMPLES}"
    --noop_atol "${NOOP_ATOL}"
  )
  args+=("${extra[@]}")

  echo "[REREAD-DIAG][LAUNCH] split=${split} gpu=${gpu} pose_depth=${POSE_DEPTH_MODE} query_mode=${QUERY_EVAL_MODE} query_num=${QUERY_EVAL_NUM} noop=${NOOP_CHECK_SAMPLES}"
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
  if [[ "${slot}" -ge "${#GPU_ARRAY[@]}" ]]; then
    wait_active_workers
    slot=0
  fi
done
if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait_active_workers
fi

echo "[REREAD-DIAG] completed: ${OUTPUT_ROOT}"
for raw_split in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw_split}" | xargs)"
  [[ -n "${split}" ]] || continue
  [[ -f "${OUTPUT_ROOT}/${split}/summary.json" ]] && \
    echo "[REREAD-DIAG] summary: ${OUTPUT_ROOT}/${split}/summary.json"
done
