#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
CKPT=${CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_bestofk_exact_diag}
CAMERA=${CAMERA:-realsense}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
QUERY_EVAL_NUM=${QUERY_EVAL_NUM:-128}
QUERY_EVAL_MODE=${QUERY_EVAL_MODE:-topk_uniform}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
NOOP_CHECK_SAMPLES=${NOOP_CHECK_SAMPLES:-2}
NOOP_ATOL=${NOOP_ATOL:-5e-5}
PROFILE_TIMING=${PROFILE_TIMING:-1}
SAVE_RAW_GRASPS=${SAVE_RAW_GRASPS:-0}
GPUS=${GPUS:-0,1,2}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}

case "${POSE_DEPTH_MODE}" in none|global_film|ray_gravity_film) ;; *) echo "Bad POSE_DEPTH_MODE" >&2; exit 2;; esac
case "${QUERY_EVAL_MODE}" in all|topk|uniform|topk_uniform) ;; *) echo "Bad QUERY_EVAL_MODE" >&2; exit 2;; esac
case "${FC_MODE}" in reuse_contacts|official) ;; *) echo "Bad FC_MODE" >&2; exit 2;; esac
case "${PROFILE_TIMING}" in 0|1) ;; *) echo "PROFILE_TIMING must be 0/1" >&2; exit 2;; esac
case "${SAVE_RAW_GRASPS}" in 0|1) ;; *) echo "SAVE_RAW_GRASPS must be 0/1" >&2; exit 2;; esac

if [[ "${POSE_DEPTH_MODE}" != "global_film" ]]; then
  echo "[RAY-BESTK][WARN] canonical Stage-1 checkpoint expects global_film; got ${POSE_DEPTH_MODE}" >&2
fi

export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
SPLITS_NORMALIZED="${SPLITS//,/ }"
read -r -a SPLIT_ARRAY <<< "${SPLITS_NORMALIZED}"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }
[[ ${#SPLIT_ARRAY[@]} -gt 0 ]] || { echo "No splits specified" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}"
PIDS=()
ACTIVE_SPLITS=()

wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[RAY-BESTK][ERROR] split=${ACTIVE_SPLITS[$i]} failed; inspect diagnostic.log" >&2
      failed=1
    fi
  done
  PIDS=(); ACTIVE_SPLITS=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_split() {
  local split="$1" gpu="$2" out="${OUTPUT_ROOT}/${split}"
  mkdir -p "${out}"
  local args=(
    "${SCRIPT_DIR}/diagnose_ray_bestofk_exact_action.py"
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${CKPT}"
    --output_dir "${out}"
    --split "${split}"
    --camera "${CAMERA}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --max_samples "${MAX_SAMPLES}"
    --num_workers "${NUM_WORKERS}"
    --pose_depth_mode "${POSE_DEPTH_MODE}"
    "--offsets_mm=${OFFSETS_MM}"
    --query_eval_num "${QUERY_EVAL_NUM}"
    --query_eval_mode "${QUERY_EVAL_MODE}"
    --fc_mode "${FC_MODE}"
    --verify_n "${VERIFY_N}"
    --noop_check_samples "${NOOP_CHECK_SAMPLES}"
    --noop_atol "${NOOP_ATOL}"
  )
  if [[ "${PROFILE_TIMING}" == "1" ]]; then args+=(--profile_timing); fi
  if [[ "${SAVE_RAW_GRASPS}" == "1" ]]; then args+=(--save_raw_grasps); fi

  echo "[RAY-BESTK][LAUNCH] split=${split} gpu=${gpu} offsets=${OFFSETS_MM} queries=${QUERY_EVAL_NUM}/${QUERY_EVAL_MODE}"
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
  if [[ ${slot} -ge ${#GPU_ARRAY[@]} ]]; then
    wait_wave
    slot=0
  fi
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

echo "[RAY-BESTK] completed: ${OUTPUT_ROOT}"
for raw_split in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw_split}" | xargs)"
  [[ -f "${OUTPUT_ROOT}/${split}/summary.json" ]] && echo "  ${OUTPUT_ROOT}/${split}/summary.json"
done
