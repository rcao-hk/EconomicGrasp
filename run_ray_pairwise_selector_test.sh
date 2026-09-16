#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
SELECTOR_CKPT=${SELECTOR_CKPT:-${WORK_ROOT}/train/checkpoint_best.tar}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/test}

GPUS=${GPUS:-0,1,2}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
QUERY_EVAL_NUM=${QUERY_EVAL_NUM:-128}
QUERY_EVAL_MODE=${QUERY_EVAL_MODE:-topk_uniform}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
NOOP_CHECK_SAMPLES=${NOOP_CHECK_SAMPLES:-2}
NOOP_ATOL=${NOOP_ATOL:-5e-5}
PROFILE_TIMING=${PROFILE_TIMING:-1}
SAVE_CANDIDATE_ROWS=${SAVE_CANDIDATE_ROWS:-0}
SELECTOR_THRESHOLD=${SELECTOR_THRESHOLD:-}

export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -f "${SELECTOR_CKPT}" ]] || { echo "Selector checkpoint not found: ${SELECTOR_CKPT}" >&2; exit 2; }
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
SPLITS_NORMALIZED="${SPLITS//,/ }"
read -r -a SPLIT_ARRAY <<< "${SPLITS_NORMALIZED}"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No GPUS specified" >&2; exit 2; }
[[ ${#SPLIT_ARRAY[@]} -gt 0 ]] || { echo "No SPLITS specified" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}"
PIDS=()
ACTIVE=()

wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[PAIR-TEST][ERROR] split=${ACTIVE[$i]} failed; inspect diagnostic.log" >&2
      failed=1
    fi
  done
  PIDS=(); ACTIVE=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_split() {
  local split="$1" gpu="$2" out="${OUTPUT_ROOT}/${split}"
  mkdir -p "${out}"
  local args=(
    "${SCRIPT_DIR}/test_ray_pairwise_selector.py"
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${STAGE1_CKPT}"
    --selector_checkpoint "${SELECTOR_CKPT}"
    --output_dir "${out}"
    --split "${split}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --max_samples "${MAX_SAMPLES}"
    --num_workers "${NUM_WORKERS}"
    --pose_depth_mode "${POSE_DEPTH_MODE}"
    --offsets_mm "${OFFSETS_MM}"
    --query_eval_num "${QUERY_EVAL_NUM}"
    --query_eval_mode "${QUERY_EVAL_MODE}"
    --fc_mode "${FC_MODE}"
    --verify_n "${VERIFY_N}"
    --noop_check_samples "${NOOP_CHECK_SAMPLES}"
    --noop_atol "${NOOP_ATOL}"
  )
  [[ "${PROFILE_TIMING}" == "1" ]] && args+=(--profile_timing)
  [[ "${SAVE_CANDIDATE_ROWS}" == "1" ]] && args+=(--save_candidate_rows)
  [[ -n "${SELECTOR_THRESHOLD}" ]] && args+=(--selector_threshold "${SELECTOR_THRESHOLD}")
  echo "[PAIR-TEST][LAUNCH] split=${split} gpu=${gpu} selector=${SELECTOR_CKPT}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" >"${out}/diagnostic.log" 2>&1 &
  PIDS+=("$!"); ACTIVE+=("${split}")
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

echo "[PAIR-TEST] completed: ${OUTPUT_ROOT}"
for raw_split in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw_split}" | xargs)"
  [[ -f "${OUTPUT_ROOT}/${split}/summary.json" ]] && echo "  ${OUTPUT_ROOT}/${split}/summary.json"
done
