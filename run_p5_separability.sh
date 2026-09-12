#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${CKPT:?Set CKPT to a trained P5-v1.1 checkpoint}"

GPUS="${GPUS:-0,1,2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/p5_separability}"
TRAIN_SAMPLE_INTERVAL="${TRAIN_SAMPLE_INTERVAL:-0.1}"
EVAL_SAMPLE_INTERVAL="${EVAL_SAMPLE_INTERVAL:-0.1}"
QUERY_SAMPLE_PER_FRAME="${QUERY_SAMPLE_PER_FRAME:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DIAG_SAFE_RADIUS_M="${DIAG_SAFE_RADIUS_M:-0.005}"
DIAG_CORRUPT_SEED="${DIAG_CORRUPT_SEED:-55117}"
DIAG_MAX_BATCHES="${DIAG_MAX_BATCHES:-0}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"
PROBE_EPOCHS="${PROBE_EPOCHS:-15}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-8192}"
MAX_TRAIN_QUERIES="${MAX_TRAIN_QUERIES:-250000}"
# Set to 1 to delete intermediate rank*_chunk*.npz extraction shards after
# successful analysis (or after extraction-only smoke). summary.json/extract.log
# and all files under analysis/ are preserved.
REMOVE_DUMP_FILES="${REMOVE_DUMP_FILES:-0}"

case "${REMOVE_DUMP_FILES}" in
  0|1) ;;
  *)
    echo "REMOVE_DUMP_FILES must be 0 or 1, got: ${REMOVE_DUMP_FILES}" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ ${#GPU_ARRAY[@]} -lt 1 ]]; then
  echo "Provide at least one GPU in GPUS" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

# Four frozen extraction conditions.  Run in waves so no GPU receives two P5
# models simultaneously when fewer than four GPUs are provided.
TASK_NAMES=(train_native train_corrupt seen_native seen_corrupt)
TASK_SPLITS=(train train test_seen test_seen)
TASK_CONDS=(native corrupt native corrupt)
TASK_FRACS=("${TRAIN_SAMPLE_INTERVAL}" "${TRAIN_SAMPLE_INTERVAL}" "${EVAL_SAMPLE_INTERVAL}" "${EVAL_SAMPLE_INTERVAL}")

cleanup_dump_files() {
  if [[ "${REMOVE_DUMP_FILES}" != "1" ]]; then
    return 0
  fi
  echo "[P5-DIAG] removing intermediate dump shards (rank*_chunk*.npz)"
  local name dir
  for name in "${TASK_NAMES[@]}"; do
    dir="${OUTPUT_ROOT}/${name}"
    if [[ -d "${dir}" ]]; then
      find "${dir}" -maxdepth 1 -type f -name 'rank*_chunk*.npz' -delete
    fi
  done
  echo "[P5-DIAG] dump cleanup complete; summaries/logs and analysis outputs preserved"
}

run_task() {
  local idx="$1"
  local gpu="$2"
  local name="${TASK_NAMES[$idx]}"
  local split="${TASK_SPLITS[$idx]}"
  local cond="${TASK_CONDS[$idx]}"
  local frac="${TASK_FRACS[$idx]}"
  local out="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out}"
  echo "[P5-DIAG] ${name}: ${split}/${cond} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python diagnose_p5_separability.py \
    --dataset_root "${DATASET_ROOT}" \
    --checkpoint_path "${CKPT}" \
    --batch_size 1 \
    --num_workers "${NUM_WORKERS}" \
    --sample_interval "${frac}" \
    --graspness_mode scene \
    --kview_mode A1 \
    --diag_split "${split}" \
    --diag_condition "${cond}" \
    --diag_output_dir "${out}" \
    --diag_query_sample_per_frame "${QUERY_SAMPLE_PER_FRAME}" \
    --diag_safe_radius_m "${DIAG_SAFE_RADIUS_M}" \
    --diag_corrupt_seed "${DIAG_CORRUPT_SEED}" \
    --diag_max_batches "${DIAG_MAX_BATCHES}" \
    > "${out}/extract.log" 2>&1
}

ntasks=${#TASK_NAMES[@]}
ngpu=${#GPU_ARRAY[@]}
start=0
while [[ ${start} -lt ${ntasks} ]]; do
  pids=()
  for ((j=0; j<ngpu && start+j<ntasks; j++)); do
    idx=$((start+j))
    run_task "${idx}" "${GPU_ARRAY[$j]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then status=1; fi
  done
  if [[ ${status} -ne 0 ]]; then
    echo "P5 diagnostic extraction failed; inspect */extract.log" >&2
    exit ${status}
  fi
  start=$((start+ngpu))
done

if [[ "${DIAG_MAX_BATCHES}" != "0" ]]; then
  echo "[P5-DIAG] extraction smoke complete; skip probe analysis because DIAG_MAX_BATCHES != 0"
  cleanup_dump_files
  exit 0
fi

# Linear probes are tiny; use the first listed GPU unless CPU was requested.
if [[ "${PROBE_DEVICE}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}"
else
  export CUDA_VISIBLE_DEVICES=""
fi
python analyze_p5_separability.py \
  --train_native "${OUTPUT_ROOT}/train_native" \
  --train_corrupt "${OUTPUT_ROOT}/train_corrupt" \
  --eval_native "${OUTPUT_ROOT}/seen_native" \
  --eval_corrupt "${OUTPUT_ROOT}/seen_corrupt" \
  --output_dir "${OUTPUT_ROOT}/analysis" \
  --probe_device "${PROBE_DEVICE}" \
  --probe_epochs "${PROBE_EPOCHS}" \
  --probe_batch_size "${PROBE_BATCH_SIZE}" \
  --max_train_queries_per_condition "${MAX_TRAIN_QUERIES}" \
  > "${OUTPUT_ROOT}/analysis.log" 2>&1

# Only clean up after analyze_p5_separability.py returns successfully. If probe
# analysis fails, the shards are kept for debugging/re-running analysis.
cleanup_dump_files

echo "[P5-DIAG] done: ${OUTPUT_ROOT}/analysis/RESULTS.md"
