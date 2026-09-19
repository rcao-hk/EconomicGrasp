#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/cache}

SPLITS=${SPLITS:-train,test_seen,test_similar,test_novel}
MINE_GPUS=${MINE_GPUS:-0,1,2,3,4,5}
SOURCES=${SOURCES:-pred,sensor,rendered,cad_full}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
QUERY_EVAL_NUM=${QUERY_EVAL_NUM:-64}
QUERY_EVAL_MODE=${QUERY_EVAL_MODE:-topk_uniform}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
VOXEL_SIZE=${VOXEL_SIZE:-0.005}
NUM_WORKERS=${NUM_WORKERS:-0}
MAX_SAMPLES_PER_SHARD=${MAX_SAMPLES_PER_SHARD:-0}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME=${RESUME:-1}
REPAIR_INVALID_CACHE=${REPAIR_INVALID_CACHE:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-20}
MEMORY_REPORT_EVERY=${MEMORY_REPORT_EVERY:-10}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -f "${STAGE1_CKPT}" ]] || { echo "Missing Stage-1 checkpoint: ${STAGE1_CKPT}" >&2; exit 2; }
[[ -d "${DATASET_ROOT}" ]] || { echo "Missing dataset root: ${DATASET_ROOT}" >&2; exit 2; }
mkdir -p "${CACHE_ROOT}"

IFS=',' read -r -a GPU_ARRAY <<< "${MINE_GPUS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
NUM_SHARDS=${#GPU_ARRAY[@]}
[[ ${NUM_SHARDS} -gt 0 ]] || { echo "No MINE_GPUS specified" >&2; exit 2; }

for raw_split in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw_split}" | xargs)"
  [[ -n "${split}" ]] || continue
  echo "[REP-P0-MINE] split=${split} shards=${NUM_SHARDS} sharding=scene-level resume=${RESUME} workers=${NUM_WORKERS}"

  pids=()
  names=()
  for shard in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$shard]}"
    log_dir="${WORK_ROOT}/logs/mine/${split}"
    mkdir -p "${log_dir}"
    args=(
      "${ROOT_DIR}/mine_rep_p0_geometry_sources.py"
      --dataset_root "${DATASET_ROOT}"
      --checkpoint_path "${STAGE1_CKPT}"
      --output_root "${CACHE_ROOT}"
      --split "${split}"
      --camera realsense
      --sample_interval "${SAMPLE_INTERVAL}"
      --query_eval_num "${QUERY_EVAL_NUM}"
      --query_eval_mode "${QUERY_EVAL_MODE}"
      "--offsets_mm=${OFFSETS_MM}"
      --sources "${SOURCES}"
      --voxel_size "${VOXEL_SIZE}"
      --num_workers "${NUM_WORKERS}"
      --shard_id "${shard}"
      --num_shards "${NUM_SHARDS}"
      --max_samples "${MAX_SAMPLES_PER_SHARD}"
      --fc_mode "${FC_MODE}"
      --verify_n "${VERIFY_N}"
      --progress_every "${PROGRESS_EVERY}"
      --memory_report_every "${MEMORY_REPORT_EVERY}"
    )
    [[ "${OVERWRITE}" == "1" ]] && args+=(--overwrite)
    [[ "${RESUME}" == "1" ]] && args+=(--resume)
    [[ "${REPAIR_INVALID_CACHE}" == "1" ]] && args+=(--repair_invalid_cache)
    echo "  launch shard=${shard} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}"       >"${log_dir}/shard_$(printf '%02d' "${shard}").log" 2>&1 &
    pids+=("$!")
    names+=("shard_${shard}")
  done

  failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[REP-P0-MINE][ERROR] ${split}/${names[$i]} failed" >&2
      failed=1
    fi
  done
  [[ ${failed} -eq 0 ]] || exit 1
  count=$(find "${CACHE_ROOT}/${split}" -path '*/ann_*.npz' -type f | wc -l)
  echo "[REP-P0-MINE] split=${split} cache_frames=${count}"
done

echo "[REP-P0-MINE] completed: ${CACHE_ROOT}"
