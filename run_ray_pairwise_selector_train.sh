#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/cache_train}
TRAIN_OUT=${TRAIN_OUT:-${WORK_ROOT}/train}
PHASES=${PHASES:-mine,train}

MINE_GPUS=${MINE_GPUS:-0,1,2,3,4,5}
MINE_SAMPLE_INTERVAL=${MINE_SAMPLE_INTERVAL:-0.1}
MINE_QUERY_EVAL_NUM=${MINE_QUERY_EVAL_NUM:-64}
MINE_QUERY_EVAL_MODE=${MINE_QUERY_EVAL_MODE:-topk_uniform}
MINE_NUM_WORKERS=${MINE_NUM_WORKERS:-2}
MINE_MAX_SAMPLES=${MINE_MAX_SAMPLES:-0}
MINE_OVERWRITE=${MINE_OVERWRITE:-0}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
NOOP_CHECK_SAMPLES=${NOOP_CHECK_SAMPLES:-1}
NOOP_ATOL=${NOOP_ATOL:-5e-5}

TRAIN_GPU=${TRAIN_GPU:-0}
VAL_SCENE_START=${VAL_SCENE_START:-80}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-256}
DROPOUT=${DROPOUT:-0.1}
REG_WEIGHT=${REG_WEIGHT:-1.0}
SIGN_WEIGHT=${SIGN_WEIGHT:-0.5}
LISTWISE_WEIGHT=${LISTWISE_WEIGHT:-0.5}
TARGET_TEMPERATURE=${TARGET_TEMPERATURE:-0.15}
THRESHOLD_MAX=${THRESHOLD_MAX:-0.30}
THRESHOLD_STEPS=${THRESHOLD_STEPS:-31}
SEED=${SEED:-0}
RESUME=${RESUME:-}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}

export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

has_phase() { [[ ",${PHASES}," == *",$1,"* ]]; }
mkdir -p "${WORK_ROOT}" "${CACHE_ROOT}" "${TRAIN_OUT}" "${WORK_ROOT}/logs"

if has_phase mine; then
  IFS=',' read -r -a GPU_ARRAY <<< "${MINE_GPUS}"
  NSHARDS=${#GPU_ARRAY[@]}
  [[ ${NSHARDS} -gt 0 ]] || { echo "No MINE_GPUS specified" >&2; exit 2; }
  PIDS=()
  echo "[PAIR-PIPE] mining train exact-action cache with ${NSHARDS} shards"
  for i in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$i]}"
    args=(
      "${SCRIPT_DIR}/mine_ray_pairwise_exact_cache.py"
      --dataset_root "${DATASET_ROOT}"
      --checkpoint_path "${STAGE1_CKPT}"
      --output_dir "${CACHE_ROOT}"
      --split train
      --sample_interval "${MINE_SAMPLE_INTERVAL}"
      --max_samples "${MINE_MAX_SAMPLES}"
      --num_workers "${MINE_NUM_WORKERS}"
      --pose_depth_mode "${POSE_DEPTH_MODE}"
      --offsets_mm "${OFFSETS_MM}"
      --query_eval_num "${MINE_QUERY_EVAL_NUM}"
      --query_eval_mode "${MINE_QUERY_EVAL_MODE}"
      --fc_mode "${FC_MODE}"
      --verify_n "${VERIFY_N}"
      --shard_id "${i}"
      --num_shards "${NSHARDS}"
      --noop_check_samples "${NOOP_CHECK_SAMPLES}"
      --noop_atol "${NOOP_ATOL}"
    )
    if [[ "${MINE_OVERWRITE}" == "1" ]]; then args+=(--overwrite); fi
    echo "[PAIR-PIPE][MINE] shard=${i}/${NSHARDS} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" \
      >"${WORK_ROOT}/logs/mine_shard${i}.log" 2>&1 &
    PIDS+=("$!")
  done
  failed=0
  for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
  [[ ${failed} -eq 0 ]] || { echo "Mining failed; inspect ${WORK_ROOT}/logs/mine_shard*.log" >&2; exit 1; }
  echo "[PAIR-PIPE] cache mining complete"
fi

if has_phase train; then
  args=(
    "${SCRIPT_DIR}/train_ray_pairwise_selector.py"
    --cache_root "${CACHE_ROOT}"
    --output_dir "${TRAIN_OUT}"
    --val_scene_start "${VAL_SCENE_START}"
    --epochs "${EPOCHS}"
    --learning_rate "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --hidden_dim "${HIDDEN_DIM}"
    --dropout "${DROPOUT}"
    --reg_weight "${REG_WEIGHT}"
    --sign_weight "${SIGN_WEIGHT}"
    --listwise_weight "${LISTWISE_WEIGHT}"
    --target_temperature "${TARGET_TEMPERATURE}"
    --threshold_max "${THRESHOLD_MAX}"
    --threshold_steps "${THRESHOLD_STEPS}"
    --seed "${SEED}"
    --device cuda:0
    --max_train_frames "${MAX_TRAIN_FRAMES}"
    --max_val_frames "${MAX_VAL_FRAMES}"
  )
  if [[ -n "${RESUME}" ]]; then args+=(--resume "${RESUME}"); fi
  echo "[PAIR-PIPE][TRAIN] gpu=${TRAIN_GPU} cache=${CACHE_ROOT}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${PYTHON_BIN}" "${args[@]}" \
    2>&1 | tee "${WORK_ROOT}/logs/train.log"
fi

echo "[PAIR-PIPE] completed"
echo "  cache: ${CACHE_ROOT}"
echo "  best selector: ${TRAIN_OUT}/checkpoint_best.tar"
