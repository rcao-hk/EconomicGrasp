#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/cache_train_seenval}
TRAIN_OUT=${TRAIN_OUT:-${WORK_ROOT}/train}
PHASES=${PHASES:-mine,train}

# Cache protocol:
#   train split     -> scenes 0000-0099, used for selector fitting
#   test_seen split -> scenes 0100-0129, used ONLY for validation/checkpoint and
#                      native-fallback threshold selection
# Because test_seen is consumed as validation, it is no longer a held-out test
# split for this protocol. Final generalization tests should focus on
# test_similar/test_novel.
TRAIN_MINE_SPLIT=${TRAIN_MINE_SPLIT:-train}
VAL_MINE_SPLIT=${VAL_MINE_SPLIT:-test_seen}
VAL_SCENE_START=${VAL_SCENE_START:-100}

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

# Launch one dataset split over all requested GPUs. Both train and validation
# splits deliberately write scene_xxxx/ann_xxxx.npz files into the SAME cache
# root; their GraspNet scene IDs are disjoint. The miner's generic protocol file
# names are renamed after each wave so the second split cannot overwrite the
# first split's metadata.
mine_one_split() {
  local split="$1"
  local tag="${split//\//_}"
  local -a pids=()
  local i gpu failed pid src dst

  echo "[PAIR-PIPE] mining split=${split} exact-action cache with ${NSHARDS} shards"
  for i in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$i]}"
    args=(
      "${SCRIPT_DIR}/mine_ray_pairwise_exact_cache.py"
      --dataset_root "${DATASET_ROOT}"
      --checkpoint_path "${STAGE1_CKPT}"
      --output_dir "${CACHE_ROOT}"
      --split "${split}"
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
    echo "[PAIR-PIPE][MINE] split=${split} shard=${i}/${NSHARDS} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" \
      >"${WORK_ROOT}/logs/mine_${tag}_shard${i}.log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  [[ ${failed} -eq 0 ]] || {
    echo "Mining split=${split} failed; inspect ${WORK_ROOT}/logs/mine_${tag}_shard*.log" >&2
    exit 1
  }

  # Preserve each split's mining protocol instead of letting the next split
  # overwrite protocol_shardXX.json in the shared cache root.
  for i in "${!GPU_ARRAY[@]}"; do
    src="${CACHE_ROOT}/protocol_shard$(printf '%02d' "${i}").json"
    dst="${CACHE_ROOT}/protocol_${tag}_shard$(printf '%02d' "${i}").json"
    if [[ -f "${src}" ]]; then
      mv -f "${src}" "${dst}"
    fi
  done
  echo "[PAIR-PIPE] mining split=${split} complete"
}

if has_phase mine; then
  IFS=',' read -r -a GPU_ARRAY <<< "${MINE_GPUS}"
  NSHARDS=${#GPU_ARRAY[@]}
  [[ ${NSHARDS} -gt 0 ]] || { echo "No MINE_GPUS specified" >&2; exit 2; }

  echo "[PAIR-PIPE] validation protocol: train=${TRAIN_MINE_SPLIT}, val=${VAL_MINE_SPLIT}, VAL_SCENE_START=${VAL_SCENE_START}"
  if [[ "${TRAIN_MINE_SPLIT}" != "train" || "${VAL_MINE_SPLIT}" != "test_seen" || "${VAL_SCENE_START}" != "100" ]]; then
    echo "[PAIR-PIPE][WARN] non-canonical split override requested. Ensure cache scene IDs satisfy train < VAL_SCENE_START and val >= VAL_SCENE_START." >&2
  fi

  mine_one_split "${TRAIN_MINE_SPLIT}"
  mine_one_split "${VAL_MINE_SPLIT}"
  echo "[PAIR-PIPE] shared train+validation cache mining complete: ${CACHE_ROOT}"
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
  echo "[PAIR-PIPE][TRAIN] gpu=${TRAIN_GPU} cache=${CACHE_ROOT} train_scene<${VAL_SCENE_START} val_scene>=${VAL_SCENE_START}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${PYTHON_BIN}" "${args[@]}" \
    2>&1 | tee "${WORK_ROOT}/logs/train.log"
fi

echo "[PAIR-PIPE] completed"
echo "  cache: ${CACHE_ROOT}"
echo "  train split: ${TRAIN_MINE_SPLIT} (expected scenes 0000-0099)"
echo "  validation split: ${VAL_MINE_SPLIT} (expected scenes 0100-0129)"
echo "  best selector: ${TRAIN_OUT}/checkpoint_best.tar"
