#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/cache_train_seenval}
VAL_CACHE_ROOT=${VAL_CACHE_ROOT:-${WORK_ROOT}/cache_val_seen_tmp}
TRAIN_OUT=${TRAIN_OUT:-${WORK_ROOT}/train}
PHASES=${PHASES:-mine,train}

# Canonical protocol:
#   train split     -> scenes 0000-0099, selector fitting
#   test_seen split -> scenes 0100-0129, validation/checkpoint/threshold selection
# test_seen is therefore NOT an untouched test split in this protocol.
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
mkdir -p "${WORK_ROOT}" "${CACHE_ROOT}" "${VAL_CACHE_ROOT}" "${TRAIN_OUT}" "${WORK_ROOT}/logs"

# IMPORTANT: mine train and validation into DIFFERENT roots.
# The miner's cheap pre-forward skip uses a split-local dataset index; if both
# splits share a root, test_seen local scene 0..29 can be mistaken for existing
# train scene 0..29 before the batch exposes the true global scene id 100..129.
mine_one_split() {
  local split="$1"
  local output_root="$2"
  local tag="${split//\//_}"
  local -a pids=()
  local i gpu failed pid src dst

  mkdir -p "${output_root}"
  echo "[PAIR-PIPE] mining split=${split} output=${output_root} with ${NSHARDS} shards"
  for i in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$i]}"
    args=(
      "${SCRIPT_DIR}/mine_ray_pairwise_exact_cache.py"
      --dataset_root "${DATASET_ROOT}"
      --checkpoint_path "${STAGE1_CKPT}"
      --output_dir "${output_root}"
      --split "${split}"
      --sample_interval "${MINE_SAMPLE_INTERVAL}"
      --max_samples "${MINE_MAX_SAMPLES}"
      --num_workers "${MINE_NUM_WORKERS}"
      --pose_depth_mode "${POSE_DEPTH_MODE}"
      "--offsets_mm=${OFFSETS_MM}"
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

  for i in "${!GPU_ARRAY[@]}"; do
    src="${output_root}/protocol_shard$(printf '%02d' "${i}").json"
    dst="${output_root}/protocol_${tag}_shard$(printf '%02d' "${i}").json"
    if [[ -f "${src}" ]]; then
      mv -f "${src}" "${dst}"
    fi
  done
  echo "[PAIR-PIPE] mining split=${split} complete"
}

merge_validation_cache() {
  local src dst base
  local copied=0 skipped=0
  shopt -s nullglob
  for src in "${VAL_CACHE_ROOT}"/scene_*; do
    [[ -d "${src}" ]] || continue
    base="$(basename "${src}")"
    dst="${CACHE_ROOT}/${base}"
    if [[ -e "${dst}" ]]; then
      if [[ "${MINE_OVERWRITE}" == "1" ]]; then
        rm -rf "${dst}"
      else
        skipped=$((skipped + 1))
        continue
      fi
    fi
    # Prefer hard links (same WORK_ROOT filesystem, no duplicated cache bytes).
    # Fall back to a normal copy if hard-linking is unavailable.
    if cp -al "${src}" "${dst}" 2>/dev/null; then
      :
    else
      cp -a "${src}" "${dst}"
    fi
    copied=$((copied + 1))
  done
  for src in "${VAL_CACHE_ROOT}"/protocol_${VAL_MINE_SPLIT}_shard*.json; do
    [[ -f "${src}" ]] || continue
    cp -f "${src}" "${CACHE_ROOT}/$(basename "${src}")"
  done
  shopt -u nullglob
  echo "[PAIR-PIPE] validation cache merged into ${CACHE_ROOT}: scene_dirs_added=${copied}, existing=${skipped}"
}

check_combined_cache() {
  CACHE_ROOT_ENV="${CACHE_ROOT}" VAL_START_ENV="${VAL_SCENE_START}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["CACHE_ROOT_ENV"])
start = int(os.environ["VAL_START_ENV"])
paths = list(root.glob("scene_*/ann_*.npz"))
train = []
val = []
for p in paths:
    sid = int(p.parent.name.split("_")[-1])
    (train if sid < start else val).append(p)
print(f"[PAIR-PIPE][CACHE] train={len(train)} val={len(val)} val_scene_start={start}")
if not train or not val:
    raise SystemExit(
        f"Combined cache incomplete: train={len(train)}, val={len(val)}. "
        "Re-run PHASES=mine,train after pulling the split-root fix."
    )
PY
}

if has_phase mine; then
  IFS=',' read -r -a GPU_ARRAY <<< "${MINE_GPUS}"
  NSHARDS=${#GPU_ARRAY[@]}
  [[ ${NSHARDS} -gt 0 ]] || { echo "No MINE_GPUS specified" >&2; exit 2; }

  echo "[PAIR-PIPE] validation protocol: train=${TRAIN_MINE_SPLIT}, val=${VAL_MINE_SPLIT}, VAL_SCENE_START=${VAL_SCENE_START}"
  if [[ "${TRAIN_MINE_SPLIT}" != "train" || "${VAL_MINE_SPLIT}" != "test_seen" || "${VAL_SCENE_START}" != "100" ]]; then
    echo "[PAIR-PIPE][WARN] non-canonical split override requested." >&2
  fi

  mine_one_split "${TRAIN_MINE_SPLIT}" "${CACHE_ROOT}"
  mine_one_split "${VAL_MINE_SPLIT}" "${VAL_CACHE_ROOT}"
  merge_validation_cache
  check_combined_cache
fi

if has_phase train; then
  check_combined_cache
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
echo "  combined cache: ${CACHE_ROOT}"
echo "  validation mining cache: ${VAL_CACHE_ROOT}"
echo "  train split: ${TRAIN_MINE_SPLIT} (expected scenes 0000-0099)"
echo "  validation split: ${VAL_MINE_SPLIT} (expected scenes 0100-0129)"
echo "  best selector: ${TRAIN_OUT}/checkpoint_best.tar"
