#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
TRAIN_CACHE_ROOT=${TRAIN_CACHE_ROOT:-${WORK_ROOT}/cache}
VAL_CACHE_ROOT=${VAL_CACHE_ROOT:-${WORK_ROOT}/cache_val}
MODEL_ROOT=${MODEL_ROOT:-${WORK_ROOT}/feature_ablation_3layer}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/feature_ablation_3layer_infer}

FEATURE_MODES=${FEATURE_MODES:-raw_offset,selected_residual,mean_residual,selected_mean,full}
INFER_GPUS=${INFER_GPUS:-0,3,5,6}
EVAL_SPLITS=${EVAL_SPLITS:-train,val}
VAL_SCENE_START=${VAL_SCENE_START:-100}
VAL_SCENE_END=${VAL_SCENE_END:-130}
MAX_FRAMES=${MAX_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}
SELECTOR_THRESHOLD=${SELECTOR_THRESHOLD:-}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

mkdir -p "${OUTPUT_ROOT}"
IFS=',' read -r -a MODES <<< "${FEATURE_MODES}"
IFS=',' read -r -a GPUS <<< "${INFER_GPUS}"
IFS=',' read -r -a SPLITS <<< "${EVAL_SPLITS}"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "No INFER_GPUS" >&2; exit 2; }

PIDS=(); NAMES=()
wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[ABLATE-INFER-RUN][ERROR] ${NAMES[$i]} failed" >&2
      failed=1
    fi
  done
  PIDS=(); NAMES=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_one() {
  local mode="$1" split="$2" gpu="$3"
  local ckpt="${MODEL_ROOT}/${mode}/checkpoint_best.tar"
  local cache scene_min scene_max out
  [[ -f "${ckpt}" ]] || { echo "Missing checkpoint: ${ckpt}" >&2; exit 2; }
  case "${split}" in
    train)
      cache="${TRAIN_CACHE_ROOT}"; scene_min=0; scene_max="${VAL_SCENE_START}" ;;
    val)
      cache="${VAL_CACHE_ROOT}"; scene_min="${VAL_SCENE_START}"; scene_max="${VAL_SCENE_END}" ;;
    *) echo "Unknown EVAL_SPLITS entry: ${split}" >&2; exit 2 ;;
  esac
  out="${OUTPUT_ROOT}/${mode}/${split}"
  mkdir -p "${out}"
  args=(
    "${ROOT_DIR}/infer_ray_pairwise_feature_ablation.py"
    --cache_root "${cache}"
    --selector_checkpoint "${ckpt}"
    --output_dir "${out}"
    --split_name "${split}"
    --scene_min "${scene_min}"
    --scene_max "${scene_max}"
    --device cuda:0
    --max_frames "${MAX_FRAMES}"
    --progress_every "${PROGRESS_EVERY}"
  )
  [[ -n "${SELECTOR_THRESHOLD}" ]] && args+=(--threshold "${SELECTOR_THRESHOLD}")
  echo "[ABLATE-INFER-RUN] mode=${mode} split=${split} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" >"${out}/infer.log" 2>&1 &
  PIDS+=("$!"); NAMES+=("${mode}/${split}")
}

slot=0
for mode_raw in "${MODES[@]}"; do
  mode="$(echo "${mode_raw}" | xargs)"; [[ -n "${mode}" ]] || continue
  for split_raw in "${SPLITS[@]}"; do
    split="$(echo "${split_raw}" | xargs)"; [[ -n "${split}" ]] || continue
    launch_one "${mode}" "${split}" "${GPUS[$slot]}"
    slot=$((slot+1))
    if [[ ${slot} -ge ${#GPUS[@]} ]]; then
      wait_wave
      slot=0
    fi
  done
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

echo "[ABLATE-INFER-RUN] complete: ${OUTPUT_ROOT}"
find "${OUTPUT_ROOT}" -maxdepth 3 -name summary.json -print
