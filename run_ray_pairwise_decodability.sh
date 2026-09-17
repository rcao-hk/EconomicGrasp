#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/cache_train_seenval}
TRAIN_OUT=${TRAIN_OUT:-${WORK_ROOT}/train}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/decodability}
VAL_SCENE_START=${VAL_SCENE_START:-100}
ANALYSIS_GPU=${ANALYSIS_GPU:-0}
CHECKPOINTS=${CHECKPOINTS:-best,latest}
TOP_FRACS=${TOP_FRACS:-0.05,0.10,0.20}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}
SELECTOR_THRESHOLD=${SELECTOR_THRESHOLD:-}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -d "${CACHE_ROOT}" ]] || { echo "Cache root not found: ${CACHE_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a CKPT_NAMES <<< "${CHECKPOINTS}"
for name_raw in "${CKPT_NAMES[@]}"; do
  name="$(echo "${name_raw}" | xargs)"
  [[ -n "${name}" ]] || continue
  case "${name}" in
    best) ckpt="${TRAIN_OUT}/checkpoint_best.tar" ;;
    latest) ckpt="${TRAIN_OUT}/checkpoint_latest.tar" ;;
    *)
      if [[ -f "${name}" ]]; then
        ckpt="${name}"
        name="$(basename "${name}" .tar)"
      else
        echo "Unknown CHECKPOINTS entry or missing file: ${name}" >&2
        exit 2
      fi
      ;;
  esac
  [[ -f "${ckpt}" ]] || { echo "Checkpoint not found: ${ckpt}" >&2; exit 2; }
  out="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out}"

  args=(
    "${SCRIPT_DIR}/analyze_ray_pairwise_decodability.py"
    --cache_root "${CACHE_ROOT}"
    --selector_checkpoint "${ckpt}"
    --output_dir "${out}"
    --val_scene_start "${VAL_SCENE_START}"
    --device cuda:0
    --top_fracs "${TOP_FRACS}"
    --max_train_frames "${MAX_TRAIN_FRAMES}"
    --max_val_frames "${MAX_VAL_FRAMES}"
    --progress_every "${PROGRESS_EVERY}"
  )
  if [[ -n "${SELECTOR_THRESHOLD}" ]]; then
    args+=(--threshold "${SELECTOR_THRESHOLD}")
  fi

  echo "[DECODE-RUN] checkpoint=${name} path=${ckpt} gpu=${ANALYSIS_GPU}"
  CUDA_VISIBLE_DEVICES="${ANALYSIS_GPU}" "${PYTHON_BIN}" "${args[@]}" \
    2>&1 | tee "${out}/analysis.log"
done

echo "[DECODE-RUN] completed: ${OUTPUT_ROOT}"
find "${OUTPUT_ROOT}" -maxdepth 2 -name REPORT.md -print
