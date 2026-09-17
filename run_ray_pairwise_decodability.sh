#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}

# Current canonical layout keeps selector-fit and validation caches separate:
#   TRAIN_CACHE_ROOT: GraspNet train scenes 0000-0099
#   VAL_CACHE_ROOT:   test_seen scenes 0100-0129, used as validation only
# The analyzer itself expects one cache root, so this launcher creates a
# symlink-only union view. No .npz payload is copied or modified.
TRAIN_CACHE_ROOT=${TRAIN_CACHE_ROOT:-${WORK_ROOT}/cache}
VAL_CACHE_ROOT=${VAL_CACHE_ROOT:-${WORK_ROOT}/cache_val}
UNION_CACHE_ROOT=${UNION_CACHE_ROOT:-${WORK_ROOT}/cache_decodability_union}

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

[[ -d "${TRAIN_CACHE_ROOT}" ]] || { echo "Train cache root not found: ${TRAIN_CACHE_ROOT}" >&2; exit 2; }
[[ -d "${VAL_CACHE_ROOT}" ]] || { echo "Validation cache root not found: ${VAL_CACHE_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

prepare_union_cache() {
  local src dst base sid
  local train_dirs=0 val_dirs=0

  rm -rf "${UNION_CACHE_ROOT}"
  mkdir -p "${UNION_CACHE_ROOT}"
  shopt -s nullglob

  for src in "${TRAIN_CACHE_ROOT}"/scene_*; do
    [[ -d "${src}" ]] || continue
    base="$(basename "${src}")"
    sid=$((10#${base#scene_}))
    if (( sid >= VAL_SCENE_START )); then
      echo "Unexpected validation-range scene in TRAIN_CACHE_ROOT: ${src}" >&2
      exit 2
    fi
    dst="${UNION_CACHE_ROOT}/${base}"
    ln -s "$(readlink -f "${src}")" "${dst}"
    train_dirs=$((train_dirs + 1))
  done

  for src in "${VAL_CACHE_ROOT}"/scene_*; do
    [[ -d "${src}" ]] || continue
    base="$(basename "${src}")"
    sid=$((10#${base#scene_}))
    if (( sid < VAL_SCENE_START )); then
      echo "Unexpected train-range scene in VAL_CACHE_ROOT: ${src}" >&2
      exit 2
    fi
    dst="${UNION_CACHE_ROOT}/${base}"
    if [[ -e "${dst}" || -L "${dst}" ]]; then
      echo "Duplicate scene while building union cache: ${base}" >&2
      exit 2
    fi
    ln -s "$(readlink -f "${src}")" "${dst}"
    val_dirs=$((val_dirs + 1))
  done
  shopt -u nullglob

  echo "[DECODE-RUN][UNION] train_scene_dirs=${train_dirs} val_scene_dirs=${val_dirs} root=${UNION_CACHE_ROOT}"

  TRAIN_CACHE_ENV="${TRAIN_CACHE_ROOT}" VAL_CACHE_ENV="${VAL_CACHE_ROOT}" UNION_CACHE_ENV="${UNION_CACHE_ROOT}" VAL_START_ENV="${VAL_SCENE_START}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

train_root = Path(os.environ["TRAIN_CACHE_ENV"])
val_root = Path(os.environ["VAL_CACHE_ENV"])
union_root = Path(os.environ["UNION_CACHE_ENV"])
start = int(os.environ["VAL_START_ENV"])

def files(root):
    return list(root.glob("scene_*/ann_*.npz"))

train_src = files(train_root)
val_src = files(val_root)
union = files(union_root)
train_union = [p for p in union if int(p.parent.name.split("_")[-1]) < start]
val_union = [p for p in union if int(p.parent.name.split("_")[-1]) >= start]

print(
    f"[DECODE-RUN][CACHE] train_src={len(train_src)} val_src={len(val_src)} "
    f"union_train={len(train_union)} union_val={len(val_union)}"
)
if not train_src or not val_src:
    raise SystemExit("Train or validation cache is empty.")
if len(train_src) != len(train_union) or len(val_src) != len(val_union):
    raise SystemExit(
        "Union cache count mismatch; check scene IDs and cache roots before analysis."
    )
PY
}

prepare_union_cache

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
    --cache_root "${UNION_CACHE_ROOT}"
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
echo "  train cache: ${TRAIN_CACHE_ROOT}"
echo "  validation cache: ${VAL_CACHE_ROOT}"
echo "  union cache (symlinks only): ${UNION_CACHE_ROOT}"
find "${OUTPUT_ROOT}" -maxdepth 2 -name REPORT.md -print
