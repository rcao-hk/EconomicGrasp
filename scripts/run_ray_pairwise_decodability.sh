#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}

# Cache layout accepted by this launcher:
#   TRAIN_CACHE_ROOT may be either a pure train cache (scene 0000-0099)
#   or a previously merged cache that also contains validation scenes.
#   VAL_CACHE_ROOT may be a pure validation cache or may contain extra scenes.
# The union is built by scene-id range, not by assuming either source root is pure:
#   scene_id <  VAL_SCENE_START -> TRAIN_CACHE_ROOT
#   scene_id >= VAL_SCENE_START -> VAL_CACHE_ROOT
# This keeps train/val provenance explicit without copying .npz payloads.
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
  local ignored_train_val_dirs=0 ignored_val_train_dirs=0

  rm -rf "${UNION_CACHE_ROOT}"
  mkdir -p "${UNION_CACHE_ROOT}"
  shopt -s nullglob

  # TRAIN_CACHE_ROOT is authoritative only for train-range scenes.  It is legal
  # for this root to already contain merged validation scenes from an earlier
  # pipeline; those scenes are deliberately ignored here.
  for src in "${TRAIN_CACHE_ROOT}"/scene_*; do
    [[ -d "${src}" ]] || continue
    base="$(basename "${src}")"
    sid=$((10#${base#scene_}))
    if (( sid >= VAL_SCENE_START )); then
      ignored_train_val_dirs=$((ignored_train_val_dirs + 1))
      continue
    fi
    dst="${UNION_CACHE_ROOT}/${base}"
    ln -s "$(readlink -f "${src}")" "${dst}"
    train_dirs=$((train_dirs + 1))
  done

  # VAL_CACHE_ROOT is authoritative only for validation-range scenes. Extra
  # train-range scenes are ignored rather than treated as a fatal layout error.
  for src in "${VAL_CACHE_ROOT}"/scene_*; do
    [[ -d "${src}" ]] || continue
    base="$(basename "${src}")"
    sid=$((10#${base#scene_}))
    if (( sid < VAL_SCENE_START )); then
      ignored_val_train_dirs=$((ignored_val_train_dirs + 1))
      continue
    fi
    dst="${UNION_CACHE_ROOT}/${base}"
    if [[ -e "${dst}" || -L "${dst}" ]]; then
      echo "Duplicate validation scene while building union cache: ${base}" >&2
      exit 2
    fi
    ln -s "$(readlink -f "${src}")" "${dst}"
    val_dirs=$((val_dirs + 1))
  done
  shopt -u nullglob

  echo "[DECODE-RUN][UNION] train_scene_dirs=${train_dirs} val_scene_dirs=${val_dirs} root=${UNION_CACHE_ROOT}"
  if (( ignored_train_val_dirs > 0 )); then
    echo "[DECODE-RUN][UNION] ignored ${ignored_train_val_dirs} validation-range scene dirs already present in TRAIN_CACHE_ROOT"
  fi
  if (( ignored_val_train_dirs > 0 )); then
    echo "[DECODE-RUN][UNION] ignored ${ignored_val_train_dirs} train-range scene dirs present in VAL_CACHE_ROOT"
  fi

  TRAIN_CACHE_ENV="${TRAIN_CACHE_ROOT}" VAL_CACHE_ENV="${VAL_CACHE_ROOT}" UNION_CACHE_ENV="${UNION_CACHE_ROOT}" VAL_START_ENV="${VAL_SCENE_START}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

train_root = Path(os.environ["TRAIN_CACHE_ENV"])
val_root = Path(os.environ["VAL_CACHE_ENV"])
union_root = Path(os.environ["UNION_CACHE_ENV"])
start = int(os.environ["VAL_START_ENV"])

def files(root):
    return list(root.glob("scene_*/ann_*.npz"))

def sid(p):
    return int(p.parent.name.split("_")[-1])

# Only compare the ranges that each source root is authoritative for.
train_src_all = files(train_root)
val_src_all = files(val_root)
train_src = [p for p in train_src_all if sid(p) < start]
val_src = [p for p in val_src_all if sid(p) >= start]
union = files(union_root)
train_union = [p for p in union if sid(p) < start]
val_union = [p for p in union if sid(p) >= start]

print(
    f"[DECODE-RUN][CACHE] train_src_valid={len(train_src)} val_src_valid={len(val_src)} "
    f"union_train={len(train_union)} union_val={len(val_union)} "
    f"train_root_total={len(train_src_all)} val_root_total={len(val_src_all)}"
)
if not train_src or not val_src:
    raise SystemExit(
        "Train or validation cache is empty in its authoritative scene-id range."
    )
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
    "analyze_ray_pairwise_decodability.py"
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
echo "  train source cache: ${TRAIN_CACHE_ROOT} (only scene < ${VAL_SCENE_START} used)"
echo "  validation source cache: ${VAL_CACHE_ROOT} (only scene >= ${VAL_SCENE_START} used)"
echo "  union cache (symlinks only): ${UNION_CACHE_ROOT}"
find "${OUTPUT_ROOT}" -maxdepth 2 -name REPORT.md -print
