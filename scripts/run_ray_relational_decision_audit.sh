#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
TRAIN_CACHE_ROOT=${TRAIN_CACHE_ROOT:-${WORK_ROOT}/cache}
VAL_CACHE_ROOT=${VAL_CACHE_ROOT:-${WORK_ROOT}/cache_val}
REL_ROOT=${REL_ROOT:-${WORK_ROOT}/relational_selective}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/relational_decision_audit}

CHECKPOINTS=${CHECKPOINTS:-best,latest}
VAL_SCENE_START=${VAL_SCENE_START:-100}
AUDIT_GPU=${AUDIT_GPU:-0}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}
SAVE_PER_QUERY=${SAVE_PER_QUERY:-0}
DELTA_SWEEP_POINTS=${DELTA_SWEEP_POINTS:-81}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -d "${TRAIN_CACHE_ROOT}" ]] || { echo "Train cache not found: ${TRAIN_CACHE_ROOT}" >&2; exit 2; }
[[ -d "${VAL_CACHE_ROOT}" ]] || { echo "Validation cache not found: ${VAL_CACHE_ROOT}" >&2; exit 2; }
[[ -d "${REL_ROOT}" ]] || { echo "Relational training root not found: ${REL_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a CKPTS <<< "${CHECKPOINTS}"
for raw_name in "${CKPTS[@]}"; do
  name="$(echo "${raw_name}" | xargs)"
  [[ -n "${name}" ]] || continue

  case "${name}" in
    best) ckpt="${REL_ROOT}/checkpoint_best.tar" ;;
    latest) ckpt="${REL_ROOT}/checkpoint_latest.tar" ;;
    *)
      if [[ -f "${name}" ]]; then
        ckpt="${name}"
        name="$(basename "${name}" .tar)"
      else
        echo "Unknown checkpoint entry or missing file: ${name}" >&2
        exit 2
      fi
      ;;
  esac

  [[ -f "${ckpt}" ]] || { echo "Checkpoint not found: ${ckpt}" >&2; exit 2; }
  out="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out}"

  args=(
    "${ROOT_DIR}/analyze_ray_relational_decision_audit.py"
    --train_cache_root "${TRAIN_CACHE_ROOT}"
    --val_cache_root "${VAL_CACHE_ROOT}"
    --selector_checkpoint "${ckpt}"
    --output_dir "${out}"
    --val_scene_start "${VAL_SCENE_START}"
    --device cuda:0
    --max_train_frames "${MAX_TRAIN_FRAMES}"
    --max_val_frames "${MAX_VAL_FRAMES}"
    --progress_every "${PROGRESS_EVERY}"
    --delta_sweep_points "${DELTA_SWEEP_POINTS}"
  )
  [[ "${SAVE_PER_QUERY}" == "1" ]] && args+=(--save_per_query)

  echo "[REL-AUDIT-RUN] checkpoint=${name} path=${ckpt} gpu=${AUDIT_GPU}"
  CUDA_VISIBLE_DEVICES="${AUDIT_GPU}" "${PYTHON_BIN}" "${args[@]}" \
    2>&1 | tee "${out}/audit.log"
done

echo "[REL-AUDIT-RUN] completed: ${OUTPUT_ROOT}"
find "${OUTPUT_ROOT}" -maxdepth 2 -name REPORT.md -print
