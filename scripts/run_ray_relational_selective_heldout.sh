#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
SELECTOR_CKPT=${SELECTOR_CKPT:-${WORK_ROOT}/relational_selective/checkpoint_best.tar}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/relational_selective_heldout}

GPUS=${GPUS:-0,1}
SPLITS=${SPLITS:-test_similar,test_novel}
CAMERA=${CAMERA:-realsense}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
QUERY_EVAL_NUM=${QUERY_EVAL_NUM:-128}
QUERY_EVAL_MODE=${QUERY_EVAL_MODE:-topk_uniform}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
NOOP_CHECK_SAMPLES=${NOOP_CHECK_SAMPLES:-2}
NOOP_ATOL=${NOOP_ATOL:-5e-5}
PROFILE_TIMING=${PROFILE_TIMING:-1}
SAVE_CANDIDATE_ROWS=${SAVE_CANDIDATE_ROWS:-0}
MOVE_THRESHOLD=${MOVE_THRESHOLD:-}

export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -f "${SELECTOR_CKPT}" ]] || { echo "Selector checkpoint not found: ${SELECTOR_CKPT}" >&2; exit 2; }
[[ -f "${STAGE1_CKPT}" ]] || { echo "Stage-1 checkpoint not found: ${STAGE1_CKPT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }

PIDS=()
ACTIVE=()
wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REL-HELDOUT][ERROR] ${ACTIVE[$i]} failed; inspect diagnostic.log" >&2
      failed=1
    fi
  done
  PIDS=(); ACTIVE=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_split() {
  local split="$1" gpu="$2" out="${OUTPUT_ROOT}/${split}"
  mkdir -p "${out}"
  local args=(
    "${ROOT_DIR}/test_ray_relational_selective.py"
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${STAGE1_CKPT}"
    --selector_checkpoint "${SELECTOR_CKPT}"
    --output_dir "${out}"
    --split "${split}"
    --camera "${CAMERA}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --max_samples "${MAX_SAMPLES}"
    --num_workers "${NUM_WORKERS}"
    --pose_depth_mode "${POSE_DEPTH_MODE}"
    "--offsets_mm=${OFFSETS_MM}"
    --query_eval_num "${QUERY_EVAL_NUM}"
    --query_eval_mode "${QUERY_EVAL_MODE}"
    --fc_mode "${FC_MODE}"
    --verify_n "${VERIFY_N}"
    --noop_check_samples "${NOOP_CHECK_SAMPLES}"
    --noop_atol "${NOOP_ATOL}"
  )
  [[ "${PROFILE_TIMING}" == "1" ]] && args+=(--profile_timing)
  [[ "${SAVE_CANDIDATE_ROWS}" == "1" ]] && args+=(--save_candidate_rows)
  [[ -n "${MOVE_THRESHOLD}" ]] && args+=(--move_threshold "${MOVE_THRESHOLD}")

  echo "[REL-HELDOUT][LAUNCH] split=${split} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" >"${out}/diagnostic.log" 2>&1 &
  PIDS+=("$!"); ACTIVE+=("${split}")
}

slot=0
for raw_split in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw_split}" | xargs)"
  [[ -n "${split}" ]] || continue
  launch_split "${split}" "${GPU_ARRAY[$slot]}"
  slot=$((slot+1))
  if [[ ${slot} -ge ${#GPU_ARRAY[@]} ]]; then
    wait_wave; slot=0
  fi
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

OUTPUT_ROOT_ENV="${OUTPUT_ROOT}" SPLITS_ENV="${SPLITS}" "${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ["OUTPUT_ROOT_ENV"])
for split in [x.strip() for x in os.environ["SPLITS_ENV"].split(",") if x.strip()]:
    p=root/split/"summary.json"
    if not p.is_file():
        continue
    d=json.loads(p.read_text())
    a=d["aggregate"]
    du=d["gaps"]["learned_minus_native_utility"]
    ds=100*d["gaps"]["learned_minus_native_success08"]
    print(
        f"[REL-HELDOUT] {split:12s} "
        f"dU={du:+.5f} dS08={ds:+.2f}pp "
        f"rescue={100*d['learned_rescue08']:.2f}% harm={100*d['learned_harm08']:.2f}% "
        f"move={100*d['move_rate']:.2f}% headroom={100*d['utility_headroom_recovery']:.2f}%"
    )
PY

echo "[REL-HELDOUT] completed: ${OUTPUT_ROOT}"
