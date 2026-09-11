#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${CKPT:?Set CKPT to trained ray-confidence checkpoint}"

GPUS="${GPUS:-0,1,2}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/ray_confidence_inference}"
CAMERA="${CAMERA:-realsense}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"
GRASPNESS_MODE="${GRASPNESS_MODE:-scene}"
SCORE_MODE="${SCORE_MODE:-calibrated}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
SAVE_NOCOLLISION="${SAVE_NOCOLLISION:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"
MAX_BATCHES="${MAX_BATCHES:-0}"

case "${SCORE_MODE}" in raw|calibrated) ;; *) echo "Invalid SCORE_MODE=${SCORE_MODE}" >&2; exit 2;; esac
case "${RUN_EVAL}" in 0|1) ;; *) echo "RUN_EVAL must be 0 or 1" >&2; exit 2;; esac
if [[ "${RUN_EVAL}" == "1" && "${MAX_BATCHES}" != "0" ]]; then
  echo "RUN_EVAL=1 requires MAX_BATCHES=0" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
mkdir -p "${OUTPUT_ROOT}"

PIDS=()
launch() {
  local split="$1" gpu="$2" out="${OUTPUT_ROOT}/${split}"
  mkdir -p "${out}"
  args=("${SCRIPT_DIR}/inference_ray_confidence.py"
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${CKPT}"
    --camera "${CAMERA}"
    --test_mode "${split}"
    --save_dir "${out}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --graspness_mode "${GRASPNESS_MODE}"
    --rc_score_mode "${SCORE_MODE}"
    --collision_thresh "${COLLISION_THRESH}"
    --collision_voxel_size "${COLLISION_VOXEL_SIZE}"
    --rc_max_batches "${MAX_BATCHES}")
  if [[ "${SAVE_NOCOLLISION}" == "1" ]]; then args+=(--save_nocollision); fi
  echo "[RC][INFER] split=${split} gpu=${gpu} score=${SCORE_MODE} out=${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${args[@]}" >"${out}/inference.log" 2>&1 &
  PIDS+=("$!")
}

slot=0
for raw_split in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw_split}" | xargs)"
  [[ -n "${split}" ]] || continue
  launch "${split}" "${GPU_ARRAY[$slot]}"
  slot=$((slot + 1))
  if [[ "${slot}" -ge "${#GPU_ARRAY[@]}" ]]; then
    failed=0
    for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
    PIDS=(); slot=0
    [[ "${failed}" == "0" ]] || { echo "RC inference worker failed; inspect inference.log" >&2; exit 1; }
  fi
done
if [[ "${#PIDS[@]}" -gt 0 ]]; then
  failed=0
  for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" == "0" ]] || { echo "RC inference worker failed; inspect inference.log" >&2; exit 1; }
fi

echo "[RC] inference completed"

if [[ "${RUN_EVAL}" == "1" ]]; then
  for raw_split in "${SPLIT_ARRAY[@]}"; do
    split="$(echo "${raw_split}" | xargs)"
    [[ -n "${split}" ]] || continue
    out="${OUTPUT_ROOT}/${split}"
    echo "[RC][EVAL] split=${split} out=${out}"
    python "${SCRIPT_DIR}/inference_ray_confidence.py" \
      --dataset_root "${DATASET_ROOT}" \
      --camera "${CAMERA}" \
      --test_mode "${split}" \
      --save_dir "${out}" \
      --rc_eval_only \
      --rc_eval_workers "${EVAL_NUM_WORKERS}" \
      >"${out}/evaluation_stdout.log" 2>&1
    cat "${out}/rc_eval_summary.json"
  done
fi

echo "[RC] completed: ${OUTPUT_ROOT}"
