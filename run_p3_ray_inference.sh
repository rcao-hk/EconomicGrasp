#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${CKPT:?Set CKPT to a trained P3 checkpoint}"

GPUS="${GPUS:-0,1,2}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/p3_ray_inference}"
CAMERA="${CAMERA:-realsense}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"
GRASPNESS_MODE="${GRASPNESS_MODE:-scene}"
SELECTION_SCORE="${SELECTION_SCORE:-joint}"
FINAL_SCORE="${FINAL_SCORE:-same}"
FORCE_ZERO="${FORCE_ZERO:-0}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
SAVE_NOCOLLISION="${SAVE_NOCOLLISION:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"
P3_MAX_BATCHES="${P3_MAX_BATCHES:-0}"

case "${SELECTION_SCORE}" in joint|raw) ;; *) echo "SELECTION_SCORE must be joint or raw" >&2; exit 2;; esac
case "${FINAL_SCORE}" in same|joint|raw) ;; *) echo "FINAL_SCORE must be same, joint or raw" >&2; exit 2;; esac
case "${FORCE_ZERO}" in 0|1) ;; *) echo "FORCE_ZERO must be 0 or 1" >&2; exit 2;; esac
case "${RUN_EVAL}" in 0|1) ;; *) echo "RUN_EVAL must be 0 or 1" >&2; exit 2;; esac
if [[ "${RUN_EVAL}" == "1" && "${P3_MAX_BATCHES}" != "0" ]]; then
  echo "RUN_EVAL=1 requires P3_MAX_BATCHES=0." >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
[[ "${#GPU_ARRAY[@]}" -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

PIDS=()
launch_split() {
  local split="$1" gpu="$2" out="${OUTPUT_ROOT}/${split}"
  mkdir -p "${out}"
  args=("${SCRIPT_DIR}/inference_p3_ray.py"
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${CKPT}"
    --camera "${CAMERA}"
    --test_mode "${split}"
    --save_dir "${out}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --graspness_mode "${GRASPNESS_MODE}"
    --p3_selection_score "${SELECTION_SCORE}"
    --p3_final_score "${FINAL_SCORE}"
    --p3_max_batches "${P3_MAX_BATCHES}"
    --collision_thresh "${COLLISION_THRESH}"
    --collision_voxel_size "${COLLISION_VOXEL_SIZE}"
  )
  if [[ "${FORCE_ZERO}" == "1" ]]; then args+=(--p3_force_zero); fi
  if [[ "${SAVE_NOCOLLISION}" == "1" ]]; then args+=(--save_nocollision); fi
  echo "[P3][INFER] split=${split} gpu=${gpu} selection=${SELECTION_SCORE} final=${FINAL_SCORE} zero=${FORCE_ZERO}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${args[@]}" >"${out}/inference.log" 2>&1 &
  PIDS+=("$!")
}

slot=0
for raw_split in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw_split}" | xargs)"
  [[ -n "${split}" ]] || continue
  launch_split "${split}" "${GPU_ARRAY[$slot]}"
  slot=$((slot + 1))
  if [[ "${slot}" -ge "${#GPU_ARRAY[@]}" ]]; then
    failed=0
    for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
    PIDS=(); slot=0
    [[ "${failed}" == "0" ]] || { echo "P3 inference worker failed; inspect per-split inference.log" >&2; exit 1; }
  fi
done
if [[ "${#PIDS[@]}" -gt 0 ]]; then
  failed=0
  for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" == "0" ]] || { echo "P3 inference worker failed; inspect per-split inference.log" >&2; exit 1; }
fi

echo "[P3] inference completed"

# Keep GraspNet multiprocessing pools serial across splits.
if [[ "${RUN_EVAL}" == "1" ]]; then
  for raw_split in "${SPLIT_ARRAY[@]}"; do
    split="$(echo "${raw_split}" | xargs)"
    [[ -n "${split}" ]] || continue
    out="${OUTPUT_ROOT}/${split}"
    echo "[P3][EVAL] split=${split}"
    python "${SCRIPT_DIR}/inference_p3_ray.py" \
      --dataset_root "${DATASET_ROOT}" \
      --camera "${CAMERA}" \
      --test_mode "${split}" \
      --save_dir "${out}" \
      --p3_eval_only \
      --p3_eval_workers "${EVAL_NUM_WORKERS}" \
      >"${out}/evaluation_stdout.log" 2>&1
    cat "${out}/p3_eval_summary.json"
  done
fi

echo "[P3] completed: ${OUTPUT_ROOT}"
