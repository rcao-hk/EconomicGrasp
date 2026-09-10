#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${CKPT:?Set CKPT to trained P2-v2 checkpoint}"

GPUS="${GPUS:-0,1,2}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/p2_ray_v2_inference}"
CAMERA="${CAMERA:-realsense}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"
GRASPNESS_MODE="${GRASPNESS_MODE:-scene}"
V2_SELECTION="${V2_SELECTION:-relational}"
V2_FINAL_SCORE="${V2_FINAL_SCORE:-raw}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
SAVE_NOCOLLISION="${SAVE_NOCOLLISION:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"
V2_MAX_BATCHES="${V2_MAX_BATCHES:-0}"

case "${V2_SELECTION}" in relational|zero|raw|supported) ;; *) echo "Invalid V2_SELECTION=${V2_SELECTION}" >&2; exit 2;; esac
case "${V2_FINAL_SCORE}" in raw|contextual|product) ;; *) echo "Invalid V2_FINAL_SCORE=${V2_FINAL_SCORE}" >&2; exit 2;; esac
case "${RUN_EVAL}" in 0|1) ;; *) echo "RUN_EVAL must be 0 or 1" >&2; exit 2;; esac
if [[ "${RUN_EVAL}" == "1" && "${V2_MAX_BATCHES}" != "0" ]]; then
  echo "RUN_EVAL=1 requires V2_MAX_BATCHES=0; AP needs a complete sampled split." >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then echo "No GPU specified" >&2; exit 2; fi
mkdir -p "${OUTPUT_ROOT}"

PIDS=()
launch_split() {
  local split="$1" gpu="$2" out="${OUTPUT_ROOT}/${split}"
  mkdir -p "${out}"
  args=(
    "${SCRIPT_DIR}/inference_ray_grasp_v2.py"
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${CKPT}"
    --camera "${CAMERA}"
    --test_mode "${split}"
    --save_dir "${out}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --graspness_mode "${GRASPNESS_MODE}"
    --v2_selection "${V2_SELECTION}"
    --v2_final_score "${V2_FINAL_SCORE}"
    --collision_thresh "${COLLISION_THRESH}"
    --collision_voxel_size "${COLLISION_VOXEL_SIZE}"
    --v2_max_batches "${V2_MAX_BATCHES}"
  )
  if [[ "${SAVE_NOCOLLISION}" == "1" ]]; then args+=(--save_nocollision); fi
  echo "[P2-V2][INFER] split=${split} gpu=${gpu} selection=${V2_SELECTION} final_score=${V2_FINAL_SCORE} out=${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${args[@]}" >"${out}/inference.log" 2>&1 &
  PIDS+=("$!")
}

wait_wave() {
  local failed=0 pid
  for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
  PIDS=()
  [[ "${failed}" == "0" ]] || { echo "P2-v2 inference worker failed; inspect per-split inference.log" >&2; exit 1; }
}

slot=0
for raw in "${SPLIT_ARRAY[@]}"; do
  split="$(echo "${raw}" | xargs)"
  [[ -n "${split}" ]] || continue
  launch_split "${split}" "${GPU_ARRAY[$slot]}"
  slot=$((slot + 1))
  if [[ "${slot}" -ge "${#GPU_ARRAY[@]}" ]]; then
    wait_wave
    slot=0
  fi
done
if [[ "${#PIDS[@]}" -gt 0 ]]; then wait_wave; fi

echo "[P2-V2] inference completed"

# GraspNet evaluation is deliberately sequential to avoid multiple heavy CPU pools.
if [[ "${RUN_EVAL}" == "1" ]]; then
  for raw in "${SPLIT_ARRAY[@]}"; do
    split="$(echo "${raw}" | xargs)"
    [[ -n "${split}" ]] || continue
    out="${OUTPUT_ROOT}/${split}"
    echo "[P2-V2][EVAL] split=${split}"
    python "${SCRIPT_DIR}/inference_ray_grasp_v2.py" \
      --dataset_root "${DATASET_ROOT}" \
      --camera "${CAMERA}" \
      --test_mode "${split}" \
      --save_dir "${out}" \
      --v2_eval_only \
      --v2_eval_workers "${EVAL_NUM_WORKERS}" \
      >"${out}/evaluation_stdout.log" 2>&1
    cat "${out}/ray_v2_eval_summary.json"
  done
fi

echo "[P2-V2] completed: ${OUTPUT_ROOT}"
