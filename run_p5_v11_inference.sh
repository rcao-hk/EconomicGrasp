#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${CKPT:?Set CKPT to checkpoint_best_native.tar or another P5-v1.1 checkpoint}"

GPUS="${GPUS:-0,1,2}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/p5_v11_eval/repair}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"
MODE="${MODE:-repair}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"
P5_MAX_BATCHES="${P5_MAX_BATCHES:-0}"

if [[ "${RUN_EVAL}" == "1" && "${P5_MAX_BATCHES}" != "0" ]]; then
  echo "RUN_EVAL=1 requires P5_MAX_BATCHES=0" >&2
  exit 2
fi

# Fail early if the supplied checkpoint is not from P5-v1.1.
python - "${CKPT}" <<'PY'
import sys, torch
path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)
if int(ck.get("p5_protocol_version", -1)) != 2:
    raise SystemExit(f"{path} is not a P5-v1.1 checkpoint (p5_protocol_version != 2)")
if int(ck.get("p5_smoke_max_batches", 0)) != 0:
    raise SystemExit("P5-v1.1 official inference rejects smoke checkpoints")
print(f"[P5-v1.1] checkpoint ok: epoch={ck.get('epoch')} best_native={ck.get('best_native_set_violation_m')}")
PY

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
if [[ ${#GPU_ARRAY[@]} -lt ${#SPLIT_ARRAY[@]} ]]; then
  echo "Need at least one GPU per requested split for parallel inference." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
pids=()
for i in "${!SPLIT_ARRAY[@]}"; do
  split="${SPLIT_ARRAY[$i]}"
  gpu="${GPU_ARRAY[$i]}"
  out="${OUTPUT_ROOT}/${split}"
  mkdir -p "${out}"
  echo "[P5-v1.1] launch ${split} on GPU ${gpu} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python inference_p5_v11.py \
    --dataset_root "${DATASET_ROOT}" \
    --checkpoint_path "${CKPT}" \
    --test_mode "${split}" \
    --save_dir "${out}" \
    --sample_interval "${SAMPLE_INTERVAL}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --collision_thresh "${COLLISION_THRESH}" \
    --collision_voxel_size "${COLLISION_VOXEL_SIZE}" \
    --graspness_mode scene \
    --kview_mode A1 \
    --p5_mode "${MODE}" \
    --p5_max_batches "${P5_MAX_BATCHES}" \
    > "${out}/inference.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ ${status} -ne 0 ]]; then
  echo "At least one P5-v1.1 inference split failed; inspect inference.log." >&2
  exit ${status}
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  for split in "${SPLIT_ARRAY[@]}"; do
    out="${OUTPUT_ROOT}/${split}"
    echo "[P5-v1.1] evaluate ${split}"
    CUDA_VISIBLE_DEVICES="" python inference_p5_v11.py \
      --dataset_root "${DATASET_ROOT}" \
      --test_mode "${split}" \
      --save_dir "${out}" \
      --sample_interval "${SAMPLE_INTERVAL}" \
      --collision_thresh "${COLLISION_THRESH}" \
      --p5_mode "${MODE}" \
      --p5_eval_only \
      --p5_eval_workers "${EVAL_NUM_WORKERS}" \
      > "${out}/evaluation_stdout.log" 2>&1
  done
fi

echo "[P5-v1.1] done: ${OUTPUT_ROOT}"
