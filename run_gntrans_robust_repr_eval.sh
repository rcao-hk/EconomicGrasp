#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?Set DATASET_ROOT to GraspNet root}"
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT to GN-Trans RGB root}"
: "${CKPT:?Set CKPT to a robust representation checkpoint}"

GPUS="${GPUS:-0,1,2,3,4,5}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
VARIANT="${VARIANT:-dual}"          # metric | wide | dual
WIDE_SCALE="${WIDE_SCALE:-1.5}"
IMAGE_RADIUS_PX="${IMAGE_RADIUS_PX:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/gntrans_robust_eval/${VARIANT}}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"
POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
GNTRANS_COLLISION_SOURCE="${GNTRANS_COLLISION_SOURCE:-original_sensor}"
REMOVE_DUMP="${REMOVE_DUMP:-0}"

case "${VARIANT}" in
  metric|wide|dual) ;;
  *) echo "VARIANT must be metric, wide, or dual" >&2; exit 2 ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
if [[ ${#GPU_ARRAY[@]} -lt 1 ]]; then
  echo "Provide at least one GPU." >&2
  exit 2
fi
if [[ "${COLLISION_THRESH}" == "0" || "${COLLISION_THRESH}" == "0.0" ]]; then
  GNTRANS_COLLISION_SOURCE="none"
fi
mkdir -p "${OUTPUT_ROOT}"

JOB_DOMAIN=()
JOB_SPLIT=()
for domain in original gntrans; do
  for split in "${SPLIT_ARRAY[@]}"; do
    JOB_DOMAIN+=("${domain}")
    JOB_SPLIT+=("${split}")
  done
done

run_job() {
  local idx="$1"
  local gpu="$2"
  local domain="${JOB_DOMAIN[$idx]}"
  local split="${JOB_SPLIT[$idx]}"
  local out="${OUTPUT_ROOT}/${domain}/${split}"
  mkdir -p "${out}"
  echo "[ROBUST-EVAL] ${domain}/${split} variant=${VARIANT} GPU=${gpu}"

  common=(
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${CKPT}"
    --test_mode "${split}"
    --save_dir "${out}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --collision_thresh "${COLLISION_THRESH}"
    --collision_voxel_size "${COLLISION_VOXEL_SIZE}"
    --robust_support_mode "${VARIANT}"
    --robust_wide_scale "${WIDE_SCALE}"
    --robust_image_radius_px "${IMAGE_RADIUS_PX}"
    --multi_modal --use_cdf --kview_mode A1
    --pose_depth_mode "${POSE_DEPTH_MODE}"
  )

  if [[ "${domain}" == "original" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python inference_cva_robust_repr.py "${common[@]}" \
      > "${out}/inference.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python inference_cva_gntrans_robust_repr.py \
      --gntrans_rgb_root "${GNTRANS_RGB_ROOT}" \
      --gntrans_collision_source "${GNTRANS_COLLISION_SOURCE}" \
      "${common[@]}" \
      > "${out}/inference.log" 2>&1
  fi
}

njobs=${#JOB_DOMAIN[@]}
ngpu=${#GPU_ARRAY[@]}
start=0
while [[ ${start} -lt ${njobs} ]]; do
  pids=()
  for ((j=0; j<ngpu && start+j<njobs; j++)); do
    idx=$((start+j))
    run_job "${idx}" "${GPU_ARRAY[$j]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then status=1; fi
  done
  if [[ ${status} -ne 0 ]]; then
    echo "At least one inference job failed; inspect */*/inference.log" >&2
    exit ${status}
  fi
  start=$((start+ngpu))
done

STRIDE=$(python - <<PY
x=float("${SAMPLE_INTERVAL}")
if x <= 0:
    raise SystemExit("SAMPLE_INTERVAL must be > 0")
print(1 if x >= 1 else max(1, round(1.0/x)))
PY
)

for domain in original gntrans; do
  for split in "${SPLIT_ARRAY[@]}"; do
    out="${OUTPUT_ROOT}/${domain}/${split}"
    extra=()
    if [[ "${REMOVE_DUMP}" == "1" ]]; then extra+=(--remove_dump); fi
    echo "[ROBUST-EVAL] official evaluator ${domain}/${split}, stride=${STRIDE}"
    python eval.py \
      --dataset_root "${DATASET_ROOT}" \
      --dump_dir "${out}" \
      --camera realsense \
      --split "${split}" \
      --num_workers "${EVAL_NUM_WORKERS}" \
      --sample_interval "${STRIDE}" \
      "${extra[@]}" \
      > "${out}/evaluation.log" 2>&1
  done
done

echo "[ROBUST-EVAL] complete: ${OUTPUT_ROOT}"
