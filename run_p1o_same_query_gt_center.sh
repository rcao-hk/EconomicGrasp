#!/usr/bin/env bash
set -euo pipefail

# P1-O same-query GT-center oracle over GraspNet test splits.
# Generates paired Stage-1 baseline and GT-center-oracle predictions, then
# evaluates both variants with the repository's GraspNet evaluator.
#
# Required:
#   DATASET_ROOT=/path/to/graspnet
#   STAGE1_CKPT=/path/to/controlled_stage1_checkpoint.tar
#
# Useful:
#   GPUS=0,1,2
#   SPLITS=test_seen,test_similar,test_novel
#   OUTPUT_ROOT=/path/to/p1o_same_query_gt_center
#   SAMPLE_INTERVAL=0.1
#   POSE_DEPTH_MODE=global_film
#   USE_FUSE_DEPTH=1
#   GRASPNESS_MODE=scene
#   COLLISION_THRESH=0.01
#   BATCH_SIZE=3
#   P1O_MAX_BATCHES=0
#   RUN_EVAL=1
#   EVAL_NUM_WORKERS=10
#   EVAL_REMOVE_DUMP=0
#   EXTRA_ARGS="..."

: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${STAGE1_CKPT:?Set STAGE1_CKPT}"

GPUS="${GPUS:-0}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-log/p1o_same_query_gt_center}"
CAMERA="${CAMERA:-realsense}"
POSE_DEPTH_MODE="${POSE_DEPTH_MODE:-global_film}"
USE_FUSE_DEPTH="${USE_FUSE_DEPTH:-1}"
GRASPNESS_MODE="${GRASPNESS_MODE:-scene}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
SAVE_NOCOLLISION="${SAVE_NOCOLLISION:-0}"
P1O_MAX_BATCHES="${P1O_MAX_BATCHES:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"
EVAL_REMOVE_DUMP="${EVAL_REMOVE_DUMP:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "${GRASPNESS_MODE}" in
  scene|instance) ;;
  *)
    echo "Invalid GRASPNESS_MODE=${GRASPNESS_MODE}; expected scene or instance" >&2
    exit 2
    ;;
esac

case "${RUN_EVAL}" in
  0|1) ;;
  *)
    echo "Invalid RUN_EVAL=${RUN_EVAL}; expected 0 or 1" >&2
    exit 2
    ;;
esac

case "${EVAL_REMOVE_DUMP}" in
  0|1) ;;
  *)
    echo "Invalid EVAL_REMOVE_DUMP=${EVAL_REMOVE_DUMP}; expected 0 or 1" >&2
    exit 2
    ;;
esac

if [[ "${RUN_EVAL}" == "1" && "${P1O_MAX_BATCHES}" != "0" ]]; then
  echo "RUN_EVAL=1 requires P1O_MAX_BATCHES=0 because GraspNet evaluation needs a complete sampled split." >&2
  echo "For a smoke test, use RUN_EVAL=0 with P1O_MAX_BATCHES>0." >&2
  exit 2
fi

# Inference uses SAMPLE_INTERVAL as a fraction (0.1 -> keep 10%).
# eval.py uses an integer frame stride (10 -> evaluate 0,10,20,...).
# Match the exact conversion used by inference_cva_distill._build_subset():
#   stride = max(1, round(1 / sample_interval)).
EVAL_SAMPLE_STRIDE="$(python - "${SAMPLE_INTERVAL}" <<'PY'
import sys
x = float(sys.argv[1])
if x <= 0.0:
    raise SystemExit("SAMPLE_INTERVAL must be positive")
print(1 if x >= 1.0 else max(1, int(round(1.0 / x))))
PY
)"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
NGPU="${#GPU_ARRAY[@]}"
if [[ "${NGPU}" -le 0 ]]; then
  echo "No GPU specified" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
declare -a ACTIVE_PIDS=()

launch_split() {
  local SPLIT="$1"
  local GPU="$2"
  local OUT="${OUTPUT_ROOT}/${SPLIT}"
  mkdir -p "${OUT}"

  local -a ARGS=(
    diagnose_p1o_same_query_gt_center.py
    --dataset_root "${DATASET_ROOT}"
    --camera "${CAMERA}"
    --checkpoint_path "${STAGE1_CKPT}"
    --distill_stage 1
    --save_dir "${OUT}"
    --test_mode "${SPLIT}"
    --multi_modal
    --use_cdf
    --pose_depth_mode "${POSE_DEPTH_MODE}"
    --graspness_mode "${GRASPNESS_MODE}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --collision_thresh "${COLLISION_THRESH}"
    --collision_voxel_size "${COLLISION_VOXEL_SIZE}"
    --p1o_max_batches "${P1O_MAX_BATCHES}"
  )

  if [[ "${USE_FUSE_DEPTH}" == "1" ]]; then
    ARGS+=(--use_fuse_depth)
  fi
  if [[ "${SAVE_NOCOLLISION}" == "1" ]]; then
    ARGS+=(--save_nocollision)
  fi

  echo "[P1-O] split=${SPLIT} GPU=${GPU} out=${OUT} sample=${SAMPLE_INTERVAL}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${GPU}" python "${ARGS[@]}" ${EXTRA_ARGS} \
    >"${OUT}/p1o.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
}

wait_wave() {
  local pid
  local failed=0
  for pid in "${ACTIVE_PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  ACTIVE_PIDS=()
  if [[ "${failed}" != "0" ]]; then
    echo "At least one P1-O worker failed. Check split p1o.log files." >&2
    exit 1
  fi
}

slot=0
for SPLIT in "${SPLIT_ARRAY[@]}"; do
  SPLIT="$(echo "${SPLIT}" | xargs)"
  [[ -n "${SPLIT}" ]] || continue
  GPU="${GPU_ARRAY[$slot]}"
  launch_split "${SPLIT}" "${GPU}"
  slot=$((slot + 1))
  if [[ "${slot}" -ge "${NGPU}" ]]; then
    wait_wave
    slot=0
  fi
done

if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
  wait_wave
fi

echo "[P1-O] inference completed. Outputs: ${OUTPUT_ROOT}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[P1-O][EVAL] starting paired GraspNet evaluation: inference_sample=${SAMPLE_INTERVAL}, eval_stride=${EVAL_SAMPLE_STRIDE}, workers=${EVAL_NUM_WORKERS}"

  for SPLIT_RAW in "${SPLIT_ARRAY[@]}"; do
    SPLIT="$(echo "${SPLIT_RAW}" | xargs)"
    [[ -n "${SPLIT}" ]] || continue
    OUT="${OUTPUT_ROOT}/${SPLIT}"

    for VARIANT in baseline gt_center; do
      DUMP_DIR="${OUT}/${VARIANT}"
      if [[ ! -d "${DUMP_DIR}" ]]; then
        echo "[P1-O][EVAL] missing prediction directory: ${DUMP_DIR}" >&2
        exit 1
      fi

      EVAL_LOG="${OUT}/eval_${VARIANT}.log"
      EVAL_ARGS=(
        eval.py
        --dataset_root "${DATASET_ROOT}"
        --dump_dir "${DUMP_DIR}"
        --camera "${CAMERA}"
        --split "${SPLIT}"
        --num_workers "${EVAL_NUM_WORKERS}"
        --sample_interval "${EVAL_SAMPLE_STRIDE}"
      )
      if [[ "${EVAL_REMOVE_DUMP}" == "1" ]]; then
        EVAL_ARGS+=(--remove_dump)
      fi

      echo "[P1-O][EVAL] split=${SPLIT} variant=${VARIANT} dump=${DUMP_DIR}"
      python "${EVAL_ARGS[@]}" 2>&1 | tee "${EVAL_LOG}"
    done
  done

  # Compact paired summary from the exact result tensors saved by eval.py.
  python - "${OUTPUT_ROOT}" "${CAMERA}" "${SPLITS}" <<'PY'
import json
import os
import sys

import numpy as np

root, camera, split_text = sys.argv[1:4]
splits = [s.strip() for s in split_text.split(",") if s.strip()]
variants = ("baseline", "gt_center")
summary = {
    "protocol": "p1o_same_query_gt_center_paired_graspnet_eval_v1",
    "camera": camera,
    "splits": {},
}

for split in splits:
    split_record = {}
    for variant in variants:
        path = os.path.join(root, split, variant, f"ap_{split}_{camera}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        result = np.asarray(np.load(path), dtype=np.float64)
        if result.ndim < 1 or result.shape[-1] < 4:
            raise RuntimeError(
                f"Unexpected GraspNet result shape for {split}/{variant}: {result.shape}"
            )
        split_record[variant] = {
            "result_npy": os.path.abspath(path),
            "result_shape": list(result.shape),
            "ap": float(result.mean()),
            "ap0.4": float(result[..., 1].mean()),
            "ap0.8": float(result[..., 3].mean()),
            "ap_percent": 100.0 * float(result.mean()),
            "ap0.4_percent": 100.0 * float(result[..., 1].mean()),
            "ap0.8_percent": 100.0 * float(result[..., 3].mean()),
        }

    base = split_record["baseline"]
    oracle = split_record["gt_center"]
    split_record["delta_gt_center_minus_baseline"] = {
        "ap": oracle["ap"] - base["ap"],
        "ap0.4": oracle["ap0.4"] - base["ap0.4"],
        "ap0.8": oracle["ap0.8"] - base["ap0.8"],
        "ap_points": oracle["ap_percent"] - base["ap_percent"],
        "ap0.4_points": oracle["ap0.4_percent"] - base["ap0.4_percent"],
        "ap0.8_points": oracle["ap0.8_percent"] - base["ap0.8_percent"],
    }
    summary["splits"][split] = split_record

if splits:
    for variant in variants:
        summary[f"mean_{variant}_ap"] = float(
            np.mean([summary["splits"][s][variant]["ap"] for s in splits])
        )
        summary[f"mean_{variant}_ap_percent"] = 100.0 * summary[f"mean_{variant}_ap"]
    summary["mean_delta_ap"] = (
        summary["mean_gt_center_ap"] - summary["mean_baseline_ap"]
    )
    summary["mean_delta_ap_points"] = 100.0 * summary["mean_delta_ap"]

out_path = os.path.join(root, "p1o_eval_summary.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print("[P1-O][EVAL] paired summary")
for split in splits:
    rec = summary["splits"][split]
    b = rec["baseline"]["ap_percent"]
    o = rec["gt_center"]["ap_percent"]
    d = rec["delta_gt_center_minus_baseline"]["ap_points"]
    print(f"  {split}: baseline={b:.4f} gt_center={o:.4f} delta={d:+.4f} AP points")
if splits:
    print(
        "  mean: baseline={:.4f} gt_center={:.4f} delta={:+.4f} AP points".format(
            summary["mean_baseline_ap_percent"],
            summary["mean_gt_center_ap_percent"],
            summary["mean_delta_ap_points"],
        )
    )
print(f"[P1-O][EVAL] saved {out_path}")
PY
fi

echo "[P1-O] completed. Outputs: ${OUTPUT_ROOT}"
