#!/usr/bin/env bash
# Frozen e19 point-depth baseline, Original RealSense 10% Seen/Similar/Novel.
set -euo pipefail
: "${DATASET_ROOT:?Set DATASET_ROOT to GraspNet root}"
: "${CKPT:?Set CKPT to checkpoint_epoch_019.tar}"
: "${OUTPUT_ROOT:?Set NEW OUTPUT_ROOT for RealSense evaluation}"
GPUS="${GPUS:-0,1,2}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-16}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
ENABLE_INFERENCE="${ENABLE_INFERENCE:-1}"
ENABLE_EVAL="${ENABLE_EVAL:-1}"
MAX_BATCHES="${MAX_BATCHES:-0}"
SEED="${SEED:-0}"
for name in ENABLE_INFERENCE ENABLE_EVAL; do
  if [[ "${!name}" != 0 && "${!name}" != 1 ]]; then
    echo "$name must be 0/1" >&2; exit 2
  fi
done
if [[ "$MAX_BATCHES" != 0 && "$ENABLE_EVAL" == 1 ]]; then
  echo 'Smoke (MAX_BATCHES>0) is not official AP; use ENABLE_EVAL=0' >&2; exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
IFS=',' read -r -a SPLIT_ARRAY <<< "$SPLITS"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then echo 'No GPU' >&2; exit 2; fi
for split in "${SPLIT_ARRAY[@]}"; do
  case "$split" in test_seen|test_similar|test_novel) ;; *) echo "Bad split: $split" >&2; exit 2 ;; esac
done
if [[ "$ENABLE_INFERENCE" == 1 && ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2; exit 2
fi
mkdir -p "$OUTPUT_ROOT"
inference_one () {
  local split="$1" gpu="$2"
  local dst="$OUTPUT_ROOT/$split"
  if [[ -e "$dst/inference.log" || -e "$dst/inference_protocol.json" ]]; then
    echo "Refusing to overwrite existing inference in $dst" >&2; return 1
  fi
  mkdir -p "$dst"
  echo "[POINT-EVAL] inference $split GPU=$gpu"
  (set -C; CUDA_VISIBLE_DEVICES="$gpu" MAX_BATCHES="$MAX_BATCHES" \
    python inference_point_depth_pose.py \
    --dataset_root "$DATASET_ROOT" --checkpoint_path "$CKPT" \
    --test_mode "$split" --camera realsense \
    --save_dir "$dst" --sample_interval 0.1 \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --collision_thresh "$COLLISION_THRESH" \
    --collision_voxel_size "$COLLISION_VOXEL_SIZE" \
    --pose_depth_mode global_film --min_depth 0.2 --max_depth 1.0 \
    --seed "$SEED" > "$dst/inference.log" 2>&1)
}
if [[ "$ENABLE_INFERENCE" == 1 ]]; then
  i=0
  while (( i < ${#SPLIT_ARRAY[@]} )); do
    pids=()
    for ((j=0; j<${#GPU_ARRAY[@]} && i+j<${#SPLIT_ARRAY[@]}; j++)); do
      inference_one "${SPLIT_ARRAY[i+j]}" "${GPU_ARRAY[j]}" &
      pids+=("$!")
    done
    code=0
    for pid in "${pids[@]}"; do wait "$pid" || code=1; done
    if [[ "$code" != 0 ]]; then
      echo "Inference failure: inspect $OUTPUT_ROOT/*/inference.log" >&2; exit 1
    fi
    i=$((i+${#GPU_ARRAY[@]}))
  done
fi
if [[ "$ENABLE_EVAL" == 1 ]]; then
  # Official evaluator processes are deliberately serialized (each uses 16 workers).
  for split in "${SPLIT_ARRAY[@]}"; do
    dst="$OUTPUT_ROOT/$split"
    if [[ -e "$dst/evaluation.log" || -e "$dst/ap_${split}_realsense.npy" ]]; then
      echo "Existing AP/evaluation log in $dst; refusing overwrite" >&2; exit 1
    fi
    python - "$dst" "$split" <<'PY'
import sys,json
from pathlib import Path
dst,split=Path(sys.argv[1]),sys.argv[2]
protocol=json.loads((dst/"inference_protocol.json").read_text())
summary=json.loads((dst/"inference_summary.json").read_text())
n=len(list(dst.glob("scene_*/realsense/*.npy")))
if (summary["processed_frames"],summary["expected_frames"],n)!=(780,780,780):
    raise SystemExit(f"Expected 780 complete frame NPYs; got {summary}, found {n}")
if protocol["status"]!="formal_inference" or protocol["test_mode"]!=split:
    raise SystemExit("Not a full formal inference dump; evaluation blocked")
PY
    echo "[POINT-EVAL] official $split (stride=10)"
    (set -C; python eval.py --dataset_root "$DATASET_ROOT" \
      --dump_dir "$dst" --camera realsense \
      --split "$split" --sample_interval 10 \
      --num_workers "$EVAL_NUM_WORKERS" > "$dst/evaluation.log" 2>&1)
    python - "$dst" "$split" <<'PY'
import sys,numpy as np
from pathlib import Path
p=Path(sys.argv[1])/f"ap_{sys.argv[2]}_realsense.npy"
a=np.load(p,allow_pickle=False)
if a.shape!=(30,26,50,6) or not np.isfinite(a).all():
    raise SystemExit(f"Bad official AP tensor: {p} shape={a.shape}")
print(f"[POINT-EVAL] validated {p}: {a.shape}",flush=True)
PY
  done
fi
echo "[POINT-EVAL] output root: $OUTPUT_ROOT"
