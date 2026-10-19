#!/usr/bin/env bash
# 100%-trained point legacy/CDF models evaluated on identical GraspNet RealSense.
set -euo pipefail
: "${DATASET_ROOT:?Set GraspNet-1B root}"
: "${TRAIN_ROOT:?Set training root containing legacy/cdf e19 checkpoints}"
: "${OUTPUT_ROOT:?Set new, separate evaluation root}"
GPUS="${GPUS:-0,1,2}"
HEADS="${HEADS:-legacy,cdf}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-16}"
COLLISION_THRESH="${COLLISION_THRESH:-0.01}"
COLLISION_VOXEL_SIZE="${COLLISION_VOXEL_SIZE:-0.01}"
INFERENCE="${INFERENCE:-1}"
EVAL="${EVAL:-1}"
MAX_BATCHES="${MAX_BATCHES:-0}"
SEED="${SEED:-0}"
if [[ "$TEST_FRACTION" == 0.1 ]]; then
  EXPECTED=780
  EVAL_STRIDE=10
elif [[ "$TEST_FRACTION" == 1.0 ]]; then
  EXPECTED=7680
  EVAL_STRIDE=1
else
  echo "TEST_FRACTION must be 0.1 or 1.0 to use official matching evaluation slices" >&2; exit 2
fi
if [[ "$MAX_BATCHES" != 0 && "$EVAL" == 1 ]]; then
  echo "Smoke cannot run official evaluation" >&2; exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
IFS=',' read -r -a HEAD_ARRAY <<< "$HEADS"
IFS=',' read -r -a SPLIT_ARRAY <<< "$SPLITS"
if ((${#GPU_ARRAY[@]}==0)); then echo "No GPUs specified" >&2; exit 2; fi
for head in "${HEAD_ARRAY[@]}"; do
  case "$head" in legacy|cdf) ;; *) echo "Bad head=$head" >&2; exit 2;; esac
done
for split in "${SPLIT_ARRAY[@]}"; do
  case "$split" in test_seen|test_similar|test_novel) ;; *) echo "Bad split=$split" >&2; exit 2;; esac
done
if [[ "$INFERENCE" == 1 ]]; then
  # Avoid overwriting any existing output, across every job.
  for head in "${HEAD_ARRAY[@]}"; do
    ckpt="$TRAIN_ROOT/$head/checkpoint_epoch_019.tar"
    [[ -f "$ckpt" ]] || { echo "Missing $ckpt" >&2; exit 1; }
    for split in "${SPLIT_ARRAY[@]}"; do
      dst="$OUTPUT_ROOT/$head/$split"
      if [[ -d "$dst" && -n "$(ls -A "$dst")" ]]; then
        echo "Existing inference directory $dst; choose a fresh OUTPUT_ROOT" >&2
        exit 1
      fi
    done
  done
fi
run_one () {
  local head="$1" split="$2" gpu="$3"
  local dst="$OUTPUT_ROOT/$head/$split"
  mkdir -p "$dst"
  echo "[POINT100] inference head=$head split=$split gpu=$gpu fraction=$TEST_FRACTION"
  (set -C; CUDA_VISIBLE_DEVICES="$gpu" MAX_BATCHES="$MAX_BATCHES" \
    python inference_point_depth_pose_graspnet100.py \
    --point_head "$head" \
    --checkpoint_path "$TRAIN_ROOT/$head/checkpoint_epoch_019.tar" \
    --dataset_root "$DATASET_ROOT" --camera realsense \
    --test_mode "$split" --save_dir "$dst" \
    --sample_interval "$TEST_FRACTION" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --collision_thresh "$COLLISION_THRESH" \
    --collision_voxel_size "$COLLISION_VOXEL_SIZE" \
    --pose_depth_mode global_film --graspness_mode scene \
    --seed "$SEED" --min_depth 0.2 --max_depth 1.0 \
    > "$dst/inference.log" 2>&1)
}
if [[ "$INFERENCE" == 1 ]]; then
  for head in "${HEAD_ARRAY[@]}"; do
    i=0
    while ((i<${#SPLIT_ARRAY[@]})); do
      pids=()
      for ((j=0; j<${#GPU_ARRAY[@]} && i+j<${#SPLIT_ARRAY[@]}; j++)); do
        run_one "$head" "${SPLIT_ARRAY[i+j]}" "${GPU_ARRAY[j]}" &
        pids+=("$!")
      done
      failed=0
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
      [[ "$failed" == 0 ]] || { echo "Inference failed; inspect logs" >&2; exit 1; }
      i=$((i+${#GPU_ARRAY[@]}))
    done
  done
fi
if [[ "$EVAL" == 1 ]]; then
  for head in "${HEAD_ARRAY[@]}"; do
    for split in "${SPLIT_ARRAY[@]}"; do
      dst="$OUTPUT_ROOT/$head/$split"
      if [[ -e "$dst/evaluation.log" || -e "$dst/ap_${split}_realsense.npy" ]]; then
        echo "Existing official evaluation evidence; refuse overwrite: $dst" >&2; exit 1
      fi
      python - "$dst" "$split" "$head" "$EXPECTED" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); split=sys.argv[2]; head=sys.argv[3]; expected=int(sys.argv[4])
summary=json.loads((p/"inference_summary.json").read_text())
protocol=json.loads((p/"inference_protocol.json").read_text())
frames=list(p.glob("scene_*/realsense/*.npy"))
if len(frames)!=expected or summary["processed_frames"]!=expected or summary["smoke"]:
    raise SystemExit(f"Incomplete/smoke grasp dump {head}/{split}: {len(frames)}/{expected}")
if protocol["point_head"]!=head or protocol["test_mode"]!=split or protocol["dataset_count"]!=expected:
    raise SystemExit("Head/split/frame contract mismatch")
PY
      echo "[POINT100] serial official AP head=$head split=$split stride=$EVAL_STRIDE"
      (set -C; python eval.py \
        --dataset_root "$DATASET_ROOT" --dump_dir "$dst" \
        --camera realsense --split "$split" \
        --sample_interval "$EVAL_STRIDE" \
        --num_workers "$EVAL_NUM_WORKERS" > "$dst/evaluation.log" 2>&1)
      python - "$dst" "$split" "$EXPECTED" <<'PY'
import sys,numpy as np
from pathlib import Path
p=Path(sys.argv[1])/f"ap_{sys.argv[2]}_realsense.npy"
x=np.load(p,allow_pickle=False)
expected_frames=int(sys.argv[3])//30
if x.shape!=(30,expected_frames,50,6) or not np.isfinite(x).all():
    raise SystemExit(f"Invalid official AP shape/finite: {p} shape={x.shape}")
print(f"Validated {p}: {x.shape}",flush=True)
PY
    done
  done
fi
echo "[POINT100] all requested jobs finished: $OUTPUT_ROOT"
