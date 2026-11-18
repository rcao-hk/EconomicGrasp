#!/usr/bin/env bash
# Fresh paired RealSense GraspNet+GN-Trans mix50, RayRoPE+GVAR, height analytic.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET_ROOT:?Set absolute GraspNet root}"
: "${GNTRANS_RGB_ROOT:?Set absolute GN-Trans RGB root}"
: "${OUTPUT_ROOT:?Set a NEW, unique training output directory}"
[[ ! -e "$OUTPUT_ROOT/checkpoint.tar" ]] || { echo "Refusing existing checkpoint; explicit audited resume only" >&2; exit 2; }
[[ ! -e "$OUTPUT_ROOT/preflight_height.json" ]] || { echo "Preflight already exists: use a NEW output root" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
export POSE_DEPTH_MODE=height_analytic
export TRAIN_FRACTION=0.5 EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
export ENABLE_POINT_RAYROPE=1 ENABLE_GVAR=1
export USE_FUSE_DEPTH="${USE_FUSE_DEPTH:-1}"
export GPUS="${GPUS:-0,1,2}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_EPOCH="${MAX_EPOCH:-20}"
export LR="${LR:-0.0001}"
export SEED="${SEED:-0}"
export EVAL_START_EPOCH="${EVAL_START_EPOCH:-10}"
export CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-5}"
export RAYROPE_GROUP_CHUNK="${RAYROPE_GROUP_CHUNK:-32}"
export GVAR_ACTION_CHUNK="${GVAR_ACTION_CHUNK:-256}"
export HEIGHT_MIN_M="${HEIGHT_MIN_M:--0.05}"
export HEIGHT_MAX_M="${HEIGHT_MAX_M:-0.35}"
export HEIGHT_MIN_DOWNWARD_DOT="${HEIGHT_MIN_DOWNWARD_DOT:-0.1}"
python tools/preflight_height_rayrope_gvar.py \
  --dataset_root "$DATASET_ROOT" --gntrans_rgb_root "$GNTRANS_RGB_ROOT" \
  --height_min_downward_dot "$HEIGHT_MIN_DOWNWARD_DOT" \
  --check_rgb_files --output "$OUTPUT_ROOT/preflight_height.json"
python check_gntrans_mix_data.py \
  --dataset_root "$DATASET_ROOT" --gntrans_rgb_root "$GNTRANS_RGB_ROOT" \
  --camera realsense --fraction 0.5 --load_one \
  --output "$OUTPUT_ROOT/preflight_pairs.json"
python - "$OUTPUT_ROOT/preflight_pairs.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print("Pair preflight summary:", x)
PY
bash run_gntrans_mix_cva_cdf_train.sh
