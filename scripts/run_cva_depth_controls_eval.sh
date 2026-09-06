#!/usr/bin/env bash
# Edit the settings below, then run: bash scripts/run_cva_depth_controls_eval.sh
set -euo pipefail

# ---- User settings: these values override inherited environment variables. ----
PYTHON="python"                          # Python from the activated grasp environment.
GPU_IDS="1,2"                            # CUDA device IDs; one inference process per GPU.
INFER_BATCH_SIZE=3                       # Per GPU, independent of training batch size.
INFER_NUM_WORKERS=2                      # DataLoader workers per GPU process.
EVAL_NUM_WORKERS=4                       # CPU processes for AP; total, not per GPU.

DATASET_ROOT="/data/robotarm/dataset/graspnet"
BASE_CHECKPOINT="/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar"
OUTPUT_ROOT="/data2/robotarm/result/grasp/rgbgrasp/diagnosis/cva_depth_controls"
RUN_TAG="stage1_e15_seed0"                # Actual training tag, including any suffix.
CONTROLS_DIR="$OUTPUT_ROOT/controls_${RUN_TAG}"
CHECKPOINT_NAME="checkpoint.tar"          # Same saved filename in each trained arm.
VARIANTS="base,none,foreground,anchor"     # Remove base if only the three controls are needed.
SPLITS="test_seen,test_similar,test_novel"

CAMERA="realsense"
TOPK_VIEWS=1                             # 1 or 4; identical for every variant.
FRAME_STRIDE=1                           # 1 = full benchmark; 10 = sampled API fork required.
COLLISION_THRESH=0                       # >0 uses the captured-cloud collision postprocessor.
COLLISION_VOXEL_SIZE=0.01
SEED=0
M_POINT=1024
NUM_POINT=20000
GRASPNESS_THRESHOLD=0.1
GRASPNESS_MODE="scene"
MIN_DEPTH=0.2
MAX_DEPTH=1.0
BIN_NUM=256

PREDICTION_ROOT="$OUTPUT_ROOT/ap_${RUN_TAG}_top${TOPK_VIEWS}_stride${FRAME_STRIDE}_bs${INFER_BATCH_SIZE}_mgpu"
RUN_MODE="all"                          # all | inference | ap
DRY_RUN=0                               # 1 prints commands without loading checkpoints/CUDA.
CHECK_ONLY=0                            # 1 validates predictions instead of computing AP.
FORCE_EVAL=0                            # 1 recomputes otherwise reusable AP results.
# ---- End of user settings. ----

if (( $# )); then
  printf 'Edit the user settings in this script; run it without command-line arguments.\n' >&2
  exit 2
fi
case "$RUN_MODE" in all|inference|ap) ;; *) printf 'Invalid RUN_MODE: %s\n' "$RUN_MODE" >&2; exit 2 ;; esac
if [[ -z "$GPU_IDS" && "$RUN_MODE" != ap ]]; then
  printf 'Set GPU_IDS to at least one CUDA device, e.g. "1" or "1,2".\n' >&2
  exit 2
fi

export PYTHON GPU_IDS INFER_BATCH_SIZE INFER_NUM_WORKERS EVAL_NUM_WORKERS
export DATASET_ROOT BASE_CHECKPOINT OUTPUT_ROOT RUN_TAG CONTROLS_DIR CHECKPOINT_NAME VARIANTS SPLITS
export CAMERA TOPK_VIEWS FRAME_STRIDE COLLISION_THRESH COLLISION_VOXEL_SIZE SEED
export M_POINT NUM_POINT GRASPNESS_THRESHOLD GRASPNESS_MODE MIN_DEPTH MAX_DEPTH BIN_NUM
export PREDICTION_ROOT DRY_RUN CHECK_ONLY FORCE_EVAL

DEPTH_EVAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
printf '[CONFIG] mode=%s GPUs=%s batch_per_GPU=%s variants=%s splits=%s\n' \
  "$RUN_MODE" "$GPU_IDS" "$INFER_BATCH_SIZE" "$VARIANTS" "$SPLITS"
printf '[OUTPUT] %s\n' "$PREDICTION_ROOT"
if [[ "$RUN_MODE" != ap ]]; then
  bash "$DEPTH_EVAL_SCRIPT_DIR/inference_cva_depth_controls.sh"
fi
# set -e prevents AP from starting if any inference job fails.
if [[ "$RUN_MODE" != inference ]]; then
  bash "$DEPTH_EVAL_SCRIPT_DIR/eval_cva_depth_controls.sh"
fi
