#!/usr/bin/env bash
# Queue all P1 experiments on one disjoint GPU pool while P0 uses another pool.
# P1 families run sequentially to avoid GPU oversubscription; the queue itself
# can run concurrently with P0.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS=${GPUS:-3,5,6}
STAGES=${STAGES:-p1_1,p1_3,p1_2}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_mixed_matched/train/checkpoint_latest.pt}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
IFS=, read -ra TODO <<< "$STAGES"

for stage in "${TODO[@]}"; do
  case "$stage" in
    p1_1)
      env GPUS="$GPUS" SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT" DATASET_ROOT="$DATASET_ROOT"         bash "$ROOT/scripts/run_mgf_p1_1_online.sh"
      ;;
    p1_3)
      env GPUS="$GPUS" SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT" DATASET_ROOT="$DATASET_ROOT"         bash "$ROOT/scripts/run_mgf_p1_3_online.sh"
      ;;
    p1_2)
      env GPUS="$GPUS" SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT" DATASET_ROOT="$DATASET_ROOT"         bash "$ROOT/scripts/run_mgf_p1_2_online.sh"
      ;;
    *) echo "Unknown P1 stage: $stage" >&2; exit 2;;
  esac
done
