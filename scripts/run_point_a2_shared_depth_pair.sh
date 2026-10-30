#!/usr/bin/env bash
# Shared-frozen DPT A1/A2 pair; neither run trains or mutates the source.
set -euo pipefail

: "${DATASET_ROOT:?Set GraspNet root}"
: "${SHARED_DEPTH_CKPT:?Both A1 and A2 require a verified pose-aware checkpoint}"
: "${OUTPUT_ROOT:?Choose new root with a1/cdf and a2/cdf}"
if [[ ! -f "$SHARED_DEPTH_CKPT" ]]; then
  echo "Missing shared source checkpoint: $SHARED_DEPTH_CKPT" >&2; exit 1
fi
if [[ -d "$OUTPUT_ROOT" && -n "$(ls -A "$OUTPUT_ROOT")" ]]; then
  echo "Refusing to overwrite existing A1/A2 output: $OUTPUT_ROOT" >&2; exit 1
fi
TRAIN_DOMAIN="${TRAIN_DOMAIN:-mixed}"
if [[ "$TRAIN_DOMAIN" == mixed ]]; then
  : "${GNTRANS_RGB_ROOT:?Mixed A1/A2 requires GN-Trans RGB root}"
elif [[ "$TRAIN_DOMAIN" != original ]]; then
  echo "TRAIN_DOMAIN must be mixed or original" >&2; exit 2
fi
export TRAIN_DOMAIN SHARED_DEPTH_CKPT HEADS=cdf
mkdir -p "$OUTPUT_ROOT"
sha256sum "$SHARED_DEPTH_CKPT" > "$OUTPUT_ROOT/shared_depth_source_sha256.txt"
git rev-parse HEAD > "$OUTPUT_ROOT/source_git_sha.txt"

for mode in none dino_multiscale; do
  if [[ "$mode" == none ]]; then label=a1; else label=a2; fi
  echo "[A1-A2] $label: point RGB fusion=$mode"
  POINT_RGB_FUSION="$mode" \
  OUTPUT_ROOT="$OUTPUT_ROOT/$label" \
  bash scripts/run_point_depth_pose_graspnet100_train.sh
done
echo "[A1-A2] Finished. Verify identical frozen DPT states:"
echo "python tools/verify_point_a2_shared_geometry.py \\"
echo " --a1 $OUTPUT_ROOT/a1/cdf/checkpoint_epoch_019.tar \\"
echo " --a2 $OUTPUT_ROOT/a2/cdf/checkpoint_epoch_019.tar \\"
echo " --source $SHARED_DEPTH_CKPT"
