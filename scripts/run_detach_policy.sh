#!/usr/bin/env bash
set -euo pipefail
# Uses the archived production Namespace (all original resolved script args),
# then applies only output paths, seed0, and explicit arm E/Q/C overrides.
: "${GPU_IDS:?Set exactly three free GPU IDs}"
: "${POLICY_OUT:?Set experiment output root}"
: "${POLICY_LOGS:?Set checkpoint root}"
: "${ARM:?A, B, or C}"
: "${RUN_NAME:?Set unique group/run directory}"
PYTHON=${PYTHON:-/home/robotarm/miniconda3/envs/grasp/bin/python}
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ ${#GPUS[@]} == 3 ]] || { echo 'Each arm requires exactly3 GPUs' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU_IDS" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=3 \
  detach_policy_formation.py train --output "$POLICY_OUT" --logs "$POLICY_LOGS" \
  --arm "$ARM" --name "$RUN_NAME" "$@"
