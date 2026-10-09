#!/usr/bin/env bash
# U0--U4: same mixed data, seed/budget, CVA-CDF, and grasp->depth detach.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${DATASET_ROOT:?Set DATASET_ROOT to the GraspNet root}"
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT to the GN-Trans rendered RGB root}"

P1_VARIANT="${P1_VARIANT:-U0}"
case "$P1_VARIANT" in
  U0) export VARIANT=rayrope_point RAY_ENCODING=point; export UNCERTAINTY=fixed
      export UNCERTAINTY_LOSS=interval ;;
  U1) export VARIANT=rayrope RAY_ENCODING=expected; export UNCERTAINTY=fixed
      export UNCERTAINTY_LOSS=interval ;;
  U2) export VARIANT=rayrope_learned RAY_ENCODING=expected; export UNCERTAINTY=learned
      export UNCERTAINTY_LOSS=interval ;;
  U3) export VARIANT=rayrope_learned RAY_ENCODING=expected; export UNCERTAINTY=learned
      export UNCERTAINTY_LOSS=laplace_decoupled ;;
  U4) export VARIANT=rayrope_learned RAY_ENCODING=expected; export UNCERTAINTY=learned
      export UNCERTAINTY_LOSS=laplace_joint ;;
  *) echo "Use P1_VARIANT=U0|U1|U2|U3|U4" >&2; exit 2 ;;
esac
export USE_MOGE=0 USE_RAYROPE=1
export WORK_ROOT="${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/$P1_VARIANT}"
export TRAIN_FRACTION=0.1
export EVAL_FRACTION=0.1
export USE_FUSE_DEPTH=1
export SEED="${SEED:-0}"
export EPOCHS="${EPOCHS:-20}"
export GPUS="${GPUS:-0,1,2}"
export BATCH_SIZE="${BATCH_SIZE:-3}"
export GROUP_CHUNK="${GROUP_CHUNK:-32}"
export PHASES="${PHASES:-train,infer,eval}"

if [[ "$PHASES" == "diagnose" ]]; then
  "${PYTHON_BIN:-python}" scripts/diagnose_rayrope_paired_depth.py \
    --dataset-root "$DATASET_ROOT" --gntrans-rgb-root "$GNTRANS_RGB_ROOT" \
    --split train --fraction .1 \
    --output-prefix "$WORK_ROOT/paired_depth" \
    --max-pairs "${DIAG_MAX_PAIRS:-0}" \
    --loader-audit-pairs "${DIAG_LOADER_AUDIT_PAIRS:-8}"
  exit 0
fi
if [[ "${USE_MOGE:-0}" != 0 ]]; then
  echo "P1 experiment forbids MoGe" >&2; exit 2
fi
echo "[P1] $P1_VARIANT: $VARIANT / $UNCERTAINTY / $UNCERTAINTY_LOSS"
echo "[P1] RealSense original RGB + full TSDF depth; GN-Trans rendered RGB + full rendered depth"
exec bash scripts/run_moge_rayrope20.sh
