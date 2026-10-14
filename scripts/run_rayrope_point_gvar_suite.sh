#!/usr/bin/env bash
# Run fresh Point control then Point+GVAR on one GPU pool; never overlap them.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT}"
SUITE_ROOT="${SUITE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rayrope_point_gvar_mixed_v2}"
PHASES="${PHASES:-train,infer,eval}"
for variant in rayrope_point rayrope_point_gvar; do
    echo "[RPG SUITE] Starting $variant phases=$PHASES"
    VARIANT="$variant" WORK_ROOT="$SUITE_ROOT/$variant" PHASES="$PHASES" \
      bash scripts/run_rayrope_point_gvar_mixed_v2.sh
done
if [[ "$PHASES" == *eval* && "${SKIP_COMPARE:-0}" != 1 ]]; then
    "${PYTHON_BIN:-python}" compare_rayrope_gvar.py --root "$SUITE_ROOT" \
      --output "$SUITE_ROOT/comparison" \
      --bootstrap "${BOOTSTRAP:-50000}"
fi
echo "[RPG SUITE] Done: $SUITE_ROOT"
