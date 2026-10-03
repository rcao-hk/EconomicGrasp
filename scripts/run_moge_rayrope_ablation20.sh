#!/usr/bin/env bash
# Sequential matched runs on one GPU pool; can run alongside P1 on disjoint GPUs.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
SUITE_ROOT=${SUITE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20}
VARIANTS=${VARIANTS:-baseline,moge,rayrope,moge_rayrope}
GPUS=${GPUS:-0,1,2}
PHASES=${PHASES:-train,infer,eval}
IFS=, read -ra VS <<< "$VARIANTS"
# Component flags must come from the named experiment, not inherited settings.
for key in USE_MOGE USE_RAYROPE RAY_ENCODING UNCERTAINTY TRAIN_ROOT TEST_ROOT; do
  if [[ -n "${!key:-}" ]]; then echo "Unset $key when using the named ablation suite." >&2; exit 2; fi
done
for variant in "${VS[@]}"; do
  env VARIANT="$variant" GPUS="$GPUS" PHASES="$PHASES" WORK_ROOT="$SUITE_ROOT/$variant" \
    bash "$ROOT/scripts/run_moge_rayrope20.sh"
done
if [[ ",$PHASES," == *",eval,"* && "${INFER_MAX_FRAMES:-0}" == 0 ]]; then
  "$PYTHON_BIN" "$ROOT/compare_moge_rayrope.py" --root "$SUITE_ROOT" --variants "$VARIANTS" \
    --collision "${COLLISION:-both}" --splits "${SPLITS:-test_seen,test_similar,test_novel}"
fi
