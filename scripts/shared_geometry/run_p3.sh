#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?check|smoke|diagnose}"
if [[ "$STAGE" == check ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
    tests.test_shared_geometry_p3 tests.test_shared_geometry_p2 \
    tests.test_shared_geometry_p0_p1 -v
  python -m compileall -q shared_geometry/p3_action_geometry_diagnostics.py tests/test_shared_geometry_p3.py
  bash -n scripts/shared_geometry/run_p3.sh
  exit 0
fi
if [[ "$STAGE" != diagnose && "$STAGE" != smoke ]]; then
  echo "Unsupported P3 stage: $STAGE" >&2; exit 2
fi
: "${P0P1_ROOT:?Signed existing P1 cache root is required}"
: "${P2_ROOT:?Frozen P2 shared-reader root is required}"
: "${P3_ROOT:?Separate new P3 diagnostic output root is required}"
for root in "$P0P1_ROOT" "$P2_ROOT" "${RUN_ROOT:-}"; do
  [[ -z "$root" ]] && continue
  if [[ "$(realpath -m "$P3_ROOT")" == "$(realpath -m "$root")" ]]; then
    echo "P3_ROOT must differ from all existing experiment roots" >&2; exit 2
  fi
done
[[ -f "$P0P1_ROOT/p1_cache/validation/index.json" ]] || { echo 'Signed P1 validation index not found' >&2; exit 2; }
[[ -f "$P2_ROOT/shared_reader/best.pt" ]] || { echo 'Frozen P2 shared reader best.pt not found' >&2; exit 2; }
[[ -f "$P2_ROOT/shared_fixed_weight_comparison.json" ]] || { echo 'Signed P2 comparison required for parity' >&2; exit 2; }
DEVICE="${DEVICE:-cuda:0}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
SEED="${SEED:-0}"
EXTRA=()
if [[ "$STAGE" == smoke ]]; then
  EXTRA=(--smoke --limit "${SMOKE_LIMIT:-2}")
  OUT="$P3_ROOT/p3_smoke.json"
else
  OUT="$P3_ROOT/p3_action_delta_diagnostics.json"
fi
[[ "${SKIP_REGION_SWAPS:-0}" == 1 ]] && EXTRA+=(--skip-interventions)
[[ ! -e "$OUT" ]] || { echo "P3 output exists: $OUT (no overwrite)" >&2; exit 2; }
mkdir -p "$P3_ROOT"
python -m shared_geometry.p3_action_geometry_diagnostics \
  --index "$P0P1_ROOT/p1_cache/validation/index.json" \
  --shared-weight "$P2_ROOT/shared_reader/best.pt" \
  --expected-p2 "$P2_ROOT/shared_fixed_weight_comparison.json" \
  --output "$OUT" --device "$DEVICE" --batch-size "$VAL_BATCH_SIZE" \
  --bootstrap "$BOOTSTRAP" --seed "$SEED" "${EXTRA[@]}"