#!/usr/bin/env bash
# Guarded P6-0/P6-1 diagnostics + P6-2 external holdout / official AP.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?check|p60|p61|p62-independent|p62-register|p62-dry-run|p62-official}"

if [[ "$STAGE" == check ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest tests.test_shared_geometry_p6 -v
  python -m compileall -q shared_geometry/p6_opportunity.py \
    shared_geometry/p6_transmission.py shared_geometry/p6_independent_eval.py \
    shared_geometry/p6_official_ap.py tests/test_shared_geometry_p6.py
  bash -n scripts/shared_geometry/run_p6.sh
  exit 0
fi
: "${P6_ROOT:?Set a NEW P6 output root}"
P6_ABS="$(realpath -m "$P6_ROOT")"
for var in RUN_ROOT P0P1_ROOT P2_ROOT P4_ROOT P5_ROOT; do
  path="${!var:-}"
  [[ -n "$path" ]] || continue
  old="$(realpath -m "$path")"
  if [[ "$P6_ABS" == "$old" || "$P6_ABS" == "$old"/* || "$old" == "$P6_ABS"/* ]]; then
    echo "P6_ROOT overlaps historical root $var=$old" >&2; exit 2
  fi
done
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
SMOKE_ARGS=()
[[ "${SMOKE:-0}" != 1 ]] || SMOKE_ARGS=(--smoke)
case "$STAGE" in
p60)
  : "${P5_ROOT:?P5 run02 root required}"
  python -m shared_geometry.p6_opportunity \
    --p5-summary "$P5_ROOT/p51_reader_comparison/summary.json" \
    --output "$P6_ABS/p60_opportunity" \
    --bootstrap "$BOOTSTRAP" --seed "$SEED" "${SMOKE_ARGS[@]}"
  ;;
p61)
  : "${P0P1_ROOT:?P1 root required}"
  : "${P4_ROOT:?P4 root required}"
  : "${P5_ROOT:?P5 run02 root required}"
  python -m shared_geometry.p6_transmission \
    --p1-index "$P0P1_ROOT/p1_cache/validation/index.json" \
    --p4-summary "$P4_ROOT/depth_evaluate/summary.json" \
    --p5-summary "$P5_ROOT/p51_reader_comparison/summary.json" \
    --output "$P6_ABS/p61_transmission" \
    --bootstrap "$BOOTSTRAP" --seed "$SEED" "${SMOKE_ARGS[@]}"
  ;;
p62-independent)
  : "${P62_MANIFEST:?Provide independent external bank manifest}"
  : "${P5_ROOT:?P5 run02 root required}"
  python -m shared_geometry.p6_independent_eval \
    --manifest "$P62_MANIFEST" \
    --predicted-weight "$P5_ROOT/p51_models/predicted/best.pt" \
    --corrected-weight "$P5_ROOT/p51_models/corrected/best.pt" \
    --output "$P6_ABS/p62_independent" --device "$DEVICE" \
    --bootstrap "$BOOTSTRAP" --seed "$SEED" "${SMOKE_ARGS[@]}"
  ;;
p62-register)
  : "${DEPLOY_CKPT:?End-to-end RGB grasp checkpoint required (NOT a P5 reader)}"
  : "${DATASET_ROOT:?GraspNet dataset root required}"
  : "${P62_MODEL_ARGS_JSON:?Frozen inference CLI JSON required}"
  if [[ "${P62_ATTEST:-0}" != 1 ]]; then
    echo "Set P62_ATTEST=1 only after freezing checkpoint/config without test-driven tuning" >&2
    exit 2
  fi
  independent=()
  [[ -z "${P62_INDEP_SUMMARY:-}" ]] || independent=(--independent-summary "$P62_INDEP_SUMMARY")
  python -m shared_geometry.p6_official_ap register \
    --registry "$P6_ABS/p62_official_registration.json" \
    --checkpoint "$DEPLOY_CKPT" --dataset-root "$DATASET_ROOT" \
    --repo-root "$PWD" --model-args-json "$P62_MODEL_ARGS_JSON" \
    "${independent[@]}" --attest-no-official-test-tuning
  ;;
p62-dry-run|p62-official)
  : "${P62_OFFICIAL_OUTPUT:?New official AP output directory required}"
  execution=()
  if [[ "$STAGE" == p62-official ]]; then
    if [[ "${P62_EXECUTE:-0}" != 1 ]]; then
      echo "Full official test requires P62_EXECUTE=1" >&2; exit 2
    fi
    execution=(--run)
  fi
  python -m shared_geometry.p6_official_ap evaluate \
    --registry "$P6_ABS/p62_official_registration.json" \
    --output "$P62_OFFICIAL_OUTPUT" \
    --num-workers "${NUM_WORKERS:-4}" --eval-workers "${EVAL_NUM_WORKERS:-10}" \
    "${execution[@]}"
  ;;
*)
  echo "Unknown P6 stage: $STAGE" >&2; exit 2;;
esac
