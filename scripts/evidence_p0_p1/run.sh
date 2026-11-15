#!/usr/bin/env bash
# Same-branch P0 distribution matching + P1 frozen-GVAR evidence decodability.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh check|cache|p0|p1-fit|p1-audit|p1-infer|p1-eval|p1-compare}"
: "${RUN_ROOT:?Set an independent output directory; never overwrite prior rounds}"
DEVICE="${DEVICE:-cuda:0}"
FACTORY="${FACTORY:-rare_grasp.runtime:Runtime}"
mkdir -p "$RUN_ROOT"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$RUN_ROOT/evidence_cache}"
EVIDENCE_INDEX="$EVIDENCE_ROOT/manifest.json"
case "$STAGE" in
check)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m unittest discover -s tests -p 'test_evidence_decodability.py' -v
  python -m compileall -q diagnostics/evidence_decodability tests/test_evidence_decodability.py
  bash -n scripts/evidence_p0_p1/run.sh
  ;;
cache)
  : "${ACCEPTED_PLAN:?Signed 800-frame U+D plan.json}"
  : "${SUPPORT_ROOT:?P1 400-frame support_audit cache folder}"
  : "${SUPPORT_SUMMARY:?Complete P1 support summary.json}"
  : "${BOUND_CONFIG:?Verified MIX50 GVAR bound config}"
  : "${DATASET_ROOT:?GraspNet dataset root}"
  : "${GNTRANS_RGB_ROOT:?GN-Trans RGB dataset root}"
  extra=()
  if [[ "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
  if [[ "${CACHE_LIMIT:-0}" != 0 ]]; then
    [[ -n "${EVIDENCE_ROOT:-}" ]] || { echo "Smoke must use separate EVIDENCE_ROOT" >&2; exit 2; }
    extra+=(--limit "$CACHE_LIMIT")
  fi
  if [[ -n "${REFERENCE_ROOT:-}" ]]; then
    extra+=(--reference-root "$REFERENCE_ROOT" --reference-unit "${REFERENCE_UNIT:-m}")
  fi
  python -m diagnostics.evidence_decodability.cache \
    --plan "$ACCEPTED_PLAN" --support-root "$SUPPORT_ROOT" \
    --support-summary "$SUPPORT_SUMMARY" --config "$BOUND_CONFIG" \
    --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
    --factory "$FACTORY" --device "$DEVICE" \
    --output "$EVIDENCE_ROOT" "${extra[@]}"
  ;;
p0)
  : "${OLD_EXPANDED_HEADS:?Signed SHA paths: seed->{old,expanded}}"
  python -m diagnostics.evidence_decodability.p0_match \
    --cache-manifest "$EVIDENCE_INDEX" \
    --head-map "$OLD_EXPANDED_HEADS" --device "$DEVICE" \
    --min-bucket-actions "${MIN_BUCKET_ACTIONS:-20}" \
    --output "$RUN_ROOT/p0_distribution_matched"
  ;;
p1-fit)
  [[ -f "$EVIDENCE_INDEX" ]] || { echo "Run cache stage first" >&2; exit 2; }
  for SEED in ${SEEDS:-20261113 20261114 20261115}; do
    for TASK in ${TASKS:-quality failure}; do
      for MODE in ${MODES:-score64 pre_reader post_reader reader_delta role_tokens pred_geometry all}; do
        python -m diagnostics.evidence_decodability.train \
          --cache-manifest "$EVIDENCE_INDEX" \
          --mode "$MODE" --task "$TASK" \
          --seed "$SEED" --epochs "${EPOCHS:-10}" \
          --batch-frames "${BATCH_FRAMES:-4}" \
          --device "$DEVICE" \
          --output "$RUN_ROOT/p1_heads/$TASK/$SEED/$MODE"
      done
    done
  done
  ;;
p1-audit)
  : "${P1_WEIGHTS:?Fixed P1 evidence probe checkpoint}"
  LABEL="${LABEL:-quality_probe}"
  PARTITION="${PARTITION:-validation}"
  python -m diagnostics.evidence_decodability.audit \
    --cache-manifest "$EVIDENCE_INDEX" --weights "$P1_WEIGHTS" \
    --partition "$PARTITION" --device "$DEVICE" \
    --output "$RUN_ROOT/p1_audit/$LABEL/$PARTITION"
  ;;
p1-infer)
  : "${BOUND_CONFIG:?}" "${DATASET_ROOT:?}"
  LABEL="${LABEL:-native}"
  extra=()
  if [[ "$LABEL" != native ]]; then
    : "${P1_WEIGHTS:?Set fixed-epoch quality head, no oracle/reference head}"
    extra+=(--weights "$P1_WEIGHTS")
  fi
  for SPLIT in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m diagnostics.evidence_decodability.infer infer \
      --config "$BOUND_CONFIG" --factory "$FACTORY" \
      --dataset-root "$DATASET_ROOT" --gntrans-root "${GNTRANS_RGB_ROOT:-}" \
      --device "$DEVICE" --seed 20261113 \
      --split "$SPLIT" --interval "${INTERVAL:-10}" \
      --output "$RUN_ROOT/p1_inference/$LABEL/$SPLIT" "${extra[@]}"
  done
  ;;
p1-eval)
  : "${DATASET_ROOT:?}"
  LABEL="${LABEL:-native}"
  for SPLIT in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m diagnostics.evidence_decodability.infer eval \
      --inference "$RUN_ROOT/p1_inference/$LABEL/$SPLIT" \
      --dataset-root "$DATASET_ROOT" --workers "${EVAL_WORKERS:-16}"
  done
  ;;
p1-compare)
  : "${P1_RUNS_JSON:?Map seed -> native/mode -> split -> inference root}"
  python -m diagnostics.evidence_decodability.compare \
    --runs "$P1_RUNS_JSON" --bootstrap 5000 \
    --output "$RUN_ROOT/p1_comparison"
  ;;
*)
  echo "Unsupported stage: $STAGE" >&2
  exit 2
  ;;
esac
