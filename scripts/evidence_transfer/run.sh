#!/usr/bin/env bash
# Same-branch physical-evidence transfer: strict, non-training P0/P1.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh check|validation-replay|index-labels|collect|p0|stress|make-ledger|index-external|holdout}"
: "${RUN_ROOT:?Use a NEW evidence-transfer output root, not prior accepted results}"
DEVICE="${DEVICE:-cuda:0}"
mkdir -p "$RUN_ROOT"
case "$STAGE" in
check)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests \
    -p 'test_evidence_transfer.py' -v
  python -m compileall -q diagnostics/evidence_transfer tests/test_evidence_transfer.py
  bash -n scripts/evidence_transfer/run.sh
  ;;
validation-replay)
  : "${EVIDENCE_CACHE_MANIFEST:?Accepted 400-frame signed train evidence cache}"
  : "${SCORE64_WEIGHTS:?Frozen score64 quality head from same cached model}"
  : "${ALL_QUALITY_HEAD:?Frozen all quality head}"
  : "${BOUND_CONFIG:?Frozen GVAR bound config}"
  : "${DATASET_ROOT:?GraspNet root}"
  extra=()
  if [[ "${LIMIT:-0}" != 0 ]]; then extra+=(--limit "$LIMIT"); fi
  python -m diagnostics.evidence_transfer.validation_replay \
    --cache-manifest "$EVIDENCE_CACHE_MANIFEST" \
    --weights-score64 "$SCORE64_WEIGHTS" --weights-all "$ALL_QUALITY_HEAD" \
    --config "$BOUND_CONFIG" --dataset-root "$DATASET_ROOT" \
    --gntrans-root "${GNTRANS_RGB_ROOT:-}" \
    --factory "${FACTORY:-rare_grasp.runtime:Runtime}" --device "$DEVICE" \
    --domain "${DOMAIN:-original}" --snapshot-seed 20261113 \
    --output "${VALIDATION_REPLAY_ROOT:-$RUN_ROOT/validation_replay_${DOMAIN:-original}}" \
    "${extra[@]}"
  ;;
index-labels)
  : "${EXACT_LABEL_FILES_JSON:?Actual per-frame NPZ label files with SHA256 and dataset/split keys}"
  python -m diagnostics.evidence_transfer.index labels \
    --label-files "$EXACT_LABEL_FILES_JSON" \
    --output "${EXACT_LABEL_MANIFEST_OUT:-$RUN_ROOT/exact_label_index.json}"
  ;;
collect)
  : "${ACCEPTED_RUNS_JSON:?JSON native/score64/all accepted inference roots}"
  : "${DATASET_ROOT:?GraspNet root required even when reusing exact CAD labels}"
  SPLIT="${SPLIT:-test_novel}"
  OUTPUT="${COLLECT_ROOT:-$RUN_ROOT/collected/$SPLIT}"
  extra=()
  if [[ -n "${EXACT_LABEL_MANIFEST:-}" ]]; then
    extra+=(--label-manifest "$EXACT_LABEL_MANIFEST")
  fi
  if [[ "${ALLOW_CAD_RELABEL:-0}" == 1 ]]; then extra+=(--allow-relabel); fi
  if [[ "${LIMIT:-0}" != 0 ]]; then extra+=(--limit "$LIMIT"); fi
  if [[ "${CAPTURE_GEOMETRY:-0}" == 1 ]]; then
    : "${BOUND_CONFIG:?Frozen GVAR model bound.json required for true geometry replay}"
    extra+=(--capture-pred-geometry --config "$BOUND_CONFIG" \
      --gntrans-root "${GNTRANS_RGB_ROOT:-}" --device "$DEVICE" \
      --factory "${FACTORY:-rare_grasp.runtime:Runtime}")
    if [[ -n "${REFERENCE_DEPTH_ROOT:-}" ]]; then
      extra+=(--reference-depth-root "$REFERENCE_DEPTH_ROOT" \
        --reference-unit "${REFERENCE_UNIT:-m}")
    fi
  fi
  python -m diagnostics.evidence_transfer.collect \
    --runs "$ACCEPTED_RUNS_JSON" --dataset GraspNet --split "$SPLIT" \
    --dataset-root "$DATASET_ROOT" --output "$OUTPUT" "${extra[@]}"
  ;;
p0)
  : "${ACTION_MANIFEST:?Complete common-action manifest.json for one/more datasets}"
  python -m diagnostics.evidence_transfer.residual \
    --manifest "$ACTION_MANIFEST" \
    --output "${P0_OUTPUT:-$RUN_ROOT/p0_residual}" \
    --bootstrap "${BOOTSTRAP:-3000}" --min-cell "${MIN_CELL:-30}"
  ;;
stress)
  : "${EVIDENCE_CACHE_MANIFEST:?Signed 400-frame P1 evidence-cache manifest}"
  : "${ALL_QUALITY_HEAD:?Frozen P1 all quality last.pt}"
  extra=()
  if [[ "${LIMIT:-0}" != 0 ]]; then extra+=(--limit "$LIMIT"); fi
  python -m diagnostics.evidence_transfer.intervene \
    --cache-manifest "$EVIDENCE_CACHE_MANIFEST" \
    --weights "$ALL_QUALITY_HEAD" \
    --partition "${PARTITION:-validation}" --device "$DEVICE" \
    --output "${STRESS_OUTPUT:-$RUN_ROOT/stress_${PARTITION:-validation}}" "${extra[@]}"
  ;;
make-ledger)
  : "${BASE_TRAIN_SCENES_JSON:?Complete base scene keys JSON array}"
  : "${DEVELOPMENT_SCENES_JSON:?All previously used scene keys JSON array}"
  : "${OBJECT_OVERLAP_REVIEW_JSON:?Real object/CAD overlap review JSON}"
  python -m diagnostics.evidence_transfer.index ledger \
    --base-scenes "$BASE_TRAIN_SCENES_JSON" \
    --development-scenes "$DEVELOPMENT_SCENES_JSON" \
    --object-review "$OBJECT_OVERLAP_REVIEW_JSON" \
    --output "${LEDGER_OUT:-$RUN_ROOT/base_development_ledger.json}"
  ;;
index-external)
  : "${EXTERNAL_FRAME_INDEX:?List of signed external common-action per-frame NPZ}"
  : "${EXTERNAL_DATASET_NAME:?Actual external dataset name, not GraspNet}"
  : "${EXTERNAL_DATASET_PROVENANCE:?Actual dataset protocol/version source}"
  : "${EXTERNAL_EVALUATOR:?Name of quality/label evaluator actually used}"
  : "${PRIOR_USE:?Must explicitly set never or used, not inferred}"
  python -m diagnostics.evidence_transfer.index external \
    --dataset-name "$EXTERNAL_DATASET_NAME" \
    --frame-list "$EXTERNAL_FRAME_INDEX" \
    --dataset-provenance "$EXTERNAL_DATASET_PROVENANCE" \
    --evaluator-name "$EXTERNAL_EVALUATOR" \
    --prior-use "$PRIOR_USE" \
    --output "${EXTERNAL_MANIFEST_OUT:-$RUN_ROOT/external_manifest.json}"
  ;;
holdout)
  : "${ACTION_MANIFEST:?Signed external or exploratory common-action manifest}"
  : "${BASE_LEDGER:?Signed training/development provenance ledger}"
  extra=()
  if [[ "${REQUIRE_UNTOUCHED:-1}" == 1 ]]; then extra+=(--require-untouched); fi
  python -m diagnostics.evidence_transfer.holdout \
    --manifest "$ACTION_MANIFEST" \
    --ledger "$BASE_LEDGER" \
    --output "${HOLDOUT_OUTPUT:-$RUN_ROOT/holdout}" \
    --bootstrap "${BOOTSTRAP:-5000}" "${extra[@]}"
  ;;
*)
  echo "Unknown stage $STAGE" >&2
  exit 2
  ;;
esac
