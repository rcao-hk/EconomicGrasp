#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh check|audit|label-smoke|labels|verify-labels|sweep-sparse|sweep-full|infer|eval|summarize}"
: "${STAGE1_ROOT:?Set existing immutable Interactive Stage1 result root}"
: "${FOLLOWUP_ROOT:?Set NEW, distinct Stage1 follow-up output root}"
[[ "$STAGE1_ROOT" != "$FOLLOWUP_ROOT" ]] || { echo 'Never overwrite stage1 results' >&2; exit 2; }
VAL_INDEX="$STAGE1_ROOT/validation_index.json"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS=()
for V in A B C D; do WEIGHTS+=(--weight "$V=$STAGE1_ROOT/models/$V/last.pt"); done
case "$STAGE" in
check)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_interactive_followup.py' -v
  python -m compileall -q interactive_followup tests/test_interactive_followup.py
  bash -n scripts/interactive_followup/run.sh ;;
audit)
  extra=(); [[ -z "${ACCEPTANCE_JSON:-}" ]] || extra+=(--acceptance "$ACCEPTANCE_JSON")
  python -m interactive_followup.audit --run-root "$STAGE1_ROOT" \
    --output "$FOLLOWUP_ROOT/audit.json" "${extra[@]}" ;;
label-smoke|labels)
  : "${DATASET_ROOT:?GraspNet root required}"
  extra=(); [[ "$STAGE" != label-smoke ]] || extra+=(--limit 1)
  [[ "${RESUME:-0}" != 1 ]] || extra+=(--resume)
  out="$FOLLOWUP_ROOT/full_quality"; [[ "$STAGE" != label-smoke ]] || out="$FOLLOWUP_ROOT/smoke_labels"
  python -m interactive_followup.labels label --index "$VAL_INDEX" \
    --dataset-root "$DATASET_ROOT" --output "$out" \
    --shard "${SHARD:-0}" --shards "${SHARDS:-1}" "${extra[@]}" ;;
verify-labels)
  python -m interactive_followup.labels verify --index "$VAL_INDEX" \
    --output "$FOLLOWUP_ROOT/full_quality" ;;
sweep-sparse|sweep-full)
  extra=(); [[ "${ALLOW_CODE_DRIFT:-0}" != 1 ]] || extra+=(--allow-code-drift)
  full=(); out="$FOLLOWUP_ROOT/alpha_sparse"
  if [[ "$STAGE" == sweep-full ]]; then
    full=(--full-labels-root "$FOLLOWUP_ROOT/full_quality" --select-alpha)
    out="$FOLLOWUP_ROOT/alpha_full"
  fi
  python -m interactive_followup.sweep --index "$VAL_INDEX" \
    "${WEIGHTS[@]}" "${full[@]}" --output "$out" --device "$DEVICE" \
    --alphas '0,0.05,0.1,0.2,0.4,0.7,1' --bootstrap 2000 "${extra[@]}" ;;
infer)
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}" "${REFERENCE_ROOT:?existing source MSE pre-NMS dumps}"
  extra=(); [[ "${ALLOW_CODE_DRIFT:-0}" != 1 ]] || extra+=(--allow-code-drift)
  for SPLIT in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m interactive_followup.confirm --plan "$STAGE1_ROOT/plan.json" \
      --config "$STAGE1_ROOT/source.json" \
      --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
      --locked "$FOLLOWUP_ROOT/alpha_full/summary.json" \
      --split "$SPLIT" --reference-root "$REFERENCE_ROOT" \
      --output "$FOLLOWUP_ROOT/test/$SPLIT" --device "$DEVICE" \
      "${WEIGHTS[@]}" "${extra[@]}"
  done ;;
eval)
  : "${DATASET_ROOT:?}"
  for SPLIT in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m interactive_grasp.cli eval --dataset-root "$DATASET_ROOT" \
      --inference "$FOLLOWUP_ROOT/test/$SPLIT" --workers "${EVAL_WORKERS:-16}"
  done ;;
summarize)
  args=();for SPLIT in test_seen test_similar test_novel; do args+=(--split-dir "$SPLIT=$FOLLOWUP_ROOT/test/$SPLIT"); done
  python -m interactive_followup.summary "${args[@]}" \
    --locked "$FOLLOWUP_ROOT/alpha_full/summary.json" \
    --output "$FOLLOWUP_ROOT/locked_alpha_ap.json" \
    --markdown "$FOLLOWUP_ROOT/RESULTS_CN_LOCKED_ALPHA.md" ;;
*) echo "Unknown stage $STAGE" >&2;exit 2 ;;
esac