#!/usr/bin/env bash
# Sequential batch inference followed by offline official AP evaluation.
set -euo pipefail
DEPTH_EVAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# )); then
  printf 'Configure this combined launcher through environment variables; individual launchers accept Python arguments.\n' >&2
  exit 2
fi
bash "$DEPTH_EVAL_SCRIPT_DIR/inference_cva_depth_controls.sh"
bash "$DEPTH_EVAL_SCRIPT_DIR/eval_cva_depth_controls.sh"
