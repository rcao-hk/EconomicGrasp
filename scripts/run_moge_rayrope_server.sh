#!/usr/bin/env bash
# Fixed server protocol; invoke from this isolated worktree with nohup.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/home/robotarm/miniconda3/envs/grasp/bin:"$PATH"
export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export SUITE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GPUS=0,1,2 BATCH_SIZE=3 EPOCHS=20 SEEDS=1024 GROUP_CHUNK=32
export PHASES=train,infer,eval COLLISION=both
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VARIANTS=baseline,moge,rayrope,moge_rayrope
mkdir -p "$SUITE_ROOT"
exec 9>"$SUITE_ROOT/.server_queue.lock"
flock -n 9
[[ ! -e "$SUITE_ROOT/queue_started_utc.txt" ]] || { echo 'Queue already started; inspect checkpoints before recovery.' >&2; exit 2; }
[[ $(df -B1 --output=avail "$SUITE_ROOT" | tail -1) -gt 12884901888 ]] || { echo 'Need at least 12 GiB free for the full suite.' >&2; exit 2; }
date -u +%FT%TZ > "$SUITE_ROOT/queue_started_utc.txt"
echo "$$" > "$SUITE_ROOT/queue.pid"
git rev-parse HEAD > "$SUITE_ROOT/source_commit.txt"
trap 'status=$?; printf "%s\n" "$status" > "$SUITE_ROOT/queue_exit_code.txt"; date -u +%FT%TZ > "$SUITE_ROOT/queue_finished_utc.txt"' EXIT
bash scripts/run_moge_rayrope_ablation20.sh
