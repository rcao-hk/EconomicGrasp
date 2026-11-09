#!/usr/bin/env bash
# Stop the scratch-MLP P2 launcher and all workers spawned by it.
# By default this includes enrichment, training, inference, and official eval.
set -u

USER_NAME="${USER_NAME:-robotarm}"
WAIT_SECONDS="${WAIT_SECONDS:-5}"
INCLUDE_OFFICIAL_EVAL="${INCLUDE_OFFICIAL_EVAL:-1}"

if [[ "$EUID" -ne 0 && "$(id -un)" != "$USER_NAME" ]]; then
    echo "[P2-KILL] current user=$(id -un), target user=$USER_NAME; permissions may be insufficient" >&2
fi

patterns=(
    'run_p2_gripper_cdf_field\.sh'
    'mine_cva_p2_gripper_field_cache\.py'
    'train_cva_p2_cdf_field\.py'
    'inference_cva_p2_cdf_field\.py'
    'validate_cva_p2_gripper_field_cache\.py'
)
if [[ "$INCLUDE_OFFICIAL_EVAL" == "1" ]]; then
    patterns+=(
        'eval_p2_gripper_cdf_field\.py'
        'GraspNetEval'
    )
fi
pattern="$(IFS='|'; echo "${patterns[*]}")"

mapfile -t roots < <(
    pgrep -u "$USER_NAME" -f "$pattern" 2>/dev/null || true
)

if ((${#roots[@]} == 0)); then
    echo "[P2-KILL] no matching processes for user=$USER_NAME"
    exit 0
fi

# Include descendants so ProcessPool/DataLoader/evaluation children do not remain.
declare -A seen=()
queue=("${roots[@]}")
all=()
while ((${#queue[@]} > 0)); do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")
    [[ -n "${seen[$pid]:-}" ]] && continue
    seen[$pid]=1
    all+=("$pid")
    mapfile -t children < <(pgrep -P "$pid" 2>/dev/null || true)
    ((${#children[@]} == 0)) || queue+=("${children[@]}")
done

# Kill descendants first, launcher last.
reverse=()
for ((i=${#all[@]}-1; i>=0; i--)); do reverse+=("${all[$i]}"); done

echo "[P2-KILL] SIGTERM -> ${reverse[*]}"
kill -TERM "${reverse[@]}" 2>/dev/null || true
sleep "$WAIT_SECONDS"

alive=()
for pid in "${reverse[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
done
if ((${#alive[@]} > 0)); then
    echo "[P2-KILL] SIGKILL -> ${alive[*]}"
    kill -KILL "${alive[@]}" 2>/dev/null || true
fi

echo "[P2-KILL] done"
