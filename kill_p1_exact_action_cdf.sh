#!/usr/bin/env bash
# Stop P1 launcher/worker processes and their descendants.
set -u

WAIT_SECONDS="${WAIT_SECONDS:-5}"
INCLUDE_OFFICIAL_EVAL="${INCLUDE_OFFICIAL_EVAL:-1}"
TARGET_USER="${TARGET_USER:-$(id -un)}"

patterns=(
    "run_p1_exact_action_cdf.sh"
    "mine_cva_exact_action_cdf_cache.py"
    "validate_cva_exact_action_cdf_cache.py"
    "train_cva_exact_action_cdf_head.py"
    "inference_cva_exact_action_cdf.py"
)
if [[ "$INCLUDE_OFFICIAL_EVAL" == "1" ]]; then
    patterns+=("eval_p1_exact_action_cdf.py")
fi

self_pid="$$"
parent_pid="$PPID"
declare -A selected=()

for pattern in "${patterns[@]}"; do
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        [[ "$pid" == "$self_pid" || "$pid" == "$parent_pid" ]] && continue
        selected["$pid"]=1
    done < <(pgrep -u "$TARGET_USER" -f "$pattern" 2>/dev/null || true)
done

# Recursively include spawned evaluator/DataLoader children.
queue=("${!selected[@]}")
index=0
while (( index < ${#queue[@]} )); do
    pid="${queue[$index]}"
    ((index++))
    while read -r child; do
        [[ -n "$child" ]] || continue
        [[ "$child" == "$self_pid" || "$child" == "$parent_pid" ]] && continue
        if [[ -z "${selected[$child]+x}" ]]; then
            selected["$child"]=1
            queue+=("$child")
        fi
    done < <(pgrep -P "$pid" 2>/dev/null || true)
done

pids=("${!selected[@]}")
if ((${#pids[@]} == 0)); then
    echo "[P1-KILL] No matching P1 process found for user ${TARGET_USER}."
    exit 0
fi

printf '[P1-KILL] Sending SIGTERM to:'
for pid in "${pids[@]}"; do printf ' %s' "$pid"; done
printf '\n'
kill -TERM "${pids[@]}" 2>/dev/null || true

for ((second=0; second<WAIT_SECONDS; second++)); do
    alive=()
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    ((${#alive[@]} == 0)) && break
    sleep 1
done

alive=()
for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
done
if ((${#alive[@]} > 0)); then
    printf '[P1-KILL] Sending SIGKILL to remaining:'
    for pid in "${alive[@]}"; do printf ' %s' "$pid"; done
    printf '\n'
    kill -KILL "${alive[@]}" 2>/dev/null || true
fi

echo "[P1-KILL] Done. Cache and completed inference files remain resumable."
