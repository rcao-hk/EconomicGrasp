#!/usr/bin/env bash
set -u

USER_NAME="${USER:-$(id -un)}"

echo "[REP-P0-KILL] user=${USER_NAME}"
echo "[REP-P0-KILL] matching processes before stop:"
ps -u "${USER_NAME}" -o pid,ppid,rss,%mem,cmd --sort=-rss | \
  grep -E 'mine_rep_p0_geometry_sources.py|run_rep_p0_mine.sh' | grep -v grep || true

pkill -TERM -u "${USER_NAME}" -f "mine_rep_p0_geometry_sources.py" 2>/dev/null || true
pkill -TERM -u "${USER_NAME}" -f "scripts/run_rep_p0_mine.sh" 2>/dev/null || true

sleep 3

pkill -KILL -u "${USER_NAME}" -f "mine_rep_p0_geometry_sources.py" 2>/dev/null || true
pkill -KILL -u "${USER_NAME}" -f "scripts/run_rep_p0_mine.sh" 2>/dev/null || true

echo "[REP-P0-KILL] remaining matching processes:"
ps -u "${USER_NAME}" -o pid,ppid,rss,%mem,cmd --sort=-rss | \
  grep -E 'mine_rep_p0_geometry_sources.py|run_rep_p0_mine.sh' | grep -v grep || true
