#!/usr/bin/env bash
# Stop processes listening on PORT (default 8000). Dev helper for "address already in use".
set -euo pipefail
PORT="${1:-8000}"
# Only LISTEN sockets. Plain `lsof -i :PORT` on macOS also matches clients using that port as
# remote, which can grab unrelated PIDs (and in the worst case break Docker Desktop port forwards).
pids=$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null || true)
if [[ -z "${pids}" ]]; then
  exit 0
fi
echo "Freeing port ${PORT} (stopping PIDs: ${pids})..."
# shellcheck disable=SC2086
kill ${pids} 2>/dev/null || true
sleep 1
pids=$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null || true)
if [[ -n "${pids}" ]]; then
  # shellcheck disable=SC2086
  kill -9 ${pids} 2>/dev/null || true
fi
