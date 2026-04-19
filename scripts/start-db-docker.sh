#!/usr/bin/env bash
# Start only the Postgres (pgvector) service.
# Exports POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD for docker-compose interpolation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/.env.development"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  exit 1
fi
if [[ ! -s "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE is empty. Save your env file in the editor (Cmd+S), then retry."
  exit 1
fi

# Pull only safe assignment lines (no full-file source: API keys can break bash).
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
grep -E '^[[:space:]]*POSTGRES_(DB|USER|PASSWORD|PORT)=' "$ENV_FILE" | sed 's/^[[:space:]]*//' | tr -d '\r' >"$TMP"

set -a
# shellcheck disable=SC1090
source "$TMP"
set +a

# Default host port when omitted (must match docker-compose ${POSTGRES_PORT:-5433} and app connection)
POSTGRES_PORT="${POSTGRES_PORT:-5433}"
export POSTGRES_PORT

if [[ -z "${POSTGRES_DB:-}" || -z "${POSTGRES_USER:-}" || -z "${POSTGRES_PASSWORD:-}" ]]; then
  echo "Error: $ENV_FILE must define POSTGRES_DB, POSTGRES_USER, and POSTGRES_PASSWORD (non-empty)."
  if [[ -s "$ENV_FILE" ]]; then
    echo "File is non-empty but those variables were not read. Check spelling and format (KEY=value, one per line)."
  fi
  echo "Tip: save the file in your editor (Cmd+S), or run: cp .env.example .env.development && edit secrets"
  exit 1
fi

cd "$ROOT"
docker compose up -d db

echo "Waiting for PostgreSQL to accept connections..."
ready=0
for _ in $(seq 1 45); do
  if docker compose exec -T db pg_isready -U "$POSTGRES_USER" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "Error: PostgreSQL in Docker did not become ready. Check: docker compose ps && docker compose logs db"
  exit 1
fi

# Volume may have been created earlier with wrong env (empty POSTGRES_DB), so the DB name may be missing.
exists=$(docker compose exec -T db psql -U "$POSTGRES_USER" -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname = '${POSTGRES_DB}'" 2>/dev/null | tr -d '[:space:]' || true)
if [[ "${exists}" != "1" ]]; then
  echo "Creating database \"${POSTGRES_DB}\" in the container..."
  docker compose exec -T db psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE \"${POSTGRES_DB}\";"
fi

echo "Postgres is ready (database: ${POSTGRES_DB}, user: ${POSTGRES_USER})."

# Docker Desktop often publishes the mapped host port a moment after the container
# accepts connections. The API connects via localhost:${POSTGRES_PORT} from the host,
# so wait until that TCP port is reachable (avoids Uvicorn reload failing on import).
echo "Waiting for host port ${POSTGRES_PORT} (Docker port mapping)..."
host_ready=0
for _ in $(seq 1 60); do
  if command -v nc >/dev/null 2>&1 && nc -z 127.0.0.1 "${POSTGRES_PORT}" 2>/dev/null; then
    host_ready=1
    break
  fi
  if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1.0); s.connect(('127.0.0.1', int('${POSTGRES_PORT}'))); s.close()" 2>/dev/null; then
      host_ready=1
      break
    fi
  fi
  sleep 0.5
done
if [[ "$host_ready" -ne 1 ]]; then
  echo "Error: nothing is accepting TCP on 127.0.0.1:${POSTGRES_PORT} from the host."
  echo "Check: docker compose ps   and   docker compose logs db"
  exit 1
fi
echo "Host can connect to Postgres on 127.0.0.1:${POSTGRES_PORT}."
