#!/usr/bin/env bash
# Bring up the Dockerised app (backend + web), auto-fixing the WSL2 +
# Docker Desktop networking gotcha: host.docker.internal there resolves to
# Windows, not to the WSL distro actually running `ollama serve` — so the
# backend reaches an empty/no Ollama and every generation 404s, even though
# `ollama list` on the WSL side shows the models fine.
#
#   bash scripts/docker-up.sh          # same as `docker compose up --build`,
#                                       # plus the WSL2 fix when applicable
#
# Any extra args are passed through to `docker compose up`, e.g.:
#   bash scripts/docker-up.sh -d
set -euo pipefail
cd "$(dirname "$0")/.."

if grep -qi microsoft /proc/version 2>/dev/null; then
  WSL_IP=$(hostname -I | awk '{print $1}')
  echo "→ WSL2 detected — pointing the backend's OLLAMA_BASE_URL at $WSL_IP" \
       "(this IP can change on reboot; this script re-detects it every run)"
  export OLLAMA_BASE_URL="http://$WSL_IP:11434"
fi

exec docker compose up --build "$@"
