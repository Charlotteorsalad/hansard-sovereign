#!/usr/bin/env bash
# Run the whole app with one command: FastAPI backend + Next.js frontend.
#
#   bash scripts/dev.sh
#
# The API starts in the background (CPU-only, see serve.sh); the web dev server
# runs in the foreground. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

cleanup() {
  kill "$API_PID" 2>/dev/null || true
  # Safety net in case uv/uvicorn outlived its parent.
  pkill -f "uvicorn scripts.api:app" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "→ Starting API   http://localhost:8000"
bash scripts/serve.sh &
API_PID=$!

echo "→ Starting web   http://localhost:3000"
cd web
npm run dev
