#!/usr/bin/env bash
# Start the FastAPI server behind the chat UI and /eval. Long-lived: start it
# once and leave it running.
#
# CUDA_VISIBLE_DEVICES="" : the query embedder loads on CPU (see embedder.py) and
#   generation runs inside Ollama, so the API never needs the GPU. Hiding it
#   avoids a flaky torch CUDA init that segfaults on the 4 GB laptop GPU and
#   leaves all VRAM for Ollama. Bulk ingestion still uses the GPU separately.
# uv run : always uses the project's .venv, ignoring any active conda env.
# No --reload by default: it re-imports torch on every save (~10s). Pass it only
#   when editing backend code: bash scripts/serve.sh --reload
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=""
exec uv run uvicorn scripts.api:app --host 127.0.0.1 --port 8000 "$@"
