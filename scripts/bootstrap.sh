#!/usr/bin/env bash
# Build the index FROM SCRATCH: download Hansard PDFs, extract speeches, embed.
#
#   bash scripts/bootstrap.sh                 # default date range
#   bash scripts/bootstrap.sh 2024-03-01 2024-03-31
#
# Self-contained but slow — embedding ~85k speeches is GPU-bound. If you just
# want to try the app, use scripts/fetch_data.sh instead (prebuilt index).
set -euo pipefail
cd "$(dirname "$0")/.."

START="${1:-2024-03-01}"
END="${2:-2024-03-08}"

echo "→ Downloading Hansard PDFs ($START … $END)"
uv run python -m myhansard.downloader --start "$START" --end "$END"

echo "→ Extracting speeches + embedding into ChromaDB (the slow part)…"
uv run python scripts/pipeline.py --fresh

echo "✓ Index built. Now run: bash scripts/dev.sh"
