#!/usr/bin/env bash
# Download the PREBUILT index (SQLite + ChromaDB) so you can run the app without
# downloading any PDFs or embedding anything yourself.
#
#   bash scripts/fetch_data.sh
#
# Pulls a public GitHub Release asset (no auth needed). Override the release tag
# with DATA_TAG=... if a newer index is published.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="Charlotteorsalad/hansard-sovereign"
TAG="${DATA_TAG:-data-v1}"
ASSET="hansard-data.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

if [ -f data/hansard.db ] && [ -d data/chroma ]; then
  echo "data/ already looks populated — nothing to do (delete it to re-fetch)."
  exit 0
fi

mkdir -p data
echo "→ Downloading prebuilt index ($TAG, ~336 MB)…"
curl -fL "$URL" -o /tmp/"$ASSET"
echo "→ Extracting into data/"
tar -xzf /tmp/"$ASSET" -C data
rm -f /tmp/"$ASSET"
echo "✓ data/hansard.db + data/chroma ready. Now run: bash scripts/dev.sh"
