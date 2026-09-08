#!/usr/bin/env bash
# Download the QLoRA fine-tuned model (hansard-qwen) and import it into Ollama,
# so a fresh clone can use the "Fine-tuned 1.5B" model selector and Compare mode
# without redoing the QLoRA training pipeline (see finetune/README.md for that).
#
#   bash scripts/fetch_model.sh
#
# Pulls a public GitHub Release asset (no auth needed). Override the release tag
# with MODEL_TAG=... if a newer fine-tune is published. Requires Ollama running.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="Charlotteorsalad/hansard-sovereign"
TAG="${MODEL_TAG:-model-v1}"
ASSET="hansard-qwen-model.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

if ollama list 2>/dev/null | grep -q "^hansard-qwen"; then
  echo "hansard-qwen already installed — nothing to do (ollama rm hansard-qwen to re-fetch)."
  exit 0
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "→ Downloading fine-tuned model ($TAG, ~920 MB)…"
curl -fL "$URL" -o "$TMPDIR/$ASSET"
echo "→ Extracting"
tar -xzf "$TMPDIR/$ASSET" -C "$TMPDIR"
echo "→ Importing into Ollama as 'hansard-qwen'"
(cd "$TMPDIR" && ollama create hansard-qwen -f Modelfile)
echo "✓ hansard-qwen ready. The model selector and Compare mode will now work."
