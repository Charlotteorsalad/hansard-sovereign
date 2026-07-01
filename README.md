# Hansard Sovereign

A fully on-premise RAG system over Malaysian Parliament (Dewan Rakyat) Hansard
debates. It downloads the official Hansard PDFs, extracts and indexes every
speech, and answers questions about them with a locally-run LLM — no data ever
leaves the machine.

Answers are grounded in the source debates, cite the speaker and constituency,
and link back to the originating speech with `[n]` citations. Both English and
Bahasa Malaysia questions are supported.

## Features

- **End-to-end local pipeline** — download → extract → store → embed → serve,
  all offline.
- **Hybrid retrieval** — dense vectors (`BAAI/bge-m3`) combined with keyword
  search over SQLite.
- **Grounded answers with citations** — every claim is tied to a real speech via
  `[n]`, with source cards showing speaker, constituency, date and page.
- **Bilingual** — language is detected per question and the answer is written in
  the same language as the question.
- **Streaming chat UI** — a Next.js front end with conversation history, live
  token streaming, and clickable citations.
- **Live inference benchmark** — an `/eval` page that streams real TTFT,
  tokens/sec and peak VRAM straight from the local model.

## Quick start

Everything runs on your own machine. You need [`uv`](https://github.com/astral-sh/uv),
Node.js, and [Ollama](https://ollama.com).

```bash
# 1. clone + install deps
git clone https://github.com/Charlotteorsalad/hansard-sovereign.git
cd hansard-sovereign
uv sync
cd web && npm install && cd ..

# 2. pull the LLM (with Ollama running)
ollama pull llama3.1:8b-instruct-q4_K_M

# 3. grab the prebuilt index — no PDF download or embedding needed
bash scripts/fetch_data.sh

# 4. run it (API on :8000, web on :3000)
bash scripts/dev.sh
```

Open <http://localhost:3000> to chat, or <http://localhost:3000/eval> for the
live inference benchmark.

Want to build the index from source instead of downloading it? Skip step 3 and
run `bash scripts/bootstrap.sh` (downloads Hansard PDFs, extracts, and embeds —
slower, GPU-bound).

<details>
<summary><b>Maintainer: publishing the prebuilt index</b></summary>

`scripts/fetch_data.sh` downloads a public GitHub Release asset. To (re)publish
the index after rebuilding it:

```bash
# 1. pack the runtime data (raw PDFs are not needed)
tar -czf hansard-data.tar.gz -C data hansard.db chroma

# 2a. publish with the GitHub web UI (no extra tools):
#     repo → Releases → "Draft a new release"
#     → tag: data-v1   → attach hansard-data.tar.gz   → Publish

# 2b. …or with the gh CLI:
gh release create data-v1 hansard-data.tar.gz \
  -t "Prebuilt index" -n "SQLite + ChromaDB index for one-command setup"
```

The tag (`data-v1`) and asset name (`hansard-data.tar.gz`) must match
`scripts/fetch_data.sh`. Publishing a newer index under `data-v2` lets users opt
in with `DATA_TAG=data-v2 bash scripts/fetch_data.sh`.

</details>

Prefer containers? With Ollama running on the host and `data/` already built,
`docker compose up --build` brings up the whole app (see [Docker](#docker)).

## Architecture

```
parlimen.gov.my PDFs
        │  download (myhansard.downloader)
        ▼
   data/raw/*.pdf
        │  extract speeches via "]:" anchors (pdfplumber)
        ▼
   SQLite  (data/hansard.db)
        │  embed with bge-m3 (FP16, GPU)
        ▼
   ChromaDB (data/chroma)
        │
        ▼
   FastAPI  ──hybrid retrieve──> Ollama (Llama 3.1 8B) ──> grounded answer + [n]
        ▲
        │ HTTP / SSE
   Next.js chat UI  +  /eval benchmark
```

Small local models are unreliable at strict formatting, so anything that must be
exact — speaker names, citation numbering, output language, intros/conclusions —
is handled deterministically in Python rather than left to the model.

## Tech stack

| Layer        | Choice |
|--------------|--------|
| Extraction   | `pdfplumber` |
| Storage      | SQLite + ChromaDB |
| Embeddings   | `BAAI/bge-m3` via `sentence-transformers` (FP16 on CUDA) |
| Generation   | `llama3.1:8b-instruct-q4_K_M` via [Ollama](https://ollama.com) |
| Lang. detect | `lingua` |
| API          | FastAPI + Uvicorn |
| Front end    | Next.js (App Router) + Tailwind + shadcn/ui |

## Project layout

```
library/myhansard/   downloader, extractor, storage, embedder, rag, bench
scripts/             data pipeline, API server, benchmarks, dev runners
web/                 Next.js chat UI and /eval benchmark page
tools/ui-design/     standalone UI design reference CLI (CSV-backed)
```

## Prerequisites

- Python ≥ 3.10 and [`uv`](https://github.com/astral-sh/uv)
- Node.js (for the web front end)
- [Ollama](https://ollama.com) with the generation model pulled:

  ```bash
  ollama pull llama3.1:8b-instruct-q4_K_M
  ```

## Setup

Install Python dependencies:

```bash
uv sync
```

Download some Hansard PDFs (skips weekends and missing sittings):

```bash
uv run python -m myhansard.downloader --start 2024-03-01 --end 2024-03-08
```

Build the index — extract speeches into SQLite, then embed into ChromaDB:

```bash
uv run python scripts/pipeline.py --fresh
```

Install web dependencies:

```bash
cd web && npm install
```

## Running

Start everything (FastAPI on `:8000`, Next.js on `:3000`) with one command:

```bash
bash scripts/dev.sh
```

Then open <http://localhost:3000>.

To run the pieces separately:

```bash
bash scripts/serve.sh      # API only (long-lived; start once)
cd web && npm run dev      # front end only
```

## Docker

The whole app ships as two containers (backend + frontend). **Ollama is not
containerised** — it keeps the GPU and stays on the host, so there's no
NVIDIA-Container-Toolkit setup; the backend reaches it via
`host.docker.internal`. The backend itself runs CPU-only (the query embedder is
CPU; generation is Ollama), so its image needs no GPU.

```bash
ollama serve                 # on the HOST (must have the models pulled)
docker compose up --build    # backend :8000 + web :3000
```

Then open <http://localhost:3000>. The `data/` directory (sqlite + chroma) is
mounted at runtime — populate it first (see Setup), it isn't baked into the
image.

Two env vars make the same images work when hosting the pieces apart:

| Variable | Set on | Points to |
| --- | --- | --- |
| `BACKEND_URL` | frontend | the FastAPI backend (default `http://localhost:8000`) |
| `OLLAMA_BASE_URL` | backend | the Ollama server (default `http://localhost:11434`) |

> Note: `/eval`'s live VRAM and GPU/CPU-split figures read `nvidia-smi` and the
> `ollama` CLI, so they're only populated when the backend runs natively on the
> GPU host; tokens/sec and TTFT still work from inside a container.

## Inference benchmark

The `/eval` page runs the real retrieval + generation path live and streams the
numbers as they happen — time to first token, tokens/sec, and peak VRAM — and
can compare every installed Ollama model on the same query.

On a 4 GB RTX A2000 Laptop GPU, the production 8B-q4_K_M model spills ~58% of its
layers to the CPU and runs at ~5.7 tok/s, while a fully-resident Qwen2.5-1.5B
reaches ~85 tok/s. The 8B is kept for answer quality, with token streaming to
hide the latency. An offline sweep can be reproduced with:

```bash
uv run python scripts/benchmark_quantization.py
uv run python scripts/analyze_quantization.py
```

## Design reference CLI

`tools/ui-design/` is a small, self-contained CLI over CSV reference tables
(styles, palettes, font pairings, UX guidelines). See
[tools/ui-design/README.md](tools/ui-design/README.md).
