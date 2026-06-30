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
is handled deterministically in Python rather than left to the model. See
[docs/rag-lessons.md](docs/rag-lessons.md).

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
docs/                RAG lessons and the quantization benchmark write-up
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

## Inference benchmark

The `/eval` page runs the real retrieval + generation path live and streams the
numbers as they happen — time to first token, tokens/sec, and peak VRAM — and
can compare every installed Ollama model on the same query.

An offline sweep is in [docs/quantization_benchmark.md](docs/quantization_benchmark.md).
On a 4 GB RTX A2000 Laptop GPU, the production 8B-q4_K_M model spills ~58% of its
layers to the CPU and runs at ~5.7 tok/s, while a fully-resident Qwen2.5-1.5B
reaches ~85 tok/s. The 8B is kept for answer quality, with token streaming to
hide the latency. Reproduce with:

```bash
uv run python scripts/benchmark_quantization.py
uv run python scripts/analyze_quantization.py
```

## Design reference CLI

`tools/ui-design/` is a small, self-contained CLI over CSV reference tables
(styles, palettes, font pairings, UX guidelines). See
[tools/ui-design/README.md](tools/ui-design/README.md).
