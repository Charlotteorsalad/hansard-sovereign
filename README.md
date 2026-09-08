# Hansard Sovereign

A fully on-premise RAG system over Malaysian Parliament (Dewan Rakyat) Hansard
debates. It downloads the official Hansard PDFs, extracts and indexes every
speech, and answers questions about them with a locally-run LLM — no data ever
leaves the machine.

Answers are grounded in the source debates, cite the speaker and constituency,
and link back to the originating speech with `[n]` citations. Both English and
Bahasa Malaysia questions are supported.

## Demo
### Chat(Llama 8B) - English
<img width="1904" height="962" alt="image" src="https://github.com/user-attachments/assets/fa98a119-237d-401b-bd71-8f75bf968058" />

### Chat(Llama 8B) - Malay
<img width="1904" height="899" alt="image" src="https://github.com/user-attachments/assets/a0b4faf3-6faa-4bc6-bb30-fe9646ab4e12" />

### Chat(Fine-tuned 1.5B)
<img width="1914" height="901" alt="image" src="https://github.com/user-attachments/assets/b9a55e88-d116-49a1-b950-b2bbc955bd9c" />

### Compare both Llama 8B and Fine-tuned 1.5B
<img width="1910" height="900" alt="image" src="https://github.com/user-attachments/assets/4920dfbe-7f99-49c5-981a-82b5d619795b" />
<img width="1899" height="898" alt="image" src="https://github.com/user-attachments/assets/7c30f349-b5b0-4c3d-83a6-e4b916f7d781" />

### Live inference benchmark - Model comparison
to be update

### Live inference benchmark - Context length
to be update

## Features

- **End-to-end local pipeline** — download → extract → store → embed → serve,
  all offline.
- **Hybrid retrieval** — dense vectors (`BAAI/bge-m3`) and keyword search over
  SQLite, combined with reciprocal-rank fusion; English query terms are mapped
  to Malay so the keyword half fires on the Malay corpus too.
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

# 4. optional — the QLoRA fine-tune (see finetune/), for the model selector's
#    "Fine-tuned 1.5B" option and Compare mode; the base chat works without it
bash scripts/fetch_model.sh

# 5. run it (API on :8000, web on :3000)
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

<details>
<summary><b>Maintainer: publishing the fine-tuned model</b></summary>

`scripts/fetch_model.sh` downloads a public GitHub Release asset the same way.
The fine-tuned model isn't in git (large binary), so after retraining:

```bash
# 1. package the exact deployed model + its Modelfile
ollama show hansard-qwen --modelfile \
  | sed 's|^FROM .*|FROM ./hansard-qwen.q4_K_M.gguf|' > Modelfile
cp "$(ollama show hansard-qwen --modelfile | grep '^FROM ' | awk '{print $2}')" \
  ./hansard-qwen.q4_K_M.gguf   # copies the raw GGUF blob out of Ollama's store
tar -czf hansard-qwen-model.tar.gz hansard-qwen.q4_K_M.gguf Modelfile

# 2. publish (web UI or gh CLI, same pattern as the data index)
gh release create model-v1 hansard-qwen-model.tar.gz \
  -t "Fine-tuned model" -n "QLoRA fine-tune (hansard-qwen), q4_K_M GGUF"
```

The tag (`model-v1`) and asset name (`hansard-qwen-model.tar.gz`) must match
`scripts/fetch_model.sh`. A newer fine-tune under `model-v2` lets users opt in
with `MODEL_TAG=model-v2 bash scripts/fetch_model.sh`.

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
exact — speaker names, citation numbering, output language, the opening line —
is handled deterministically in Python rather than left to the model.

Given eight retrieved speeches the model also tends to emit eight items, padding
with non-answers ("X did not give a specific opinion on…") for speeches that
don't address the question. The prompt tells it to skip those, and a narrow
post-processing filter drops any that slip through, then renumbers. That filter
deliberately ignores a bare "did not answer" / "tidak menjawab": an MP pressing
a minister for not answering is real, citable content.

A subtler failure: the model anchors on the question's own wording even when a
cited speech doesn't support it — asked about "sukan larian" (running events), a
speech that only discusses sports funding in general still got summarised with a
title like "Penganjuran Larian Sukan", inventing specificity the speech never
states. An explicit "base it strictly on the source" prompt instruction alone did
not stop this (verified over repeated runs), so it's also enforced
deterministically: for each cited line, any of the question's specific terms
that don't appear in *that line's own cited source* are stripped. Generic
domain words (sukan, program, tahun, …) are excluded from this check — only
narrow, topic-specific terms get scrubbed, so ordinary recurring vocabulary
isn't second-guessed line by line.

Deliberately absent: a canned closing paragraph. Templated conclusions ("these
issues reflect the urgent need to address infrastructure…") read as fluent, but
they are editorial claims with no source behind them and get appended whatever
the question was — which contradicts the point of citing every claim with `[n]`.
The cited list ends where the evidence ends. The opening line is kept, but
worded count-neutrally, since when streaming it is emitted before the body
exists and can't know whether one item or eight will follow.

## Cross-lingual behaviour (and an honest limitation)

The Hansard corpus is **predominantly Bahasa Malaysia**, so the two question
languages take different paths:

- **Malay questions — the native path.** Both halves of retrieval work
  in-language (keyword `LIKE` matches the Malay text directly; dense `bge-m3`
  vectors are in-language), and the model summarises Malay → Malay with **no
  translation step**.
- **English questions — a cross-lingual path.** `bge-m3` is multilingual so the
  dense half still retrieves the right Malay speeches; the keyword half maps
  common English topic terms to their Malay equivalents (`subsidies → subsidi`,
  `education → pendidikan`) so it fires too. The two ranked lists are combined
  with **reciprocal-rank fusion**, so a speech both halves agree on rises to the
  top.

Retrieval is made deliberately language-robust this way, but one gap is
**inherent and not hidden**: at generation time an English answer must be
*translated* from the Malay source, and a small quantised model translates
imperfectly. So English answers can read slightly less faithfully than Malay
ones even when the retrieved sources are identical — the residual difference is
generation-side translation loss, not retrieval. Closing it further would mean a
larger/stronger generation model rather than more retrieval work.

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
finetune/            QLoRA fine-tune: self-distill data, train, evaluate
docs/                write-ups for each benchmark study, plus RAG lessons learned
results/             raw benchmark CSVs + charts (referenced from docs/)
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

A second study profiles how retrieval count (prompt length) drives prefill/TTFT
and where the KV cache sits — see
[docs/context_length_benchmark.md](docs/context_length_benchmark.md):

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/benchmark_context_length.py
uv run python scripts/analyze_context_length.py
```

A third study compresses the **KV cache** (`OLLAMA_KV_CACHE_TYPE` f16/q8_0/q4_0
with flash attention) and measures the VRAM freed and the effect on the 8B's
CPU offload — see [docs/kv_cache_benchmark.md](docs/kv_cache_benchmark.md):

```bash
uv run python scripts/benchmark_kv_cache.py
uv run python scripts/analyze_kv_cache.py
```

A fourth study profiles **model-load transfer** (NVMe → system RAM → VRAM) —
cold vs page-cache-warm loads per model size — see
[docs/nvme_load_benchmark.md](docs/nvme_load_benchmark.md):

```bash
uv run python scripts/benchmark_nvme.py
uv run python scripts/analyze_nvme.py
```

## Design reference CLI

`tools/ui-design/` is a small, self-contained CLI over CSV reference tables
(styles, palettes, font pairings, UX guidelines). See
[tools/ui-design/README.md](tools/ui-design/README.md).
