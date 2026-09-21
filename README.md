# Hansard Sovereign

A fully on-premise RAG system over Malaysian Parliament (Dewan Rakyat) Hansard
debates. It downloads the official Hansard PDFs, extracts and indexes every
speech, and answers questions about them with a locally-run LLM - no data ever
leaves the machine.

Answers are grounded in the source debates, cite the speaker and constituency,
and link back to the originating speech with `[n]` citations. Both English and
Bahasa Malaysia questions are supported.

## Table of contents

- [Demo](#demo)
  - [Chat interfaces](#chat-interfaces)
  - [Live inference benchmark Interface](#live-inference-benchmark-interface)
- [Benchmarks](#benchmarks)
  - [Live inference benchmark - Model comparison](#live-inference-benchmark---model-comparison)
  - [Live inference benchmark - Context length](#live-inference-benchmark---context-length)
- [Features](#features)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Cross-lingual behaviour (and an honest limitation)](#cross-lingual-behaviour-and-an-honest-limitation)
- [Tech stack](#tech-stack)
- [Project layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running](#running)
- [Docker](#docker)
- [Inference benchmark](#inference-benchmark)
- [Design reference CLI](#design-reference-cli)

## Demo
### Chat interfaces
#### Chat(Llama 8B) - English
<img width="1904" height="962" alt="image" src="https://github.com/user-attachments/assets/fa98a119-237d-401b-bd71-8f75bf968058" />

Production 8B model answering in English. Every claim is tied to a real
speech with an `[n]` citation, linking to a source card with speaker,
constituency, date and page.

#### Chat(Llama 8B) - Malay
<img width="1904" height="899" alt="image" src="https://github.com/user-attachments/assets/a0b4faf3-6faa-4bc6-bb30-fe9646ab4e12" />

Same production model, a Bahasa Malaysia question this time - language is
detected per question and the answer is written in the same language, no
separate Malay-specific model required.

#### Chat(Fine-tuned 1.5B)
<img width="1914" height="901" alt="image" src="https://github.com/user-attachments/assets/b9a55e88-d116-49a1-b950-b2bbc955bd9c" />

The QLoRA fine-tuned 1.5B answering the same kind of question - stays fully
resident on this 4 GB GPU and responds far faster than the 8B, trading some
answer depth for speed.

#### Compare both Llama 8B and Fine-tuned 1.5B
<img width="1910" height="900" alt="image" src="https://github.com/user-attachments/assets/4920dfbe-7f99-49c5-981a-82b5d619795b" />
<img width="1899" height="898" alt="image" src="https://github.com/user-attachments/assets/7c30f349-b5b0-4c3d-83a6-e4b916f7d781" />

Compare mode runs both models on the same question side by side, so answer
quality and generation speed can be judged directly against each other
instead of from separate runs.

### Live inference benchmark Interface
#### Model comparison UI
<img width="1909" height="901" alt="image" src="https://github.com/user-attachments/assets/a345dcd9-499b-4123-a042-166b0de67a65" />

One query, every installed model, run back-to-back - tokens/sec, TTFT and
VRAM update live as each model streams. A Stop button cancels the in-flight
run server-side without discarding whatever already finished.

#### Context length UI
<img width="1902" height="904" alt="image" src="https://github.com/user-attachments/assets/4a629419-bee2-40ad-abe0-4bdf0db8cbe3" />

N = 3 → 50 retrieved speeches run in sequence against one model, with
prefill latency plotted live as each point completes.

## Benchmarks
### Live inference benchmark - Model comparison
<img width="1910" height="906" alt="image" src="https://github.com/user-attachments/assets/f39f08d5-903b-42be-8c88-0b41da7bd017" />

Same query, every installed model, real generation (not a canned demo) - TTFT,
tokens/sec and peak VRAM read straight from Ollama's own counters.

| Model | Size | Tokens/sec | TTFT (ms) | Cold load (ms) | Total time (ms) | Peak VRAM (MB) | Processor split |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hansard-qwen:latest (fine-tuned) | 940 MB | 99.0 | 605 | 2,978 | 4,295 | 1,112 | 100% GPU |
| qwen2.5:1.5b (comparison) | 940 MB | 100.8 | 588 | 2,242 | 3,372 | 1,112 | 100% GPU |
| qwen2.5:7b-instruct (fine-tuning baseline) | 4,466 MB | 10.4 | 3,221 | 14,496 | 22,914 | 2,193 | 55% CPU / 45% GPU |
| llama3.1:8b-instruct-q4_K_M (production) | 4,693 MB | 7.9 | 3,560 | 4,481 | 16,952 | 2,247 | 58% CPU / 42% GPU |

TTFT excludes cold model load, shown separately - production keeps models
resident, so a cold load isn't what a real user experiences. The production
model is ~13x slower than the 1.5B-class models once it spills 58% of its
layers to CPU on this 4 GB GPU; kept in production anyway for answer quality.

### Live inference benchmark - Context length
#### Model 1 - llama3.1:8b-instruct-q4_K_M (4,693MB) (production to show as baseline)
<img width="1908" height="900" alt="image" src="https://github.com/user-attachments/assets/e73730bc-fda6-4353-841c-3ad52c10e8a7" />

Production model, 58% CPU / 42% GPU split. Per-token prefill cost is flat
across N (2.0 → 1.9 ms, -3%) - CPU-offload compute noise dominates and masks
any attention-scaling signal.

#### Model 2 - hansard-qwen:latest (940MB) (fine-tuned)
<img width="1912" height="909" alt="image" src="https://github.com/user-attachments/assets/698e7ab6-519b-4a98-b975-1eb24937fe70" />

Fully GPU-resident at 940 MB. Per-token prefill cost rises 0.2 → 0.3 ms
(+42%) from N=3 to N=50 - the cleanest view of attention's O(n²) term in
this sweep.

#### Model 3 - qwen2.5:1.5b (940MB)  (Comparison model)
<img width="1901" height="903" alt="image" src="https://github.com/user-attachments/assets/2e636cbd-8304-4414-92fe-559231226f27" />

Same size class as the fine-tune and also fully GPU-resident. Its curve
(+42%) nearly matches hansard-qwen's, confirming the scaling signal
reproduces across a base model and its fine-tune.

#### Model 4 - qwen2.5:7b-instruct (4,466MB) (baseline used for fine-tuning)
<img width="1906" height="905" alt="image" src="https://github.com/user-attachments/assets/ee87f54b-ba32-4978-ad57-d37dadb90a50" />

Partially offloaded (55% CPU / 45% GPU). Per-token cost only creeps up
1.6 → 1.7 ms (+3%) - CPU noise already starts dampening the signal at this
size, well before the full 8B.

#### Context length comparison

Same query, same N sweep (3 → 50 retrieved speeches), run against each model
in turn. Isolates how prefill scales with attention compute: it only shows
cleanly on a model that stays fully resident in VRAM - a CPU-spilled model's
own compute noise dominates and masks the signal.

| Model | Processor | Prefill @ N=3 (ms) | Prefill @ N=50 (ms) | Per-token cost, N=3→50 | Trend | Peak VRAM (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| hansard-qwen:latest (fine-tuned) | 100% GPU | 223 | 3,313 | 0.2 → 0.3 ms/tok | +42% - super-linear, attention term visible | 1,418 |
| qwen2.5:1.5b (comparison) | 100% GPU | 226 | 3,345 | 0.2 → 0.3 ms/tok | +42% - super-linear, attention term visible | 1,418 |
| qwen2.5:7b-instruct (fine-tuning baseline) | 55% CPU / 45% GPU | 1,806 | 19,377 | 1.6 → 1.7 ms/tok | +3% - near-linear, CPU noise dampens the signal | 2,226 |
| llama3.1:8b-instruct-q4_K_M (production) | 58% CPU / 42% GPU | 2,129 | 21,416 | 2.0 → 1.9 ms/tok | -3% - flat, CPU noise masks the signal entirely | 2,129 |

Peak VRAM is flat across every N for a given model - Ollama pre-sizes the
KV cache to the context window (`num_ctx`), not the actual prompt length, so
the cost of supporting long context is paid upfront regardless of use. Full
write-up (methodology, cache-hit and truncation pitfalls) in
[`docs/context_length_benchmark.md`](docs/context_length_benchmark.md).

## Features

- **End-to-end local pipeline** - download → extract → store → embed → serve,
  all offline.
- **Hybrid retrieval** - dense vectors (`BAAI/bge-m3`) and keyword search over
  SQLite, combined with reciprocal-rank fusion; English query terms are mapped
  to Malay so the keyword half fires on the Malay corpus too.
- **Grounded answers with citations** - every claim is tied to a real speech via
  `[n]`, with source cards showing speaker, constituency, date and page.
- **Bilingual** - language is detected per question and the answer is written in
  the same language as the question.
- **Streaming chat UI** - a Next.js front end with conversation history, live
  token streaming, and clickable citations.
- **Real example questions** - starter prompts are generated from actual
  corpus facts (topics, members, sitting dates) each new chat, not hardcoded.
- **Live inference benchmark** - an `/eval` page that streams real TTFT,
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

# 3. grab the prebuilt index - no PDF download or embedding needed
bash scripts/fetch_data.sh

# 4. optional - the QLoRA fine-tune (see finetune/), for the model selector's
#    "Fine-tuned 1.5B" option and Compare mode; the base chat works without it
bash scripts/fetch_model.sh

# 5. run it (API on :8000, web on :3000)
bash scripts/dev.sh
```

Open <http://localhost:3000> to chat, or <http://localhost:3000/eval> for the
live inference benchmark.

Want to build the index from source instead of downloading it? Skip step 3 and
run `bash scripts/bootstrap.sh` (downloads Hansard PDFs, extracts, and embeds -
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
exact - speaker names, citation numbering, output language, the opening line -
is handled deterministically in Python rather than left to the model.

Given eight retrieved speeches the model also tends to emit eight items, padding
with non-answers ("X did not give a specific opinion on…") for speeches that
don't address the question. The prompt tells it to skip those, and a narrow
post-processing filter drops any that slip through, then renumbers. That filter
deliberately ignores a bare "did not answer" / "tidak menjawab": an MP pressing
a minister for not answering is real, citable content.

A subtler failure: the model anchors on the question's own wording even when a
cited speech doesn't support it - asked about "sukan larian" (running events), a
speech that only discusses sports funding in general still got summarised with a
title like "Penganjuran Larian Sukan", inventing specificity the speech never
states. An explicit "base it strictly on the source" prompt instruction alone did
not stop this (verified over repeated runs), so it's also enforced
deterministically: for each cited line, any of the question's specific terms
that don't appear in *that line's own cited source* are stripped. Generic
domain words (sukan, program, tahun, …) are excluded from this check - only
narrow, topic-specific terms get scrubbed, so ordinary recurring vocabulary
isn't second-guessed line by line.

Deliberately absent: a canned closing paragraph. Templated conclusions ("these
issues reflect the urgent need to address infrastructure…") read as fluent, but
they are editorial claims with no source behind them and get appended whatever
the question was - which contradicts the point of citing every claim with `[n]`.
The cited list ends where the evidence ends. The opening line is kept, but
worded count-neutrally, since when streaming it is emitted before the body
exists and can't know whether one item or eight will follow.

These are a handful of the recurring small-model failure modes hit while
building this; the full list, with symptoms and root causes, is in
[docs/rag-lessons.md](docs/rag-lessons.md).

## Cross-lingual behaviour (and an honest limitation)

The Hansard corpus is **predominantly Bahasa Malaysia**, so the two question
languages take different paths:

- **Malay questions - the native path.** Both halves of retrieval work
  in-language (keyword `LIKE` matches the Malay text directly; dense `bge-m3`
  vectors are in-language), and the model summarises Malay → Malay with **no
  translation step**.
- **English questions - a cross-lingual path.** `bge-m3` is multilingual so the
  dense half still retrieves the right Malay speeches; the keyword half maps
  common English topic terms to their Malay equivalents (`subsidies → subsidi`,
  `education → pendidikan`) so it fires too. The two ranked lists are combined
  with **reciprocal-rank fusion**, so a speech both halves agree on rises to the
  top.

Retrieval is made deliberately language-robust this way, but one gap is
**inherent and not hidden**: at generation time an English answer must be
*translated* from the Malay source, and a small quantised model translates
imperfectly. So English answers can read slightly less faithfully than Malay
ones even when the retrieved sources are identical - the residual difference is
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

Build the index - extract speeches into SQLite, then embed into ChromaDB:

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
containerised** - it keeps the GPU and stays on the host, so there's no
NVIDIA-Container-Toolkit setup; the backend reaches it via
`host.docker.internal`. The backend itself runs CPU-only (the query embedder is
CPU; generation is Ollama), so its image needs no GPU.

```bash
ollama serve                 # on the HOST (must have the models pulled)
docker compose up --build    # backend :8000 + web :3000
```

Then open <http://localhost:3000>. The `data/` directory (sqlite + chroma) is
mounted at runtime - populate it first (see Setup), it isn't baked into the
image.

Two env vars make the same images work when hosting the pieces apart:

| Variable | Set on | Points to |
| --- | --- | --- |
| `BACKEND_URL` | frontend | the FastAPI backend (default `http://localhost:8000`) |
| `OLLAMA_BASE_URL` | backend | the Ollama server (default `http://localhost:11434`) |

> Note: `/eval`'s live peak-VRAM and GPU/CPU-split figures read Ollama's own
> `/api/ps` and the `ollama` CLI, not `nvidia-smi` directly, so both work
> from inside a container too (the backend image installs the `ollama` CLI
> for exactly this). Only the static hardware line (GPU name, total VRAM) on
> `/benchmark/info` needs `nvidia-smi`, which means it needs the backend
> running natively on the GPU host.

## Inference benchmark

The `/eval` page runs the real retrieval + generation path live and streams the
numbers as they happen - time to first token, tokens/sec, and peak VRAM - and
can compare every installed Ollama model on the same query; live single-run
figures are shown [above](#live-inference-benchmark---model-comparison) and
vary a little run to run with machine load.

For a more stable figure, an averaged offline benchmark (8 real queries × 3
runs each) puts the production 8B-q4_K_M at **5.7 tok/s** (58%/42% CPU/GPU
split) against a fully-resident Qwen2.5-1.5B at **85.3 tok/s** - full
write-up in [docs/quantization_benchmark.md](docs/quantization_benchmark.md).
The 8B is kept for answer quality despite the gap, with token streaming to
hide the latency. Reproduce it with:

```bash
uv run python scripts/benchmark_quantization.py
uv run python scripts/analyze_quantization.py
```

A second study profiles how retrieval count (prompt length) drives prefill/TTFT
and where the KV cache sits - see
[docs/context_length_benchmark.md](docs/context_length_benchmark.md):

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/benchmark_context_length.py
uv run python scripts/analyze_context_length.py
```

A third study compresses the **KV cache** (`OLLAMA_KV_CACHE_TYPE` f16/q8_0/q4_0
with flash attention) and measures the VRAM freed and the effect on the 8B's
CPU offload - see [docs/kv_cache_benchmark.md](docs/kv_cache_benchmark.md):

```bash
uv run python scripts/benchmark_kv_cache.py
uv run python scripts/analyze_kv_cache.py
```

A fourth study profiles **model-load transfer** (NVMe → system RAM → VRAM) -
cold vs page-cache-warm loads per model size - see
[docs/nvme_load_benchmark.md](docs/nvme_load_benchmark.md):

```bash
uv run python scripts/benchmark_nvme.py
uv run python scripts/analyze_nvme.py
```

## Design reference CLI

`tools/ui-design/` is a small, self-contained CLI over CSV reference tables
(styles, palettes, font pairings, UX guidelines). See
[tools/ui-design/README.md](tools/ui-design/README.md).
