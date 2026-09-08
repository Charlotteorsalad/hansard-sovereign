# scripts

Drivers around the [`myhansard`](../library/myhansard/) package: build the
index, serve the API, run the dev environment, and reproduce the benchmark
studies referenced in the [project README](../README.md).

## Pipeline

| Script | Role |
| --- | --- |
| `ingest.py` | Extracts speeches from every PDF in `data/raw/` and inserts them into SQLite. Skips dates already ingested. |
| `embed.py` | Embeds all speeches currently in SQLite into ChromaDB. |
| `pipeline.py` | Runs both steps together. `--fresh` drops and rebuilds the DB + ChromaDB (needed after an extractor change); `--fresh-chroma` re-embeds only; `--ingest-only` / `--embed-only` run one half. This is what `bootstrap.sh` calls. |
| `verify_extractor.py` | Prints each extracted speech next to the raw PDF text around its `]:` anchor, so extractor changes can be visually checked against one PDF before a full re-ingest. |

```bash
uv run python -m myhansard.downloader --start 2024-03-01 --end 2024-03-08
uv run python scripts/pipeline.py --fresh
```

## Serving

| Script | Role |
| --- | --- |
| `api.py` | FastAPI app — `/query`, `/query/stream` (SSE chat), `/suggestions` (example questions generated from real corpus facts), `/benchmark/*` (live inference benchmark + context-length sweep behind the `/eval` page). |
| `serve.sh` | Runs `api.py` with `CUDA_VISIBLE_DEVICES=""` (the API never touches the GPU — generation goes through Ollama, and hiding the GPU avoids a torch CUDA init that segfaults on a 4 GB laptop GPU). Pass `--reload` while editing backend code. |
| `dev.sh` | One-command dev environment: starts `serve.sh` in the background and `npm run dev` in the foreground; Ctrl-C stops both. Requires `data/hansard.db` to already exist. |
| `bootstrap.sh` | Builds the index from scratch (download → `pipeline.py --fresh`) for a chosen date range. Slow and GPU-bound; use `fetch_data.sh` instead if you just want to run the app. |
| `fetch_data.sh` | Downloads the prebuilt SQLite + ChromaDB index from a GitHub Release — no PDF download or embedding needed. |
| `fetch_model.sh` | Downloads the QLoRA fine-tuned model (`hansard-qwen`) from a GitHub Release and imports it into Ollama, for the model selector's "Fine-tuned 1.5B" option. See [finetune/](../finetune/). |

```bash
bash scripts/fetch_data.sh   # or bootstrap.sh to build from scratch
bash scripts/dev.sh          # API on :8000, web on :3000
```

## Benchmarks

Each study is a `benchmark_*.py` (runs against a local Ollama, writes a CSV to
`results/`) paired with an `analyze_*.py` (summarises the CSV into a table +
PNG in `results/`, no GPU needed). The write-up for each lives in
`docs/*_benchmark.md`.

| Study | Benchmark | Analyze | Docs |
| --- | --- | --- | --- |
| Model footprint (8B-q4_K_M vs Qwen2.5-1.5B on a 4 GB GPU) | `benchmark_quantization.py` | `analyze_quantization.py` | [docs/quantization_benchmark.md](../docs/quantization_benchmark.md) |
| Context length / prefill scaling | `benchmark_context_length.py` | `analyze_context_length.py` | [docs/context_length_benchmark.md](../docs/context_length_benchmark.md) |
| KV-cache compression (f16/q8_0/q4_0) | `benchmark_kv_cache.py` | `analyze_kv_cache.py` | [docs/kv_cache_benchmark.md](../docs/kv_cache_benchmark.md) |
| Model-load transfer (NVMe → RAM → VRAM) | `benchmark_nvme.py` | `analyze_nvme.py` | [docs/nvme_load_benchmark.md](../docs/nvme_load_benchmark.md) |

```bash
uv run python scripts/benchmark_quantization.py
uv run python scripts/analyze_quantization.py
```

`check_bench.sh` is an ad-hoc helper for polling a long-running
`benchmark_quantization.py` job (`pgrep` + tailing `scratch_bench.log`) — a
dev convenience, not part of the reproducible pipeline above.

## Notes

- All Python scripts here assume they're run from the repo root with
  `uv run` (so `myhansard` resolves and relative paths like `data/`,
  `results/` line up).
- `pipeline.py` and `verify_extractor.py` insert `library/` onto `sys.path`
  directly instead of relying on `uv sync` having installed the package —
  harmless either way, but means they also work in a bare `python` env with
  the dependencies installed.
