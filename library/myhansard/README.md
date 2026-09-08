# myhansard

The core Python package behind Hansard Sovereign: download → extract → store →
embed → retrieve → generate. Everything in `scripts/` and the FastAPI backend
(`scripts/api.py`) is a thin driver around this package. See the
[project README](../../README.md) for the full architecture and how this fits
with the web front end.

Packaged as `myhansard` (root `library/`, see [pyproject.toml](../../pyproject.toml)),
installed via `uv sync` from the repo root.

## Modules

| Module | Role |
| --- | --- |
| `downloader.py` | Downloads Hansard PDFs by date range from `parlimen.gov.my`, skipping weekends and already-downloaded files, with retry/backoff. |
| `extractor.py` | Splits a Hansard PDF into individual speeches by finding `"]:"` speaker anchors (e.g. `Dato' Seri Anwar Ibrahim [Bera]:`) and locates where debate content starts (the `"DOA"` opening-prayer marker). |
| `storage.py` | SQLite schema (`speeches` table) and insert/lookup helpers. |
| `embedder.py` | Embeds speeches into ChromaDB with `BAAI/bge-m3` (FP16 on CUDA) and runs vector queries at request time (CPU). |
| `rag.py` | The RAG pipeline: hybrid (vector + keyword) retrieval with reciprocal-rank fusion, bilingual prompt building, streaming/non-streaming Ollama generation, and deterministic Python post-processing (citations, speaker normalisation, non-answer padding removal, ungrounded-term scrubbing). Powers `answer()` / `stream_answer()` in `scripts/api.py`. |
| `bench.py` | GPU/VRAM sampling (`nvidia-smi`), Ollama processor-split parsing, and streaming benchmark generators (`live_benchmark`, `context_run`) reused by `rag.py`'s retrieval/prompt path. Powers the `/eval` page's live benchmark and context-length sweep. |

`rag.py` and `bench.py` share the same retrieval/prompt internals (`_retrieve`,
`_build_prompt`, `_build_system`) so benchmarks measure the exact prompt shape
production actually sends to Ollama, not a synthetic stand-in.

## Public API

Re-exported from `myhansard/__init__.py`:

| Name | From | Signature |
| --- | --- | --- |
| `downloadurl` | `downloader.download_by_date` | `(start: str, end: str \| None) -> dict` |
| `peek` | `extractor.peek` | `(pdf_path, max_pages=50) -> None` — print raw page text, for exploring a new PDF layout |
| `find_content_start` | `extractor.find_content_start` | `(pdf_path) -> int` |
| `extract_speeches` | `extractor.extract_speeches` | `(pdf_path) -> list[dict]` |
| `init_db` | `storage.init_db` | `(db_path) -> None` |
| `insert_speeches` | `storage.insert_speeches` | `(conn, speeches, date, source_file) -> None` |
| `date_exists` | `storage.date_exists` | `(conn, date) -> bool` |
| `get_collection` | `embedder.get_collection` | `(chroma_path, collection_name="hansard")` |
| `embed_speeches` | `embedder.embed_speeches` | `(conn, collection, model_name="BAAI/bge-m3") -> None` |
| `query_speeches` | `embedder.query_speeches` | `(collection, query, model_name="BAAI/bge-m3", n_results=10) -> dict` |

`rag.answer` / `rag.stream_answer` and everything in `bench.py` are imported
directly from their modules (`from myhansard.rag import answer, stream_answer`),
not re-exported from `__init__.py`.

## Usage

```python
import sqlite3
from pathlib import Path
import myhansard
from myhansard.rag import answer

DB_PATH, CHROMA_PATH = Path("data/hansard.db"), Path("data/chroma")

# 1. ingest one PDF
myhansard.init_db(DB_PATH)
conn = sqlite3.connect(DB_PATH)
speeches = myhansard.extract_speeches(Path("data/raw/04032024.pdf"))
myhansard.insert_speeches(conn, speeches, date="2024-03-04", source_file="04032024.pdf")

# 2. embed into ChromaDB
collection = myhansard.get_collection(CHROMA_PATH)
myhansard.embed_speeches(conn, collection)

# 3. ask a question (hybrid retrieval + grounded, cited generation)
result = answer("What was said about flood mitigation?", collection, conn)
print(result["answer"])   # numbered list with [n] citations
print(result["sources"])  # speaker, date, source_file, page per citation
```

For the full pipeline (many PDFs, resumable) use `scripts/pipeline.py` rather
than calling these directly — it handles `--fresh` rebuilds and skips
already-ingested dates via `date_exists`.

## Notes

- **Retrieval is hybrid, not vector-only.** `rag._retrieve` fuses dense
  `bge-m3` results with a SQLite keyword pass (English query terms are mapped
  to Malay first, since the corpus is predominantly Bahasa Malaysia) via
  reciprocal-rank fusion, capped at 8 speeches.
- **Formatting is enforced in Python, not the prompt.** Citation numbers,
  speaker/constituency formatting, and answer language are all deterministic
  post-processing in `rag.py` — small local models are unreliable at holding
  strict format instructions on their own (see [finetune/](../../finetune/)
  for the QLoRA fine-tune that teaches a 1.5B model the format natively,
  shrinking this scaffolding).
- **`bench.py` talks to a local Ollama and local `nvidia-smi`.** GPU
  figures degrade gracefully (`"GPU not detected"`) when neither is
  available, e.g. inside a container.
