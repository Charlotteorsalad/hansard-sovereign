"""Model-footprint benchmark for a 4 GB GPU (RTX A2000 Laptop).

A full q4_K_M / q8_0 / fp16 sweep of Llama-3.1-8B won't fit here: q4_K_M (~4.9 GB)
already exceeds 4 GB and spills to CPU, and fp16/q8_0 would OOM. So instead of a
three-way comparison the hardware can't host, this measures the choice that
actually matters on 4 GB:

    8B @ q4_K_M (production model, partially CPU-offloaded)
        vs.
    qwen2.5:1.5b (small enough to stay fully in VRAM)

Each model reuses the real RAG retrieval + prompt path, and per query we record
ttft, total time, tokens, tokens/sec, peak VRAM (sampled via nvidia-smi) and the
Ollama GPU/CPU layer split. Each (model, query) runs RUNS times; every run is
written to the CSV.

Run from the project root:
    uv run python scripts/benchmark_quantization.py
"""

import csv
import json
import sqlite3
import statistics
import subprocess
import threading
import time
from pathlib import Path

import requests

import myhansard
from myhansard.rag import _build_prompt, _build_system, _detect_lang, _retrieve

DB_PATH = Path("data/hansard.db")
CHROMA_PATH = Path("data/chroma")
OUT_CSV = Path("results/quantization_benchmark.csv")

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "llama3.1:8b-instruct-q4_K_M",  # production model, spills to CPU on 4 GB
    "qwen2.5:1.5b",                 # small enough to stay fully on GPU
]

RUNS = 3  # repeats per (model, query) to smooth out single-run jitter

# Fixed seed and temperature so runs are comparable. Production rag.py randomises
# the seed; here we pin it to measure the engine, not sampling noise.
GEN_OPTIONS = {"temperature": 0.3, "seed": 42}

# Real questions over the corpus, mirroring the topics surfaced in the API.
QUERIES = [
    "What did members say about fuel subsidies?",
    "What was discussed about education funding?",
    "What did members say about healthcare and hospitals?",
    "What was raised about flood mitigation?",
    "What did members say about public transport?",
    "What was discussed about affordable housing?",
    "What did members say about corruption?",
    "What was raised about the cost of living and the economy?",
]


def _gpu_used_mb() -> int | None:
    """Current GPU memory used (MiB), or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class VramSampler:
    """Background thread sampling GPU VRAM used; reports the peak seen."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            used = _gpu_used_mb()
            if used is not None and used > self.peak:
                self.peak = used
            time.sleep(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def _processor_split(model: str) -> str:
    """Parse `ollama ps` PROCESSOR column for the loaded model (GPU/CPU split)."""
    try:
        out = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return "unknown"
    for line in out.splitlines():
        if line.startswith(model.split(":")[0]) and model.split("-")[0] in line:
            # The PROCESSOR column can contain spaces ("47%/53% CPU/GPU"), so find
            # the first token holding '%' or 'GPU'/'CPU' and take it with the next.
            parts = line.split()
            for i, tok in enumerate(parts):
                if "%" in tok or tok in ("GPU", "CPU"):
                    return " ".join(parts[i:i + 2])
    return "unknown"


def _run_once(model: str, prompt: str, system: str) -> dict:
    """One streamed generation; returns timing + token + footprint metrics."""
    start = time.perf_counter()
    ttft = None
    tokens = 0
    eval_count = None  # Ollama's own token count (authoritative if present)

    with VramSampler() as vram:
        with requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "system": system,
                "prompt": prompt,
                "stream": True,
                "options": GEN_OPTIONS,
            },
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("response"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    tokens += 1
                if obj.get("done"):
                    eval_count = obj.get("eval_count")
        total = time.perf_counter() - start
        peak_vram = vram.peak
    split = _processor_split(model)

    tok_n = eval_count if eval_count else tokens
    ttft_ms = (ttft or 0) * 1000
    total_ms = total * 1000
    gen_ms = total_ms - ttft_ms  # decode-only time
    tps = (tok_n / (gen_ms / 1000)) if gen_ms > 0 and tok_n else 0.0

    return {
        "ttft_ms": round(ttft_ms, 1),
        "total_time_ms": round(total_ms, 1),
        "tokens_generated": tok_n,
        "tokens_per_sec": round(tps, 2),
        "peak_vram_mb": peak_vram,
        "processor": split,
    }


def _warmup(model: str):
    """Load the model into memory so the first timed run isn't paying load cost."""
    requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": "ok", "stream": False,
              "options": {"num_predict": 1}},
        timeout=600,
    )


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    collection = myhansard.get_collection(CHROMA_PATH)

    # Pre-build the identical prompt each model will see, once per query.
    prompts = []
    for qid, q in enumerate(QUERIES):
        speeches = _retrieve(q, collection, conn)
        prompts.append({
            "query_id": qid,
            "query": q,
            "prompt": _build_prompt(q, speeches),
            "system": _build_system(_detect_lang(q)),
            "n_speeches": len(speeches),
        })

    rows = []
    for model in MODELS:
        print(f"\n=== {model} ===")
        print("warming up...", flush=True)
        _warmup(model)
        for p in prompts:
            for run in range(RUNS):
                m = _run_once(model, p["prompt"], p["system"])
                row = {
                    "model": model,
                    "query_id": p["query_id"],
                    "run": run,
                    "n_speeches": p["n_speeches"],
                    **m,
                }
                rows.append(row)
                print(
                    f"q{p['query_id']} run{run}: "
                    f"ttft={m['ttft_ms']}ms tps={m['tokens_per_sec']} "
                    f"vram={m['peak_vram_mb']}MB proc={m['processor']}",
                    flush=True,
                )

    fieldnames = [
        "model", "query_id", "run", "n_speeches",
        "ttft_ms", "total_time_ms", "tokens_generated",
        "tokens_per_sec", "peak_vram_mb", "processor",
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")

    # Quick per-model averages to stdout (full breakdown lives in the CSV).
    print("\n--- per-model averages ---")
    for model in MODELS:
        mr = [r for r in rows if r["model"] == model]
        print(
            f"{model}: "
            f"ttft={statistics.mean(r['ttft_ms'] for r in mr):.0f}ms  "
            f"tps={statistics.mean(r['tokens_per_sec'] for r in mr):.1f}  "
            f"peak_vram={max(r['peak_vram_mb'] for r in mr)}MB  "
            f"proc={mr[0]['processor']}"
        )


if __name__ == "__main__":
    main()
