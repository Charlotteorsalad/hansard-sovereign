"""
Long-context / KV-cache profiling: how prompt length (number of retrieved
speeches) drives prefill latency and total latency.

Run from the project root (CPU torch is fine — retrieval only; generation is
Ollama):
    CUDA_VISIBLE_DEVICES="" uv run python scripts/benchmark_context_length.py

Why qwen2.5:1.5b and not the production 8B: the 1.5B stays fully resident in the
4 GB GPU, so time-to-first-token reflects real prefill scaling instead of the
8B's CPU-offload noise (the 8B spills ~58% of its layers and its TTFT is already
~50 s). The 8B inherits the same prompt-length scaling, amplified by the spill.

We vary the number of retrieved speeches N (so the prompt grows) with a fixed
context window and a fixed output length, and read the engine's own prefill
metrics from Ollama's final response:
    prompt_eval_count     -> actual prompt tokens (the true context length)
    prompt_eval_duration  -> prefill time (building the KV cache for the prompt)
so the x-axis and the prefill cost are measured, not estimated.
"""

import csv
import json
import sqlite3
import statistics
import time
from pathlib import Path

import myhansard
import requests
from myhansard.bench import OLLAMA_GENERATE, VramSampler
from myhansard.rag import _build_prompt, _build_system, _detect_lang

DB_PATH = Path("data/hansard.db")
CHROMA_PATH = Path("data/chroma")
OUT_CSV = Path("results/context_length_benchmark.csv")

MODEL = "qwen2.5:1.5b"
N_VALUES = [3, 5, 10, 20, 30, 50]
RUNS = 3

# Fixed context window (big enough that no prompt is truncated) and fixed output
# length, so total-time differences come from prefill, not decode. Pinned seed
# for comparability.
GEN_OPTIONS = {"temperature": 0.3, "seed": 42, "num_predict": 200, "num_ctx": 12288}

QUERIES = [
    "What did members say about fuel subsidies?",
    "What was raised about the cost of living and the economy?",
]


def pool_for(query: str, collection, conn, k: int = 60) -> list:
    """A ranked pool of up to k real speeches for the query (vector order)."""
    res = myhansard.query_speeches(collection, query, n_results=k)
    ids = [m["id"] for m in res["metadatas"][0]]
    cur = conn.cursor()
    placeholders = ",".join("?" * len(ids))
    cur.execute(
        "SELECT id, speaker_raw, content, date, source_file, page FROM speeches"
        f" WHERE id IN ({placeholders})",
        ids,
    )
    by_id = {r[0]: r for r in cur.fetchall() if len(r[2].strip()) > 100}
    return [by_id[i] for i in ids if i in by_id]


def run_once(prompt: str, system: str, nonce: str) -> dict:
    """One streamed generation; returns wall-clock TTFT plus the engine's own
    prefill/decode metrics and peak VRAM.

    Ollama/llama.cpp caches the prompt's KV and reuses the longest common prefix,
    so an identical prompt on a repeat run skips prefill entirely (~10 ms). To
    measure a real *cold* prefill every run, we prepend a unique nonce, which
    changes the prefix and forces a full re-prefill. (The nonce is ~1 token, so
    it doesn't materially change the context length.)"""
    busted = f"<!--{nonce}-->\n{prompt}"
    start = time.perf_counter()
    ttft = None
    done = {}
    with VramSampler() as vram:
        with requests.post(
            OLLAMA_GENERATE,
            json={"model": MODEL, "system": system, "prompt": busted,
                  "stream": True, "options": GEN_OPTIONS},
            stream=True, timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("response") and ttft is None:
                    ttft = time.perf_counter() - start
                if obj.get("done"):
                    done = obj
        total = time.perf_counter() - start
        peak = vram.peak

    ns = 1_000_000  # ns -> ms
    prompt_tokens = done.get("prompt_eval_count", 0)
    prompt_eval_ms = done.get("prompt_eval_duration", 0) / ns
    gen_tokens = done.get("eval_count", 0)
    gen_ms = done.get("eval_duration", 0) / ns
    gen_tps = gen_tokens / (gen_ms / 1000) if gen_ms else 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "ttft_ms": round((ttft or 0) * 1000, 1),
        "prompt_eval_ms": round(prompt_eval_ms, 1),
        "total_time_ms": round(total * 1000, 1),
        "gen_tokens": gen_tokens,
        "gen_tokens_per_sec": round(gen_tps, 1),
        "peak_vram_mb": peak,
    }


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    collection = myhansard.get_collection(CHROMA_PATH)

    # Warm the model so the first timed run isn't paying the load cost.
    requests.post(
        OLLAMA_GENERATE,
        json={"model": MODEL, "prompt": "ok", "stream": False,
              "options": {"num_predict": 1, "num_ctx": 12288}},
        timeout=600,
    )

    rows = []
    for qid, q in enumerate(QUERIES):
        pool = pool_for(q, collection, conn)
        system = _build_system(_detect_lang(q))
        print(f"\n=== q{qid}: {q!r} (pool={len(pool)}) ===")
        for n in N_VALUES:
            if n > len(pool):
                print(f"  N={n}: pool too small, skipping")
                continue
            prompt = _build_prompt(q, pool[:n])
            for run in range(RUNS):
                nonce = f"{qid}-{n}-{run}-{time.perf_counter_ns()}"
                m = run_once(prompt, system, nonce)
                rows.append({"query_id": qid, "n_results": n,
                             "prompt_chars": len(prompt), "run": run, **m})
                print(
                    f"  N={n} run{run}: {m['prompt_tokens']} tok  "
                    f"prefill={m['prompt_eval_ms']}ms  total={m['total_time_ms']}ms  "
                    f"vram={m['peak_vram_mb']}MB",
                    flush=True,
                )

    fields = ["query_id", "n_results", "prompt_chars", "prompt_tokens", "run",
              "ttft_ms", "prompt_eval_ms", "total_time_ms", "gen_tokens",
              "gen_tokens_per_sec", "peak_vram_mb"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")

    # Per-N averages (over queries + runs) to stdout.
    print("\n--- mean prefill latency by N ---")
    for n in N_VALUES:
        nr = [r for r in rows if r["n_results"] == n]
        if not nr:
            continue
        print(
            f"N={n:>2}: {statistics.mean(r['prompt_tokens'] for r in nr):.0f} tok  "
            f"prefill={statistics.mean(r['prompt_eval_ms'] for r in nr):.0f}ms  "
            f"total={statistics.mean(r['total_time_ms'] for r in nr):.0f}ms  "
            f"vram={max(r['peak_vram_mb'] for r in nr)}MB"
        )


if __name__ == "__main__":
    main()
