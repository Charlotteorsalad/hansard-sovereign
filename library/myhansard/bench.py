"""Inference-benchmark helpers: GPU/VRAM sampling, Ollama processor split, and a
streaming live_benchmark generator. Used by the /benchmark API endpoint for the
web UI's real-time numbers. Talks to a local Ollama and a local nvidia-smi.
"""

import json
import os
import subprocess
import threading
import time

import requests

from .embedder import query_speeches
from .rag import (
    OLLAMA_BASE_URL,
    PRODUCTION_NUM_CTX,
    _build_prompt,
    _build_system,
    _detect_lang,
    _retrieve,
)

OLLAMA_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_PS = f"{OLLAMA_BASE_URL}/api/ps"

# Fixed seed and temperature so we measure the engine, not sampling noise.
# num_predict caps how much each live run generates: TTFT is captured on the
# very first token regardless, and decode tokens/sec stabilises well within
# ~70 tokens, so a full multi-hundred-token answer adds wall-clock time (badly
# on this hardware's CPU-spilling 8B/7B) without making either metric more
# accurate — the same reasoning standard inference benchmarks (e.g. MLPerf)
# use: measure throughput over a fixed generation length, not "however long
# the model feels like talking." This does not make the run any less live —
# every token is still generated fresh, right then, against the real model.
# num_ctx MUST match production (imported, not a separate literal) — this
# page's whole premise is "these numbers are what production actually does."
# A bigger num_ctx reserves more KV-cache VRAM, which pushes more of a model's
# layers to CPU on this 4 GB GPU: with 12,288 (the context-sweep's stress-test
# value) the 8B's CPU/GPU split drifted from production's real 58/42 to
# 67/33 — a materially slower, non-representative number. CONTEXT_OPTIONS
# below keeps its own 12,288 on purpose: that sweep is a deliberate stress
# test of up to 50 speeches, not a stand-in for this app's real usage.
GEN_OPTIONS = {"temperature": 0.3, "seed": 42, "num_predict": 70,
               "num_ctx": PRODUCTION_NUM_CTX}


def gpu_used_mb(model: str | None = None) -> int | None:
    """VRAM currently used by one Ollama-loaded model (MiB), or None if it
    can't be read.

    Reads Ollama's own `/api/ps` (`size_vram`, bytes) rather than shelling out
    to nvidia-smi: it's a plain HTTP call to the same Ollama this app already
    depends on, so it works identically whether this runs natively on the GPU
    host or in a container with no GPU device access at all.

    `model` MUST be passed when this is used to measure one specific
    benchmark run: on this hardware two small models can both fit in VRAM at
    once, so a model switched away from moments ago (still inside its
    keep_alive window) can still be listed in /api/ps — summing across every
    entry would silently attribute its VRAM to whatever is being measured
    now. Passing `model=None` keeps the old "everything Ollama is holding"
    total, for callers that genuinely want that (none currently do).
    """
    try:
        r = requests.get(OLLAMA_PS, timeout=2)
        r.raise_for_status()
        models = r.json().get("models", [])
        if model is not None:
            models = [m for m in models if m.get("name") == model]
        if not models:
            return None
        total_bytes = sum(m.get("size_vram", 0) for m in models)
        return round(total_bytes / (1024 * 1024))
    except Exception:
        return None


def gpu_info() -> dict:
    """GPU name, total VRAM and driver, read from nvidia-smi. If it can't be read
    (no NVIDIA GPU, or inside a container), return honest 'not detected' values —
    never fabricate a spec. nvidia-smi is occasionally flaky under WSL, so retry
    a few times before giving up."""
    for _ in range(3):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if out:
                name, vram, driver = (p.strip() for p in out.splitlines()[0].split(","))
                return {"gpu": name, "vram_mb": int(vram), "driver": driver}
        except Exception:
            pass
    return {"gpu": "GPU not detected", "vram_mb": 0, "driver": ""}


def processor_split(model: str) -> str:
    """Parse `ollama ps` PROCESSOR column for the loaded model (GPU/CPU split),
    e.g. "100% GPU" or "58%/42% CPU/GPU".

    OLLAMA_HOST points the `ollama` CLI at the same server OLLAMA_BASE_URL
    already talks to over HTTP — required when this runs in a container next
    to a remote Ollama, since the CLI defaults to localhost otherwise.
    """
    try:
        out = subprocess.run(
            ["ollama", "ps"],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "OLLAMA_HOST": OLLAMA_BASE_URL},
        ).stdout
    except Exception:
        return "unknown"
    stem = model.split(":")[0]
    for line in out.splitlines():
        if not line.startswith(stem):
            continue
        parts = line.split()
        for i, tok in enumerate(parts):
            if "%" in tok or tok in ("GPU", "CPU"):
                return " ".join(parts[i:i + 2])
    return "unknown"


def list_models() -> list[dict]:
    """Models installed in Ollama, via its HTTP API — name, size, and the real
    parameter count/quantization Ollama reports, so the UI never has to guess
    or hard-code a model's size."""
    try:
        data = requests.get(OLLAMA_TAGS, timeout=5).json()
    except Exception:
        return []
    models = []
    for m in data.get("models", []):
        details = m.get("details", {})
        models.append({
            "name": m["name"],
            "size_mb": round(m.get("size", 0) / (1024 * 1024)),
            "parameter_size": details.get("parameter_size", ""),
            "quantization": details.get("quantization_level", ""),
        })
    return sorted(models, key=lambda m: m["size_mb"])


class VramSampler:
    """Background thread sampling one model's GPU VRAM used; reports the peak
    seen. `model` scopes the sample to that model — see gpu_used_mb()."""

    def __init__(self, model: str, interval: float = 0.2):
        self.model = model
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            used = gpu_used_mb(self.model)
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


def _watch_cancel(
    r: requests.Response, cancel_event: threading.Event | None
) -> None:
    """Force-close `r`'s connection the instant cancel_event fires.

    A plain `if cancel_event.is_set(): break` inside a `for line in
    r.iter_lines():` loop only gets checked once a line has actually arrived
    — while the loop is blocked waiting on Ollama's first byte (a slow model
    still prefilling can take many seconds), that check never runs at all, so
    cancellation was observed to have zero effect until Ollama happened to
    send something anyway. Closing the connection from a second thread
    interrupts that blocking socket read immediately by raising inside it,
    regardless of whether any data has arrived yet.

    This only helps once `r` exists — a client that disconnects while we're
    still inside `requests.post()` waiting for Ollama's response headers
    (e.g. queued behind another generation on this 4 GB card's single GPU
    slot) can't be interrupted this way, since there's no connection to close
    yet. Verified live: in that case Ollama itself doesn't notice the closed
    connection until whichever uninterruptible batch it's currently running
    (a prefill step, or the current decode step) finishes — a limit of
    llama.cpp's own cancellation granularity, not of this code.
    """
    if cancel_event is None:
        return
    def _watch():
        cancel_event.wait()
        r.close()
    threading.Thread(target=_watch, daemon=True).start()


def live_benchmark(
    model: str, query: str, collection, conn,
    cancel_event: threading.Event | None = None,
):
    """Run one real generation against Ollama using the production retrieval and
    prompt path, yielding event dicts as it happens so the UI can update live:

        {"type": "meta",        "n_speeches", "prompt_chars", "model"}
        {"type": "first_token", "ttft_ms"}
        {"type": "token",       "text", "tokens", "tokens_per_sec", "elapsed_ms",
                                "peak_vram_mb"}
        {"type": "done",        "ttft_ms", "load_ms", "prefill_ms", "total_time_ms",
                                "tokens", "tokens_per_sec", "peak_vram_mb", "processor"}

    ttft_ms is client-observed (request sent -> first token received), so on a
    cold model it includes however long Ollama spent loading weights — that's
    real time a user would wait, but it isn't "engine speed". load_ms and
    prefill_ms decompose it using Ollama's own load_duration/prompt_eval_duration
    (ns) so the two aren't confused; tokens_per_sec likewise comes from Ollama's
    own eval_count/eval_duration, not from client-side token arrival timestamps
    (which would fold in HTTP/JSON overhead on top of actual decode time).

    `cancel_event`: set by the API layer once the client has disconnected, so
    the read loop below can stop and let the `with` block close the Ollama
    connection — otherwise this keeps generating (and keeps Ollama's one GPU
    slot busy on this 4 GB card) until it finishes or times out, regardless of
    whether anyone asked it to stop.
    """
    speeches = _retrieve(query, collection, conn)
    prompt = _build_prompt(query, speeches)
    system = _build_system(_detect_lang(query))
    yield {"type": "meta", "n_speeches": len(speeches),
           "prompt_chars": len(prompt), "model": model}

    start = time.perf_counter()
    ttft = None
    tokens = 0
    done_obj: dict = {}

    with VramSampler(model) as vram:
        with requests.post(
            OLLAMA_GENERATE,
            json={"model": model, "system": system, "prompt": prompt,
                  "stream": True, "options": GEN_OPTIONS},
            stream=True, timeout=600,
        ) as r:
            r.raise_for_status()
            _watch_cancel(r, cancel_event)
            try:
                for line in r.iter_lines():
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("response", "")
                    if text:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - start
                            yield {"type": "first_token",
                                   "ttft_ms": round(ttft * 1000, 1)}
                        tokens += 1
                        decode_s = (now - start) - ttft
                        tps = tokens / decode_s if decode_s > 0 else 0.0
                        yield {"type": "token", "text": text, "tokens": tokens,
                               "tokens_per_sec": round(tps, 1),
                               "elapsed_ms": round((now - start) * 1000),
                               "peak_vram_mb": vram.peak}
                    if obj.get("done"):
                        done_obj = obj
            except requests.exceptions.RequestException:
                # Expected when _watch_cancel force-closed the connection —
                # a genuine network failure looks the same, but only when we
                # weren't the ones who closed it is that actually an error.
                if cancel_event is None or not cancel_event.is_set():
                    raise
            finally:
                if cancel_event is not None:
                    cancel_event.set()  # wakes the watcher even on normal completion
        total = time.perf_counter() - start
        peak_vram = vram.peak

    processor = processor_split(model)
    ns = 1_000_000  # ns -> ms
    load_ms = done_obj.get("load_duration", 0) / ns
    prefill_ms = done_obj.get("prompt_eval_duration", 0) / ns
    eval_count = done_obj.get("eval_count", 0)
    eval_duration_s = done_obj.get("eval_duration", 0) / 1_000_000_000
    tok_n = eval_count or tokens
    tps = eval_count / eval_duration_s if eval_duration_s > 0 else 0.0
    yield {"type": "done", "ttft_ms": round((ttft or 0) * 1000, 1),
           "load_ms": round(load_ms, 1), "prefill_ms": round(prefill_ms, 1),
           "total_time_ms": round(total * 1000, 1), "tokens": tok_n,
           "tokens_per_sec": round(tps, 1), "peak_vram_mb": peak_vram,
           "processor": processor}


# --------------------------------------------------------------------------- #
# Context-length / KV-cache profiling: one (query, N) point
# --------------------------------------------------------------------------- #
# num_ctx big enough that even N=50 (~11.5k tokens) isn't truncated;
# num_predict kept short (same reasoning as GEN_OPTIONS above) so total time
# stays dominated by prefill, not decode — that's the point of this sweep.
CONTEXT_OPTIONS = {"temperature": 0.3, "seed": 42, "num_predict": 70,
                   "num_ctx": 12288}
CONTEXT_N_VALUES = [3, 5, 10, 20, 30, 50]


def _speech_pool(query: str, collection, conn, k: int = 60) -> list:
    """Ranked pool of up to k real speeches for the query (vector order)."""
    res = query_speeches(collection, query, n_results=k)
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


def context_run(
    model: str, query: str, n_results: int, collection, conn,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Run ONE generation with the top-`n_results` speeches as context and return
    the engine's prefill metrics — the per-point measurement behind the /eval
    context-length curve.

    A unique nonce is prepended to the prompt so Ollama can't reuse a cached
    prefill (it reuses the longest common prefix), giving a true cold prefill.

    `cancel_event`: this call is otherwise unstoppable once started — the API
    layer sets this when the client has disconnected, so the read loop below
    can stop pulling from Ollama and let the `with` block close the
    connection, instead of the request running to completion (or its 600s
    timeout) regardless of whether anyone is still waiting on it.
    """
    pool = _speech_pool(query, collection, conn)
    n = min(n_results, len(pool))
    prompt = _build_prompt(query, pool[:n])
    system = _build_system(_detect_lang(query))
    busted = f"<!--{n}-{time.perf_counter_ns()}-->\n{prompt}"

    start = time.perf_counter()
    done = {}
    with VramSampler(model) as vram:
        with requests.post(
            OLLAMA_GENERATE,
            json={"model": model, "system": system, "prompt": busted,
                  "stream": True, "options": CONTEXT_OPTIONS},
            stream=True, timeout=600,
        ) as r:
            r.raise_for_status()
            _watch_cancel(r, cancel_event)
            try:
                for line in r.iter_lines():
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("done"):
                        done = obj
            except requests.exceptions.RequestException:
                if cancel_event is None or not cancel_event.is_set():
                    raise
            finally:
                if cancel_event is not None:
                    cancel_event.set()
        total_ms = (time.perf_counter() - start) * 1000
        peak_vram = vram.peak

    ns = 1_000_000  # ns -> ms
    return {
        "n_results": n,
        "prompt_tokens": done.get("prompt_eval_count", 0),
        "prefill_ms": round(done.get("prompt_eval_duration", 0) / ns, 1),
        "total_time_ms": round(total_ms, 1),
        "gen_tokens": done.get("eval_count", 0),
        "peak_vram_mb": peak_vram,
    }
