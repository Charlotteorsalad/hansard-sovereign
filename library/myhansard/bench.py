"""Inference-benchmark helpers: GPU/VRAM sampling, Ollama processor split, and a
streaming live_benchmark generator. Used by the /benchmark API endpoint for the
web UI's real-time numbers. Talks to a local Ollama and a local nvidia-smi.
"""

import json
import subprocess
import threading
import time

import requests

from .rag import (
    OLLAMA_BASE_URL,
    _build_prompt,
    _build_system,
    _detect_lang,
    _retrieve,
)

OLLAMA_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS = f"{OLLAMA_BASE_URL}/api/tags"

# Fixed seed and temperature so we measure the engine, not sampling noise.
GEN_OPTIONS = {"temperature": 0.3, "seed": 42}


def gpu_used_mb() -> int | None:
    """Current GPU memory used (MiB), or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def gpu_info() -> dict:
    """GPU name, total VRAM and driver. nvidia-smi is flaky under WSL, so fall
    back to the known dev-box spec rather than failing the whole request."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        name, vram, driver = (p.strip() for p in out.split(","))
        return {"gpu": name, "vram_mb": int(vram), "driver": driver}
    except Exception:
        return {"gpu": "NVIDIA RTX A2000 Laptop GPU", "vram_mb": 4096,
                "driver": "581.95"}


def processor_split(model: str) -> str:
    """Parse `ollama ps` PROCESSOR column for the loaded model (GPU/CPU split),
    e.g. "100% GPU" or "58%/42% CPU/GPU"."""
    try:
        out = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5
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
    """Models installed in Ollama, via its HTTP API (name + size in MB)."""
    try:
        data = requests.get(OLLAMA_TAGS, timeout=5).json()
    except Exception:
        return []
    models = []
    for m in data.get("models", []):
        models.append({
            "name": m["name"],
            "size_mb": round(m.get("size", 0) / (1024 * 1024)),
        })
    return sorted(models, key=lambda m: m["size_mb"])


class VramSampler:
    """Background thread sampling GPU VRAM used; reports the peak seen."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            used = gpu_used_mb()
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


def live_benchmark(model: str, query: str, collection, conn):
    """Run one real generation against Ollama using the production retrieval and
    prompt path, yielding event dicts as it happens so the UI can update live:

        {"type": "meta",        "n_speeches", "prompt_chars", "model"}
        {"type": "first_token", "ttft_ms"}
        {"type": "token",       "text", "tokens", "tokens_per_sec", "elapsed_ms",
                                "peak_vram_mb"}
        {"type": "done",        "ttft_ms", "total_time_ms", "tokens",
                                "tokens_per_sec", "peak_vram_mb", "processor"}
    """
    speeches = _retrieve(query, collection, conn)
    prompt = _build_prompt(query, speeches)
    system = _build_system(_detect_lang(query))
    yield {"type": "meta", "n_speeches": len(speeches),
           "prompt_chars": len(prompt), "model": model}

    start = time.perf_counter()
    ttft = None
    tokens = 0
    eval_count = None

    with VramSampler() as vram:
        with requests.post(
            OLLAMA_GENERATE,
            json={"model": model, "system": system, "prompt": prompt,
                  "stream": True, "options": GEN_OPTIONS},
            stream=True, timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
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
                    eval_count = obj.get("eval_count")
        total = time.perf_counter() - start
        peak_vram = vram.peak

    processor = processor_split(model)
    tok_n = eval_count or tokens
    ttft_ms = (ttft or 0) * 1000
    decode_ms = total * 1000 - ttft_ms
    tps = tok_n / (decode_ms / 1000) if decode_ms > 0 and tok_n else 0.0
    yield {"type": "done", "ttft_ms": round(ttft_ms, 1),
           "total_time_ms": round(total * 1000, 1), "tokens": tok_n,
           "tokens_per_sec": round(tps, 1), "peak_vram_mb": peak_vram,
           "processor": processor}
