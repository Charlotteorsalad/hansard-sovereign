"""
KV-cache compression benchmark (Ollama).

The JD asks for "KV cache management and compression". Ollama exposes exactly
that: with flash attention on, OLLAMA_KV_CACHE_TYPE stores the K/V cache as
f16 (default), q8_0, or q4_0. On a 4 GB GPU the KV cache competes with the model
weights for VRAM, so compressing it directly buys headroom for longer context or
a bigger model.

For each KV cache type we start a throwaway Ollama server on :11435 (so the main
:11434 the app uses is untouched — no sudo, no systemd changes), load a model at
a fixed context window, and record:
    - loaded SIZE (`ollama ps`, includes the KV cache) and peak VRAM (nvidia-smi)
    - GPU/CPU processor split
    - prefill (TTFT) and decode tokens/sec
so the VRAM saved and any speed/offload change from compression are measured.

    uv run python scripts/benchmark_kv_cache.py     # no torch needed
"""

import csv
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import requests

PORT = 11435
HOST = f"127.0.0.1:{PORT}"
BASE = f"http://{HOST}"
def _models_dir() -> str:
    """Where Ollama keeps its models. Respect OLLAMA_MODELS, else probe the two
    common locations (Linux systemd install vs per-user), so this isn't pinned to
    one machine."""
    if os.environ.get("OLLAMA_MODELS"):
        return os.environ["OLLAMA_MODELS"]
    for p in ("/usr/share/ollama/.ollama/models",
              os.path.expanduser("~/.ollama/models")):
        if os.path.isdir(p):
            return p
    return "/usr/share/ollama/.ollama/models"


MODELS_DIR = _models_dir()
OUT_CSV = Path("results/kv_cache_benchmark.csv")

KV_TYPES = ["f16", "q8_0", "q4_0"]
# (model, context window). Big ctx on the 1.5B makes the KV cache dominate VRAM
# so compression is dramatic; the production 8B at its serving ctx shows whether
# compressing the KV frees enough VRAM to reduce CPU offload.
CONFIGS = [
    ("qwen2.5:1.5b", 16384),
    ("llama3.1:8b-instruct-q4_K_M", 8192),
]
PROMPT = ("Summarise the main arguments for and against fuel subsidy reform, "
          "in a numbered list, citing typical economic trade-offs.")
NUM_PREDICT = 80


# --------------------------------------------------------------------------- #
# GPU / server helpers (torch-free)
# --------------------------------------------------------------------------- #
def gpu_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class VramSampler:
    def __init__(self, interval=0.2):
        self.interval, self.peak = interval, 0
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while not self._stop.is_set():
            u = gpu_used_mb()
            if u and u > self.peak:
                self.peak = u
            time.sleep(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)


def start_server(kv_type: str):
    env = {**os.environ, "OLLAMA_HOST": HOST, "OLLAMA_FLASH_ATTENTION": "1",
           "OLLAMA_KV_CACHE_TYPE": kv_type, "OLLAMA_MODELS": MODELS_DIR}
    proc = subprocess.Popen(
        ["ollama", "serve"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,  # own process group so we can kill children too
    )
    for _ in range(60):
        try:
            requests.get(f"{BASE}/api/tags", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    stop_server(proc)
    raise RuntimeError(f"ollama :{PORT} did not become ready")


def stop_server(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(2)  # let VRAM free before the next server


def ps_row(model: str):
    """(size_str, processor) from `ollama ps` for the loaded model."""
    out = subprocess.run(
        ["ollama", "ps"], env={**os.environ, "OLLAMA_HOST": HOST},
        capture_output=True, text=True, timeout=5,
    ).stdout
    stem = model.split(":")[0]
    for line in out.splitlines():
        if line.startswith(stem):
            m = re.search(r"(\d+\.?\d*\s*[KMG]B)\s+(.+?)\s{2,}", line)
            size = m.group(1) if m else "?"
            proc = "unknown"
            parts = line.split()
            for i, tok in enumerate(parts):
                if "%" in tok or tok in ("GPU", "CPU"):
                    proc = " ".join(parts[i:i + 2])
                    break
            return size, proc
    return "?", "unknown"


def measure(model: str, num_ctx: int) -> dict:
    """Load + generate once; return footprint + prefill/decode metrics.

    A discarded warmup generation loads the model and fills caches first, so the
    timed run's tokens/sec reflects steady-state decode, not one-time load cost.
    """
    requests.post(
        f"{BASE}/api/generate",
        json={"model": model, "prompt": "ok", "stream": False, "keep_alive": "5m",
              "options": {"num_ctx": num_ctx, "num_predict": 8}},
        timeout=600,
    )
    start = time.perf_counter()
    ttft = None
    done = {}
    with VramSampler() as vram:
        with requests.post(
            f"{BASE}/api/generate",
            json={"model": model, "prompt": PROMPT, "stream": True,
                  "keep_alive": "5m",
                  "options": {"num_ctx": num_ctx, "num_predict": NUM_PREDICT,
                              "temperature": 0.3, "seed": 42}},
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
        peak = vram.peak
    size, proc = ps_row(model)
    ns = 1_000_000
    gen_tokens = done.get("eval_count", 0)
    gen_ms = done.get("eval_duration", 0) / ns
    return {
        "loaded_size": size,
        "processor": proc,
        "peak_vram_mb": peak,
        "prefill_ms": round(done.get("prompt_eval_duration", 0) / ns, 1),
        "ttft_ms": round((ttft or 0) * 1000, 1),
        "gen_tokens_per_sec": round(gen_tokens / (gen_ms / 1000), 1) if gen_ms else 0.0,
    }


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for kv in KV_TYPES:
        print(f"\n=== KV cache type: {kv} (flash attention on) ===")
        proc = start_server(kv)
        try:
            for model, num_ctx in CONFIGS:
                m = measure(model, num_ctx)
                row = {"kv_cache_type": kv, "model": model, "num_ctx": num_ctx, **m}
                rows.append(row)
                print(f"  {model} @ ctx {num_ctx}: size={m['loaded_size']} "
                      f"vram={m['peak_vram_mb']}MB proc={m['processor']} "
                      f"ttft={m['ttft_ms']}ms tps={m['gen_tokens_per_sec']}",
                      flush=True)
        finally:
            stop_server(proc)

    fields = ["kv_cache_type", "model", "num_ctx", "loaded_size", "peak_vram_mb",
              "processor", "prefill_ms", "ttft_ms", "gen_tokens_per_sec"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
