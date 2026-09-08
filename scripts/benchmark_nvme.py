"""
Model-load profiling: storage/RAM -> VRAM transfer.

Every time a model is (re)loaded onto the GPU you pay to move its weights from
disk (via the OS page cache) across PCIe into VRAM. Ollama reports this as
`load_duration` in each response. This benchmark unloads a model, then times a
cold load, for models of increasing size — so the transfer cost is measured,
not guessed. It's the "optimize data transfer between NVMe, system memory and
GPUs" angle on a constrained (4 GB) GPU, and it's why keeping a model resident
(keep_alive) and avoiding model-swaps matters here.

Note: with the OS page cache warm this measures the RAM->VRAM leg (PCIe-bound);
a truly cold read would additionally be bounded by NVMe sequential read. Both
are reported per model so the size-vs-latency trend is clear either way.

    uv run python scripts/benchmark_nvme.py     # no torch needed
"""

import csv
import time
from pathlib import Path

import requests

OLLAMA = "http://localhost:11434"
OUT_CSV = Path("results/nvme_load_benchmark.csv")
RUNS = 3
MODELS = [
    "qwen2.5:1.5b",
    "hansard-qwen",
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct",
]


def size_mb(model: str) -> float:
    try:
        data = requests.get(f"{OLLAMA}/api/tags", timeout=5).json()
        for m in data.get("models", []):
            if m["name"] == model or m["name"] == f"{model}:latest":
                return round(m.get("size", 0) / (1024 * 1024), 1)
    except Exception:
        pass
    return 0.0


def unload(model: str):
    """Evict the model from VRAM so the next request is a cold load."""
    try:
        requests.post(f"{OLLAMA}/api/generate",
                      json={"model": model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass
    time.sleep(3)  # let VRAM actually free


def cold_load_ms(model: str) -> tuple[float, float]:
    """Return (load_ms, total_ms) for a cold load: load_duration is the
    storage->VRAM transfer; total also includes a 1-token generation."""
    start = time.perf_counter()
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": model, "prompt": "ok", "stream": False,
              "keep_alive": "30s", "options": {"num_predict": 1}},
        timeout=600,
    ).json()
    total_ms = (time.perf_counter() - start) * 1000
    return round(r.get("load_duration", 0) / 1_000_000, 1), round(total_ms, 1)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        mb = size_mb(model)
        if mb == 0:
            print(f"skip {model} (not installed)")
            continue
        print(f"\n=== {model}  ({mb:.0f} MB) ===")
        for run in range(RUNS):
            unload(model)
            load_ms, total_ms = cold_load_ms(model)
            gbps = (mb / 1024) / (load_ms / 1000) if load_ms else 0.0
            rows.append({"model": model, "size_mb": mb, "run": run,
                         "load_ms": load_ms, "total_ms": total_ms,
                         "throughput_gbps": round(gbps, 2)})
            print(f"  run{run}: load={load_ms}ms  ({gbps:.2f} GB/s)  total={total_ms}ms",
                  flush=True)

    fields = ["model", "size_mb", "run", "load_ms", "total_ms", "throughput_gbps"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")

    print("\n--- mean cold-load by model ---")
    import statistics
    for model in MODELS:
        mr = [r for r in rows if r["model"] == model]
        if mr:
            print(f"{model:<32} {mr[0]['size_mb']:>7.0f} MB  "
                  f"load={statistics.mean(r['load_ms'] for r in mr):.0f}ms  "
                  f"{statistics.mean(r['throughput_gbps'] for r in mr):.2f} GB/s")


if __name__ == "__main__":
    main()
