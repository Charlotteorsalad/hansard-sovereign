"""
Summarise results/kv_cache_benchmark.csv into a table and a grouped bar chart.

The headline metric is the loaded footprint (`ollama ps` SIZE = model + KV
cache), which is clean for both a model that fits (VRAM tracks it) and one that
spills (where raw nvidia-smi VRAM is capped and noisy). For the spilling 8B the
GPU/CPU split is annotated, since that's what compressing the KV actually moves.

    uv run python scripts/analyze_kv_cache.py
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = Path("results/kv_cache_benchmark.csv")
PNG_PATH = Path("results/kv_cache_benchmark.png")
KV_ORDER = ["f16", "q8_0", "q4_0"]
KV_COLOR = {"f16": "#c0504d", "q8_0": "#9bbb59", "q4_0": "#4f81bd"}


def size_gb(s: str) -> float:
    m = re.match(r"([\d.]+)\s*([GM])B", s)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v if m.group(2) == "G" else v / 1024


def load():
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def main():
    rows = load()
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["kv_cache_type"]] = r

    print(f"{'model':<30}{'KV':>6}{'size_GB':>9}{'saved':>8}"
          f"{'GPU/CPU':>18}{'tok/s':>8}")
    print("-" * 80)
    for model, byk in by_model.items():
        base = size_gb(byk.get("f16", {}).get("loaded_size", "0"))
        for kv in KV_ORDER:
            r = byk.get(kv)
            if not r:
                continue
            gb = size_gb(r["loaded_size"])
            saved = f"-{base - gb:.1f}" if kv != "f16" and base else ""
            tps = r["gen_tokens_per_sec"]
            tps = "—" if tps in ("0.0", "0") else tps
            print(f"{model:<30}{kv:>6}{gb:>9.1f}{saved:>8}"
                  f"{r['processor']:>18}{tps:>8}")

    models = list(by_model)
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.6))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        byk = by_model[model]
        kvs = [kv for kv in KV_ORDER if kv in byk]
        sizes = [size_gb(byk[kv]["loaded_size"]) for kv in kvs]
        bars = ax.bar(kvs, sizes, color=[KV_COLOR[k] for k in kvs])
        ax.bar_label(bars, fmt="%.1f GB", padding=3, fontsize=9)
        ax.axhline(4.0, color="gray", ls="--", lw=1)
        ax.text(len(kvs) - 0.5, 4.0, " 4 GB VRAM", va="bottom", ha="right",
                color="gray", fontsize=8)
        ax.set_ylim(0, max(sizes) * 1.28)
        ax.set_ylabel("loaded size — model + KV cache (GB)")
        short = "Qwen2.5-1.5B" if "1.5b" in model else "Llama-3.1-8B (q4_K_M)"
        ctx = byk[kvs[0]]["num_ctx"]
        ax.set_title(f"{short}\ncontext {ctx}", fontsize=10)
        # GPU/CPU split below each bar — the thing compression moves when spilling.
        for i, kv in enumerate(kvs):
            ax.annotate(byk[kv]["processor"], (i, 0),
                        xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(0, -34),
                        ha="center", fontsize=7.5, color="#555",
                        annotation_clip=False)

    fig.suptitle("KV cache compression (flash attention) on a 4 GB GPU",
                 fontsize=12)
    fig.subplots_adjust(top=0.86, bottom=0.18, wspace=0.3)
    fig.savefig(PNG_PATH, dpi=130)
    print(f"\nWrote chart -> {PNG_PATH}")


if __name__ == "__main__":
    main()
