"""
Summarise results/quantization_benchmark.csv into a comparison table and a
two-panel bar chart (tokens/sec and peak VRAM) for the docs.

Run from the project root, after benchmark_quantization.py:
    uv run python scripts/analyze_quantization.py
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = Path("results/quantization_benchmark.csv")
PNG_PATH = Path("results/quantization_benchmark.png")

# Short display labels keyed by the full Ollama model name.
LABELS = {
    "llama3.1:8b-instruct-q4_K_M": "Llama-3.1-8B q4_K_M",
    "qwen2.5:1.5b": "Qwen2.5-1.5B",
}


def load_rows():
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def aggregate(rows):
    """model -> averaged metrics across all queries/runs."""
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    agg = {}
    for model, rs in by_model.items():
        agg[model] = {
            "ttft_ms": statistics.mean(float(r["ttft_ms"]) for r in rs),
            "tokens_per_sec": statistics.mean(float(r["tokens_per_sec"]) for r in rs),
            "total_time_ms": statistics.mean(float(r["total_time_ms"]) for r in rs),
            "peak_vram_mb": max(int(r["peak_vram_mb"]) for r in rs),
            "processor": rs[0]["processor"],
            "n": len(rs),
        }
    return agg


def print_table(agg):
    print(f"{'model':<32}{'tok/s':>8}{'ttft_ms':>10}"
          f"{'peak_vram':>11}{'processor':>20}")
    print("-" * 81)
    for model, m in agg.items():
        print(f"{model:<32}{m['tokens_per_sec']:>8.1f}{m['ttft_ms']:>10.0f}"
              f"{m['peak_vram_mb']:>9}MB{m['processor']:>20}")


def make_chart(agg):
    # Preserve MODELS order from the benchmark (8B first, then 1.5B).
    models = [m for m in LABELS if m in agg] or list(agg)
    labels = [LABELS.get(m, m) for m in models]
    tps = [agg[m]["tokens_per_sec"] for m in models]
    vram = [agg[m]["peak_vram_mb"] for m in models]
    colors = ["#c0504d", "#4f81bd"]  # 8B red (spilling) vs 1.5B blue (resident)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    b1 = ax1.bar(labels, tps, color=colors)
    ax1.set_title("Generation speed (tokens/sec)\nhigher is better")
    ax1.set_ylabel("tokens / sec")
    ax1.bar_label(b1, fmt="%.1f", padding=3)

    b2 = ax2.bar(labels, vram, color=colors)
    ax2.set_title("Peak GPU VRAM used (MB)\non a 4096 MB card")
    ax2.set_ylabel("MB")
    ax2.axhline(4096, color="gray", ls="--", lw=1)
    ax2.text(len(models) - 0.5, 4096, " 4 GB limit", va="bottom",
             ha="right", color="gray", fontsize=9)
    ax2.set_ylim(0, 4500)
    ax2.bar_label(b2, fmt="%d", padding=3)

    # Annotate the GPU/CPU split below each model name; it's the cause of the
    # speed gap. Extra bottom margin keeps it clear of the tick labels.
    for i, m in enumerate(models):
        ax1.annotate(agg[m]["processor"], (i, 0),
                     xycoords=("data", "axes fraction"),
                     textcoords="offset points", xytext=(0, -32),
                     ha="center", fontsize=8, color="#c0504d" if i == 0 else "#4f81bd",
                     fontweight="bold", annotation_clip=False)

    fig.suptitle("Llama-3.1-8B (q4_K_M) vs Qwen2.5-1.5B on a 4 GB GPU "
                 "(RTX A2000 Laptop)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(PNG_PATH, dpi=130)
    print(f"\nWrote chart -> {PNG_PATH}")


def main():
    rows = load_rows()
    agg = aggregate(rows)
    print_table(agg)
    make_chart(agg)


if __name__ == "__main__":
    main()
