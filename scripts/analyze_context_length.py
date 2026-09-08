"""
Summarise results/context_length_benchmark.csv into a table and a chart showing
how prefill latency and total latency scale with prompt length (context), plus
peak VRAM.

    uv run python scripts/analyze_context_length.py
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = Path("results/context_length_benchmark.csv")
PNG_PATH = Path("results/context_length_benchmark.png")


def load():
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def aggregate(rows):
    """n_results -> averaged metrics across queries + runs."""
    by_n = defaultdict(list)
    for r in rows:
        by_n[int(r["n_results"])].append(r)
    agg = {}
    for n, rs in sorted(by_n.items()):
        agg[n] = {
            "tokens": statistics.mean(int(r["prompt_tokens"]) for r in rs),
            "prefill_ms": statistics.mean(float(r["prompt_eval_ms"]) for r in rs),
            "total_ms": statistics.mean(float(r["total_time_ms"]) for r in rs),
            "vram_mb": max(int(r["peak_vram_mb"]) for r in rs),
        }
    return agg


def print_table(agg):
    print(f"{'N':>4}{'tokens':>9}{'prefill_ms':>12}{'total_ms':>10}{'vram_mb':>9}")
    print("-" * 44)
    for n, m in agg.items():
        print(f"{n:>4}{m['tokens']:>9.0f}{m['prefill_ms']:>12.0f}"
              f"{m['total_ms']:>10.0f}{m['vram_mb']:>9}")


def make_chart(agg):
    ns = list(agg)
    tokens = [agg[n]["tokens"] for n in ns]
    prefill = [agg[n]["prefill_ms"] for n in ns]
    total = [agg[n]["total_ms"] for n in ns]
    vram = [agg[n]["vram_mb"] for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tokens, prefill, "o-", color="#c0504d", label="prefill (TTFT) latency")
    ax.plot(tokens, total, "s-", color="#4f81bd", label="total latency (200-tok output)")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("latency (ms)")
    ax.grid(True, alpha=0.3)

    # Peak VRAM on a twin axis — expected roughly flat (Ollama sizes the KV
    # cache to num_ctx, not to the actual prompt length).
    ax2 = ax.twinx()
    ax2.plot(tokens, vram, "^--", color="#7f7f7f", label="peak VRAM")
    ax2.set_ylabel("peak VRAM (MB)")
    ax2.set_ylim(0, max(vram) * 1.6)

    # Label each point with its N (retrieved-doc count).
    for x, y, n in zip(tokens, prefill, ns):
        ax.annotate(f"N={n}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#c0504d")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=9)
    ax.set_title("Latency vs context length — Qwen2.5-1.5B on a 4 GB GPU\n"
                 "(retrieved speeches N = 3…50)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=130)
    print(f"\nWrote chart -> {PNG_PATH}")


def main():
    rows = load()
    agg = aggregate(rows)
    print_table(agg)
    make_chart(agg)


if __name__ == "__main__":
    main()
