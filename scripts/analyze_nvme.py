"""
Summarise results/nvme_load_benchmark.csv into the three-tier transfer story:
NVMe -> system RAM -> VRAM.

Per model, run 0 is a *cold* load (first read, from NVMe) and later runs are
*warm* (the file is now in the OS page cache, so it's RAM -> VRAM over PCIe).
The gap between them is the NVMe read penalty you pay on a truly cold swap.

    uv run python scripts/analyze_nvme.py
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = Path("results/nvme_load_benchmark.csv")
PNG_PATH = Path("results/nvme_load_benchmark.png")


def main():
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    agg = []
    for model, rs in by_model.items():
        rs.sort(key=lambda r: int(r["run"]))
        cold = float(rs[0]["load_ms"])                       # run 0: from NVMe
        warm = mean(float(r["load_ms"]) for r in rs[1:]) if len(rs) > 1 else cold
        agg.append({"model": model, "size_mb": float(rs[0]["size_mb"]),
                    "cold_ms": cold, "warm_ms": warm})
    agg.sort(key=lambda a: a["size_mb"])

    print(f"{'model':<32}{'size_MB':>9}{'warm_ms':>9}{'cold_ms':>9}{'NVMe_pen':>10}")
    print("-" * 69)
    for a in agg:
        pen = a["cold_ms"] - a["warm_ms"]
        print(f"{a['model']:<32}{a['size_mb']:>9.0f}{a['warm_ms']:>9.0f}"
              f"{a['cold_ms']:>9.0f}{pen:>10.0f}")

    labels = [a["model"].replace(":latest", "").replace("-instruct-q4_K_M", "-q4")
              .replace("-instruct", "") for a in agg]
    warm = [a["warm_ms"] for a in agg]
    penalty = [max(a["cold_ms"] - a["warm_ms"], 0) for a in agg]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(labels, warm, color="#4f81bd", width=0.6,
                label="warm: RAM → VRAM (PCIe)")
    b2 = ax.bar(labels, penalty, bottom=warm, color="#c0504d", width=0.6,
                label="cold add-on: NVMe → RAM read")
    ax.set_ylabel("model-load latency (ms)")
    ax.set_title("Where model-load time goes: NVMe → system RAM → VRAM\n"
                 "(measured cold vs page-cache-warm, per model size)", fontsize=11)
    for a, w, p in zip(agg, warm, penalty):
        ax.text(labels.index(
            a["model"].replace(":latest", "").replace("-instruct-q4_K_M", "-q4")
            .replace("-instruct", "")), w + p,
            f"{a['size_mb']:.0f} MB\ncold {a['cold_ms']:.0f} ms",
            ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(a["cold_ms"] for a in agg) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=130)
    print(f"\nWrote chart -> {PNG_PATH}")


if __name__ == "__main__":
    main()
