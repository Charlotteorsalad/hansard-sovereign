"""
Measure the payoff of the fine-tune: does the small model produce the citation
format NATIVELY?

Runs a held-out set of queries through the base model and the fine-tuned model,
generating RAW (no Python post-processing, single attempt) and checking each
output with the same `_well_formatted` gate the pipeline uses. The gap in
well-formatted rate is what the QLoRA bought — i.e. how much of the brittle
Python format-enforcement the model now handles itself.

    CUDA_VISIBLE_DEVICES="" uv run python finetune/evaluate.py [base] [finetuned]
    # defaults: qwen2.5:1.5b  vs  hansard-qwen
"""

import sqlite3
import sys
from pathlib import Path

import myhansard
import requests
from myhansard.rag import (
    OLLAMA_BASE_URL,
    _build_prompt,
    _build_system,
    _detect_lang,
    _retrieve,
    _well_formatted,
)

DB_PATH = Path("data/hansard.db")
CHROMA_PATH = Path("data/chroma")

# Held-out queries (not phrased like the training topics).
EVAL_QUERIES = [
    "What concerns were raised about water supply?",
    "What did members say about small business support?",
    "What was discussed regarding road safety?",
    "What issues came up about the national budget?",
    "What did members say about renewable energy?",
    "What was raised about tourism?",
    "What concerns were there about drug abuse?",
    "What did members say about the welfare of the elderly?",
    "What was discussed about internet access in rural areas?",
    "What did members say about food security?",
    "Apakah yang dibincangkan tentang bekalan air?",
    "Apa yang dibangkitkan tentang keselamatan jalan raya?",
    "Apakah isu tentang tenaga boleh diperbaharui?",
    "Apa yang dikatakan tentang kebajikan warga emas?",
]


def raw_generate(model: str, prompt: str, system: str) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "system": system, "prompt": prompt, "stream": False,
              # Cap output: the format check only needs the first few list items,
              # and it stops a fine-tune that fails to emit EOS from running away.
              "options": {"temperature": 0.3, "seed": 7, "num_predict": 400}},
        timeout=180,
    )
    return r.json().get("response", "")


# Raw auto-generated run output. The curated, presentation-ready report lives in
# eval_results.md and is not overwritten by re-runs.
RESULTS_MD = Path("finetune/eval_run.md")


def run(model: str, collection, conn) -> list[dict]:
    """Generate raw output for each query; return per-query {query, ok, raw}."""
    rows = []
    for q in EVAL_QUERIES:
        speeches = _retrieve(q, collection, conn)
        if len(speeches) < 2:
            continue
        raw = raw_generate(model, _build_prompt(q, speeches),
                           _build_system(_detect_lang(q)))
        ok = _well_formatted(raw)
        rows.append({"query": q, "ok": ok, "raw": raw})
        print(f"  [{model}] {'OK ' if ok else 'off'}  {q[:45]}", flush=True)
    return rows


def write_report(base, ft, base_rows, ft_rows):
    bg, bt = sum(r["ok"] for r in base_rows), len(base_rows)
    fg, ft_t = sum(r["ok"] for r in ft_rows), len(ft_rows)
    ft_by_q = {r["query"]: r for r in ft_rows}

    L = [
        "# Fine-tune evaluation — native citation-format adherence",
        "",
        "Each held-out query is generated **raw** (single attempt, no Python "
        "post-processing) and checked with the pipeline's `_well_formatted` gate "
        "(≥2 `**bold-title**` list items). This measures what the *model* learned.",
        "",
        "## Result",
        "",
        "| model | well-formatted |",
        "| --- | --- |",
        f"| `{base}` (base) | {bg}/{bt} — **{100 * bg / bt:.0f}%** |",
        f"| `{ft}` (QLoRA) | {fg}/{ft_t} — **{100 * fg / ft_t:.0f}%** |",
        "",
        "## Per-query",
        "",
        "| query | base | fine-tuned |",
        "| --- | :---: | :---: |",
    ]
    for r in base_rows:
        f = ft_by_q.get(r["query"], {})
        L.append(f"| {r['query']} | {'✅' if r['ok'] else '❌'} "
                 f"| {'✅' if f.get('ok') else '❌'} |")

    # Two side-by-side examples where the base fails and the fine-tune succeeds.
    examples = [r["query"] for r in base_rows
                if not r["ok"] and ft_by_q.get(r["query"], {}).get("ok")][:2]
    if examples:
        L += ["", "## Side-by-side (base fails, fine-tune holds the format)"]
        base_by_q = {r["query"]: r for r in base_rows}
        for q in examples:
            L += [
                "", f"### {q}",
                "", "**base** (raw):", "```",
                base_by_q[q]["raw"].strip()[:600], "```",
                "", "**fine-tuned** (raw):", "```",
                ft_by_q[q]["raw"].strip()[:600], "```",
            ]
    RESULTS_MD.write_text("\n".join(L) + "\n")
    print(f"\nWrote evidence -> {RESULTS_MD}")


def make_chart(base, ft, base_rows, ft_rows):
    """Bar chart (adherence %) + per-query pass grid -> results/finetune_eval.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    labels = [r["query"].replace("?", "")[:22] for r in base_rows]
    b = [int(r["ok"]) for r in base_rows]
    f = [int(r["ok"]) for r in ft_rows]
    bp, fp = 100 * sum(b) / len(b), 100 * sum(f) / len(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    bars = ax1.bar([f"base\n{base}", f"fine-tuned\n{ft}"], [bp, fp],
                   color=["#c0504d", "#4f81bd"], width=0.6)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("native format-adherence (%)")
    ax1.set_title("Format adherence — raw output, no post-processing", fontsize=10)
    for bar, g, t in zip(bars, [sum(b), sum(f)], [len(b), len(f)]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{bar.get_height():.0f}%\n({g}/{t})", ha="center",
                 va="bottom", fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    grid = np.array([b, f]).T
    ax2.imshow(grid, cmap=ListedColormap(["#e8b0ac", "#8fbf7f"]),
               aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["base", "fine-tuned"], fontsize=9)
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.set_title(f"Per-query pass ({len(labels)} held-out)", fontsize=10)
    for i in range(len(labels)):
        for j, val in enumerate([b[i], f[i]]):
            ax2.text(j, i, "✓" if val else "✗", ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")
    ax2.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax2.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=1.5)
    ax2.tick_params(which="minor", length=0)

    fig.suptitle(f"QLoRA fine-tune: citation-format adherence  {bp:.0f}% → "
                 f"{fp:.0f}%", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path("results/finetune_eval.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Wrote chart -> {out}")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"
    ft = sys.argv[2] if len(sys.argv) > 2 else "hansard-qwen"
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    collection = myhansard.get_collection(CHROMA_PATH)

    print("\n=== base model ===")
    base_rows = run(base, collection, conn)
    print("\n=== fine-tuned model ===")
    ft_rows = run(ft, collection, conn)

    bg, fg = sum(r["ok"] for r in base_rows), sum(r["ok"] for r in ft_rows)
    print("\n--- native format-adherence (raw output, no Python cleanup) ---")
    print(f"{base:<20} {bg}/{len(base_rows)}  ({100 * bg / len(base_rows):.0f}%)")
    print(f"{ft:<20} {fg}/{len(ft_rows)}  ({100 * fg / len(ft_rows):.0f}%)")
    write_report(base, ft, base_rows, ft_rows)
    make_chart(base, ft, base_rows, ft_rows)


if __name__ == "__main__":
    main()
