"""
Build a QLoRA fine-tuning dataset by self-distilling the production RAG pipeline.

The known pain point: small models don't reliably follow the strict output format
(numbered list of `**Title**: Speaker (Constituency) summary [n].`), so today the
project enforces it in Python. This script generates (prompt -> gold answer) pairs
using the EXISTING pipeline as the teacher — real retrieval + the 8B model + the
same Python cleanup (`_generate_body` applies `_auto_cite`) — so the targets are
exactly what production emits after its format enforcement. Fine-tuning a small
model on these teaches it to produce that format natively.

Only well-formatted answers are kept (the teacher occasionally drifts).

    CUDA_VISIBLE_DEVICES="" uv run python finetune/build_finetune_data.py [N]

Writes finetune/train.jsonl + finetune/val.jsonl with {system, prompt, completion}.
"""

import json
import random
import re
import sqlite3
import sys
from pathlib import Path

import myhansard
from myhansard.rag import (
    _build_prompt,
    _build_system,
    _detect_lang,
    _generate_body,
    _retrieve,
    _well_formatted,
)

DB_PATH = Path("data/hansard.db")
CHROMA_PATH = Path("data/chroma")
OUT_DIR = Path("finetune")
TEACHER_MODEL = "llama3.1:8b-instruct-q4_K_M"  # production model = teacher
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 120
VAL_FRACTION = 0.1
random.seed(42)

# Diverse queries: topics (EN + MS), real members, real sitting dates — so the
# model sees varied retrieval contexts, not one template.
TOPIC_QUERIES = [
    "What did members say about fuel subsidies?",
    "What was discussed about education funding?",
    "What did members say about healthcare and hospitals?",
    "What was raised about flood mitigation?",
    "What did members say about public transport?",
    "What was discussed about affordable housing?",
    "What did members say about corruption?",
    "What was raised about the cost of living?",
    "What did members say about agriculture and farmers?",
    "What was discussed about digitalisation and technology?",
    "What did members say about national security?",
    "What was raised about employment and jobs?",
    "What did members say about the environment?",
    "What was discussed about taxes and revenue?",
    "What did members say about rural development?",
    "Apakah yang dibincangkan tentang subsidi minyak?",
    "Apakah isu yang dibangkitkan tentang pendidikan?",
    "Apa yang dikatakan ahli tentang kesihatan?",
    "Apakah yang dibincangkan tentang banjir?",
    "Apa yang dibangkitkan tentang kos sara hidup?",
    "Apakah isu tentang pengangkutan awam?",
]


def build_queries(conn: sqlite3.Connection) -> list[str]:
    qs = list(TOPIC_QUERIES)

    # Real members (skip procedural chair roles), asked a few ways.
    not_role = "speaker_raw NOT LIKE '%Yang di-Pertua%' AND speaker_raw NOT LIKE '%Pengerusi%'"
    names = []
    for (sp,) in conn.execute(
        f"SELECT speaker_raw FROM speeches WHERE speaker_raw LIKE '%[%]' AND {not_role} "
        f"GROUP BY speaker_raw ORDER BY COUNT(*) DESC LIMIT 250"
    ):
        name = re.sub(r"\s*\[.*$", "", sp).strip()
        if 3 < len(name) < 45:
            names.append(name)
    for name in random.sample(names, min(160, len(names))):
        qs.append(random.choice([
            f"What did {name} say in parliament?",
            f"Summarise the points raised by {name}.",
            f"What issues did {name} bring up?",
            f"What did {name} raise in the debate?",
        ]))

    # Real sitting dates.
    dates = [d for (d,) in conn.execute(
        "SELECT DISTINCT date FROM speeches WHERE date IS NOT NULL")]
    for d in random.sample(dates, min(30, len(dates))):
        qs.append(f"What topics were debated on {d}?")

    random.shuffle(qs)
    return qs


def main():
    OUT_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    collection = myhansard.get_collection(CHROMA_PATH)

    queries = build_queries(conn)
    print(f"{len(queries)} candidate queries; targeting {TARGET} good examples "
          f"(teacher={TEACHER_MODEL})")

    # Append each kept example immediately so progress survives interruption,
    # and split train/val in a finally block so we always end up with usable
    # files even if the (slow) run is cut short.
    raw_path = OUT_DIR / "examples.jsonl"

    def write_split():
        rows = [json.loads(ln) for ln in raw_path.open()] if raw_path.exists() else []
        if not rows:
            return
        random.shuffle(rows)
        n_val = max(1, int(len(rows) * VAL_FRACTION))
        for name, part in [("train", rows[n_val:]), ("val", rows[:n_val])]:
            with (OUT_DIR / f"{name}.jsonl").open("w") as f:
                for r in part:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Wrote {len(part)} -> finetune/{name}.jsonl")

    kept = 0
    try:
        with raw_path.open("w") as raw:
            for i, q in enumerate(queries):
                if kept >= TARGET:
                    break
                speeches = _retrieve(q, collection, conn)
                if len(speeches) < 2:
                    continue
                lang = _detect_lang(q)
                prompt = _build_prompt(q, speeches)
                system = _build_system(lang)
                body = _generate_body(prompt, system, speeches, TEACHER_MODEL)
                if not _well_formatted(body):
                    print(f"  [{i}] skip (off-format): {q[:50]}", flush=True)
                    continue
                raw.write(json.dumps(
                    {"system": system, "prompt": prompt, "completion": body},
                    ensure_ascii=False) + "\n")
                raw.flush()
                kept += 1
                print(f"  [{kept}/{TARGET}] kept: {q[:50]}", flush=True)
    finally:
        write_split()


if __name__ == "__main__":
    main()
