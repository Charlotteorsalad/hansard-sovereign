"""
Full pipeline: ingest PDFs into SQLite, then embed into ChromaDB.

Options:
    --fresh     Drop and recreate the DB and ChromaDB before running (required
                after extractor changes so old wrong data is not kept)
    --ingest-only   Stop after ingest, skip embedding
    --embed-only    Skip ingest, only run embedding (useful if ingest already done)

Usage:
    python scripts/pipeline.py --fresh      # clean rebuild (use after extractor fix)
    python scripts/pipeline.py              # incremental (skip already-ingested dates)
    python scripts/pipeline.py --ingest-only
    python scripts/pipeline.py --embed-only
"""

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

import myhansard

DB_PATH = Path("data/hansard.db")
CHROMA_PATH = Path("data/chroma")
RAW_DIR = Path("data/raw")


def run_ingest(conn: sqlite3.Connection) -> None:
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDF(s) in {RAW_DIR}/\n")

    for idx, pdf_path in enumerate(pdfs, 1):
        date = (
            pdf_path.stem[4:]
            + "-"
            + pdf_path.stem[2:4]
            + "-"
            + pdf_path.stem[:2]
        )
        if myhansard.date_exists(conn, date):
            print(f"[{idx}/{len(pdfs)}] SKIP  {pdf_path.name} (already ingested)")
            continue

        t0 = time.time()
        speeches = myhansard.extract_speeches(pdf_path)
        myhansard.insert_speeches(conn, speeches, date, str(pdf_path))
        elapsed = time.time() - t0
        print(
            f"[{idx}/{len(pdfs)}] OK    {pdf_path.name}  "
            f"→ {len(speeches)} speeches  ({elapsed:.1f}s)"
        )

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM speeches")
    total = cursor.fetchone()[0]
    print(f"\nIngest complete. Total speeches in DB: {total}")


def run_embed(conn: sqlite3.Connection, collection) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM speeches")
    total = cursor.fetchone()[0]
    print(f"Embedding {total} speeches into ChromaDB…")
    t0 = time.time()
    myhansard.embed_speeches(conn, collection)
    elapsed = time.time() - t0
    print(f"Embedding complete ({elapsed:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing DB and ChromaDB before running",
    )
    parser.add_argument(
        "--fresh-chroma",
        action="store_true",
        help="Delete only ChromaDB before running (keeps DB intact, use to re-embed)",
    )
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    args = parser.parse_args()

    if args.fresh:
        if DB_PATH.exists():
            DB_PATH.unlink()
            print(f"Deleted {DB_PATH}")
        if CHROMA_PATH.exists():
            shutil.rmtree(CHROMA_PATH)
            print(f"Deleted {CHROMA_PATH}")
        print()
    elif args.fresh_chroma:
        if CHROMA_PATH.exists():
            shutil.rmtree(CHROMA_PATH)
            print(f"Deleted {CHROMA_PATH}\n")

    myhansard.init_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    try:
        if not args.embed_only:
            print("=== INGEST ===")
            run_ingest(conn)
            print()

        if not args.ingest_only:
            print("=== EMBED ===")
            collection = myhansard.get_collection(CHROMA_PATH)
            run_embed(conn, collection)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
