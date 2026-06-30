"""
Verify extractor correctness on a single PDF before committing to a full re-ingest.

For each extracted speech, shows:
  - The SPEAKER attributed to the content
  - The CONTENT start (first 120 chars)
  - The RAW PDF text immediately BEFORE the speaker's ]: anchor, so you can
    visually confirm the content belongs to that speaker.

Usage:
    python scripts/verify_extractor.py                    # uses first PDF in data/raw/
    python scripts/verify_extractor.py data/raw/04032024.pdf
    python scripts/verify_extractor.py data/raw/04032024.pdf --n 20
"""

import argparse
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from myhansard.extractor import extract_speeches, find_content_start


def raw_splits(pdf_path: Path, max_splits: int = 30) -> list[tuple[str, str]]:
    """
    Return (before, after) pairs for each ]: split in the PDF,
    so we can compare against what the extractor produced.
    Each 'before' is the 200 chars ending at ]: and 'after' is the 200 chars after.
    """
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        start = find_content_start(pdf_path)
        for i in range(start, len(pdf.pages)):
            full_text += (pdf.pages[i].extract_text() or "") + "\n"

    splits = []
    pos = 0
    while True:
        idx = full_text.find("]:", pos)
        if idx == -1 or len(splits) >= max_splits:
            break
        before = full_text[max(0, idx - 200) : idx + 2].replace("\n", "↵")
        after = full_text[idx + 2 : idx + 220].replace("\n", "↵")
        splits.append((before, after))
        pos = idx + 2
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", help="Path to PDF file")
    parser.add_argument("--n", type=int, default=15, help="Number of speeches to show")
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        pdfs = sorted(raw_dir.glob("*.pdf"))
        if not pdfs:
            print("No PDFs found in data/raw/")
            sys.exit(1)
        pdf_path = pdfs[0]

    print(f"Verifying: {pdf_path.name}\n")

    speeches = extract_speeches(pdf_path)
    raw = raw_splits(pdf_path, max_splits=args.n + 5)

    print(f"Extracted {len(speeches)} speeches total. Showing first {args.n}.\n")
    print("=" * 80)

    for i, s in enumerate(speeches[: args.n]):
        speaker = s["speaker_raw"].replace("\n", " ")
        content_start = s["content"][:120].replace("\n", "↵")
        content_end = s["content"][-80:].replace("\n", "↵") if len(s["content"]) > 120 else ""

        print(f"[{i+1}] SPEAKER : {speaker}")
        print(f"    CONTENT: {content_start}")
        if content_end:
            print(f"    ...END : {content_end}")
        print(f"    PAGE   : {s['page']}")
        print()

    # Cross-check: show the raw ]: splits so user can compare
    print("=" * 80)
    print("RAW ]: SPLITS from PDF (for manual cross-check):")
    print("Each split shows 200 chars BEFORE ]: and 200 chars AFTER\n")
    for i, (before, after) in enumerate(raw[: args.n]):
        print(f"[split {i+1}]")
        print(f"  BEFORE]: {before[-120:]}")
        print(f"  AFTER ]: {after[:120]}")
        print()


if __name__ == "__main__":
    main()
