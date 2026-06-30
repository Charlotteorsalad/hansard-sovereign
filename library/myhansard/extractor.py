import re
from pathlib import Path

import pdfplumber


def peek(pdf_path: Path, max_pages: int = 50) -> None:
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text() or None
            print(f"Page {i + 1}: {text[:500]}...")


def find_content_start(pdf_path: Path) -> int:
    """Return the 0-based page index where speeches start.

    The first page containing "DOA" (the standard opening prayer) marks the
    start of debate content. Tested against the 2024 Parlimen ke-15 format.
    Raises ValueError if the marker is missing.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if "DOA" in text:
                return i

    raise ValueError("Content start marker not found in PDF")


def list_speech_anchors(pdf_path: Path, max_pages: int = 50) -> None:
    start_page = find_content_start(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page, min(start_page + max_pages, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ""
            lines = text.split("\n")
            for line in lines:
                if "]:" in line:
                    print(f"Page {i + 1}: {line[:200]}...")


def list_agenda_anchors(pdf_path: Path, max_pages: int = 50) -> None:
    start_page = find_content_start(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page, min(start_page + max_pages, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ""
            lines = text.split("\n")
            for line in lines:
                if "]" in line:
                    print(f"Page {i + 1}: {line[:200]}...")


def extract_speeches(pdf_path: Path) -> list[dict]:
    """Extract speeches from a Hansard PDF.

    Splits on "]:" anchors like "Dato' Seri Anwar Ibrahim [Bera]:". Each content
    block belongs to the speaker named in the previous split, so we hold that
    name in pending_speaker rather than the one that follows the block.

    Returns dicts with keys: type, speaker_raw, content, page.
    """
    start_page = find_content_start(pdf_path)
    results = []
    pending_speaker = None

    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page, len(pdf.pages)):
            text = pdf.pages[i].extract_text() or ""
            parts = text.split("]:")

            for part in parts:
                last_bracket = part.rfind("[")

                if last_bracket == -1:
                    # No following speaker tag, so the rest is the pending speaker's
                    if pending_speaker:
                        content = part.strip()
                        if len(content) >= 80:
                            results.append(
                                {
                                    "type": "speech",
                                    "speaker_raw": pending_speaker,
                                    "content": content,
                                    "page": i + 1,
                                }
                            )
                    continue

                matches = list(re.finditer(r"\.\s*\n", part[:last_bracket]))
                if matches:
                    cut = matches[-1].end()
                else:
                    cut = part.rfind("\n", 0, last_bracket) + 1
                last_newline_after_cut = part.rfind("\n", cut, last_bracket)
                if last_newline_after_cut != -1:
                    cut = last_newline_after_cut + 1

                content = part[:cut].strip()
                next_speaker_raw = (
                    part[cut:last_bracket].strip()
                    + " "
                    + part[last_bracket:].strip()
                    + "]"
                )

                if pending_speaker and len(content) >= 80:
                    results.append(
                        {
                            "type": "speech",
                            "speaker_raw": pending_speaker,
                            "content": content,
                            "page": i + 1,
                        }
                    )

                pending_speaker = next_speaker_raw

    return results
