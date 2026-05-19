from pathlib import Path

import pdfplumber


def peek(pdf_path: Path, max_pages: int = 50) -> None:
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text() or None
            print(f"Page {i + 1}: {text[:500]}...")


def find_content_start(pdf_path: Path) -> int:
    """
    Find the page index where the content (议员发言) starts.

    Looks for marker 'DOA' (Islamic prayer, Parliament's standard opening ritual).

    KNOWN-WORKING FORMAT: 2024 Parlimen ke-15.

    Returns:
        0-based page index (e.g. returns 10 if content starts on visual page 11)

    Raises:
        ValueError: if marker not found anywhere in PDF
    """
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if "DOA" in text:
                return i

    raise ValueError("Content start marker not found in PDF")
