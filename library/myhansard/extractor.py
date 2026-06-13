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
    """
    A function to enumerate the document and extract speeches/agendas into a structured
    format.

    Anchor to find speeches: lines containing "]:" (e.g. "YB Dato' Seri Anwar Ibrahim [PKR-PKR]:...")
    Anchor to find agendas: lines containing "]" (e.g. "1. PENGENALAN YB DATO' SERI ANWAR IBRAHIM [PKR-PKR] meminta...")

    Returns: A list dictionaries, with the keys of 'type' (either 'speech' or 'agenda'), 'speaker_raw' (for speeches), 'content', and 'page'.
    """
    start_page = find_content_start(pdf_path)
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page, len(pdf.pages)):
            text = pdf.pages[i].extract_text() or ""
            lines = text.split("\n")
            for line in lines:
                if "]:" in line:
                    split = line.split("]:", 1)
                    speaker_raw = split[0] + "]"
                    content = split[1].strip()
                    results.append(
                        {
                            "type": "speech",
                            "speaker_raw": speaker_raw,
                            "content": content,
                            "page": i + 1,
                        }
                    )
    return results
