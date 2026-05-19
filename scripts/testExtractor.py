from pathlib import Path

from myhansard.extractor import find_content_start

# myhansard.peek(Path("data/raw/04032024.pdf"), max_pages=50)

for pdf_file in sorted(Path("data/raw").glob("*.pdf")):
    try:
        start = find_content_start(pdf_file)
        print(f"{pdf_file.name}: content starts at page {start + 1}")
    except ValueError as e:
        print(f"{pdf_file.name}: FAILED - {e}")
