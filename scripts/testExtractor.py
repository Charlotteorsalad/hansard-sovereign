from pathlib import Path

from myhansard.extractor import extract_speeches

# myhansard.peek(Path("data/raw/04032024.pdf"), max_pages=50)

# for pdf_file in sorted(Path("data/raw").glob("*.pdf")):
#     try:
#         start = find_content_start(pdf_file)
#         print(f"{pdf_file.name}: content starts at page {start + 1}")
#     except ValueError as e:
#         print(f"{pdf_file.name}: FAILED - {e}")

# for pdf_file in sorted(Path("data/raw").glob("*.pdf")):
#     try:
#         list_speech_anchors(pdf_file, max_pages=50)
#     except ValueError as e:
#         print(f"{pdf_file.name}: FAILED - {e}")

# for pdf_file in sorted(Path("data/raw").glob("*.pdf")):
#     try:
#         list_agenda_anchors(pdf_file, max_pages=50)
#     except ValueError as e:
#         print(f"{pdf_file.name}: FAILED - {e}")

# results = extract_speeches(Path("data/raw/04032024.pdf"))
# for r in results[:5]:
#     print(r)
#     print()

for pdf_file in sorted(Path("data/raw").glob("*.pdf")):
    results = extract_speeches(pdf_file)
    print(f"{pdf_file.name}: {len(results)} speeches")
