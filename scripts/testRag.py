import sqlite3
from pathlib import Path

import myhansard
from myhansard.rag import answer

conn = sqlite3.connect(Path("data/hansard.db"))
collection = myhansard.get_collection(Path("data/chroma"))

result = answer("What did members say about fuel subsidies?", collection, conn)
print(result["answer"])
print("\nSources:")
for s in result["sources"]:
    print(f"- {s['speaker']} on {s['date']}")
