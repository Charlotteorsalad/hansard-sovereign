import sqlite3
from pathlib import Path

import myhansard

DB_PATH = Path("data/hansard.db")
CHROMA_PATH = Path("data/chroma")


def main():
    conn = sqlite3.connect(DB_PATH)
    collection = myhansard.get_collection(CHROMA_PATH)
    myhansard.embed_speeches(conn, collection)
    conn.close()


if __name__ == "__main__":
    main()
