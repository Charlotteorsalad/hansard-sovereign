import sqlite3
from pathlib import Path

import myhansard

DB_PATH = Path("data/hansard.db")
CHROMA_PATH = Path("data/chroma")


def main():

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)

    # Initialize ChromaDB collection
    collection = myhansard.get_collection(CHROMA_PATH)

    # Embed speeches and store in ChromaDB
    myhansard.embed_speeches(conn, collection)

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    main()
