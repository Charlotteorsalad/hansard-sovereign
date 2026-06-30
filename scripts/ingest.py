import sqlite3
from pathlib import Path

import myhansard

DB_PATH = Path("data/hansard.db")
RAW_DIR = Path("data/raw")


def main():
    myhansard.init_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    for hansard_raw in RAW_DIR.glob("*.pdf"):
        print(f"Processing {hansard_raw}...")
        date = (
            hansard_raw.stem[4:]
            + "-"
            + hansard_raw.stem[2:4]
            + "-"
            + hansard_raw.stem[:2]
        )
        if myhansard.date_exists(conn, date):
            print(f"Skipping {hansard_raw.name}, already ingested.")
            continue
        speeches = myhansard.extract_speeches(hansard_raw)
        myhansard.insert_speeches(conn, speeches, date, str(hansard_raw))
    conn.close()


if __name__ == "__main__":
    main()
