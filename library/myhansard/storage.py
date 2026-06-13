import sqlite3
from pathlib import Path


def init_db(db_path: Path):
    """
    Create the SQLite database and speeches table if not exist.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS speeches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker_raw TEXT,
            content TEXT,
            page INTEGER,
            date DATE,
            type TEXT,
            source_file TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_speeches(
    conn: sqlite3.Connection, speeches: list[dict], date: str, source_file: str
) -> None:
    """
    Insert a list of speeches into the SQLite database.
    """
    cursor = conn.cursor()
    data = [
        (s["speaker_raw"], s["content"], s["page"], date, s["type"], source_file)
        for s in speeches
    ]
    cursor.executemany(
        """
        INSERT INTO speeches (speaker_raw, content, page, date, type, source_file)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        data,
    )
    conn.commit()
