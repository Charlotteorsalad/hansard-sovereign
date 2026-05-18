from abc import ABC, abstractmethod
from pathlib import Path


class BaseStorage(ABC):
    @abstractmethod
    def save(self, date_str: str, content: bytes) -> None:
        pass


class LocalStorage(BaseStorage):
    def __init__(self, output_dir: str = "data/downloaded_pdfs"):
        self.output_dir = Path(output_dir)

    def save(self, date_str: str, content: bytes) -> None:
        file_path = self.output_dir / f"{date_str}.pdf"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        print(f"Saved to {file_path}")


class MongoStorage(BaseStorage):
    def __init__(
        self, connection_string: str, db_name: str, collection_name: str = "hansards"
    ):
        import pymongo

        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save(self, date_str: str, content: bytes) -> None:
        self.collection.update_one(
            {"date": date_str},
            {"$set": {"date": date_str, "pdf_content": content}},
            upsert=True,
        )
        print(f"Saved to MongoDB collection {self.collection.name} for date {date_str}")


class SQLStorage(BaseStorage):
    def __init__(self, connection_url: str, table_name: str = "hansards"):
        import sqlalchemy as sa

        self.engine = sa.create_engine(connection_url)
        self.table_name = table_name

        with self.engine.begin() as conn:
            conn.execute(
                sa.text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    date TEXT PRIMARY KEY,
                    pdf_content BLOB
                )
                """)
            )

    def save(self, date_str: str, content: bytes) -> None:
        import sqlalchemy as sa

        query = sa.text(f"""
        INSERT INTO {self.table_name} (date, pdf_content)
        VALUES (:date, :pdf_content)
        ON CONFLICT(date) DO UPDATE SET pdf_content = :pdf_content
        """)
        with self.engine.begin() as conn:
            conn.execute(query, {"date": date_str, "pdf_content": content})
        print(f"Saved to SQL table {self.table_name} for date {date_str}")
