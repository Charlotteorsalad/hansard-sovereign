from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def get_collection(chroma_path: Path, collection_name: str = "hansard"):
    """
    Initialize ChromaDB client and return the hansard collection.
    """
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def embed_speeches(conn, collection, model_name: str = "BAAI/bge-m3") -> None:
    """
    Embed all speeches from SQLite and store in ChromaDB.
    BAAI/bge-m3 supports multilingual including Bahasa Malaysia.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, speaker_raw, content, date, source_file FROM speeches")
    rows = cursor.fetchall()

    model = SentenceTransformer(model_name, device="cuda")

    batch_size = 1000
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        print(
            f"Embedding batch {i // batch_size + 1}/{(len(rows) - 1) // batch_size + 1}..."
        )
        embeddings = model.encode([r[2] for r in batch])
        collection.add(
            ids=[str(r[0]) for r in batch],
            embeddings=[e.tolist() for e in embeddings],
            metadatas=[
                {"id": r[0], "speaker_raw": r[1], "date": r[3], "source_file": r[4]}
                for r in batch
            ],
        )


def query_speeches(
    collection, query: str, model_name: str = "BAAI/bge-m3", n_results: int = 5
) -> list[dict]:
    """Query ChromaDB for speeches similar to the query string.
    Returns top n_results matches with metadata.
    """
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )
    return results
