from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

MIN_CONTENT_LEN = 80

_query_model: "SentenceTransformer | None" = None


def _get_query_model(model_name: str = "BAAI/bge-m3") -> "SentenceTransformer":
    global _query_model
    if _query_model is None:
        _query_model = SentenceTransformer(model_name, device="cpu")
    return _query_model


def get_collection(chroma_path: Path, collection_name: str = "hansard"):
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def embed_speeches(conn, collection, model_name: str = "BAAI/bge-m3") -> None:
    """Embed speeches from SQLite into ChromaDB.

    Uses FP16 on CUDA for speed and normalised embeddings (bge-m3 needs this for
    correct cosine similarity). IDs already in ChromaDB are skipped, so an
    interrupted run can resume.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, speaker_raw, content, date, source_file FROM speeches"
        f" WHERE LENGTH(TRIM(content)) >= {MIN_CONTENT_LEN}"
    )
    rows = cursor.fetchall()
    total = len(rows)
    print(f"Records to embed: {total}")

    model = SentenceTransformer(
        model_name,
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )

    batch_size = 1000
    n_batches = (total + batch_size - 1) // batch_size
    embedded = 0

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        batch_num = i // batch_size + 1

        existing = set(collection.get(ids=[str(r[0]) for r in batch])["ids"])
        batch = [r for r in batch if str(r[0]) not in existing]

        if not batch:
            print(f"[{batch_num}/{n_batches}] already done, skipping")
            continue

        print(f"[{batch_num}/{n_batches}] embedding {len(batch)} records…")
        embeddings = model.encode(
            [r[2] for r in batch],
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        collection.add(
            ids=[str(r[0]) for r in batch],
            embeddings=[e.tolist() for e in embeddings],
            metadatas=[
                {"id": r[0], "speaker_raw": r[1], "date": r[3], "source_file": r[4]}
                for r in batch
            ],
        )
        embedded += len(batch)

    print(f"Done. Embedded {embedded} records ({total - embedded} skipped).")


def query_speeches(
    collection, query: str, model_name: str = "BAAI/bge-m3", n_results: int = 10
) -> list[dict]:
    """Query ChromaDB for speeches similar to the query string."""
    model = _get_query_model(model_name)
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )
    return results
