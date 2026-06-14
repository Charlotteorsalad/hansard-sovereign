import requests

import myhansard


def answer(
    query: str, collection, conn, model: str = "llama3.1:8b-instruct-q4_K_M"
) -> dict:
    """
    RAG pipeline: retrieve relevant speeches, then generate grounded answer.

    Returns dict with 'answer' and 'sources'.
    """
    results = myhansard.query_speeches(collection, query)
    ids = [m["id"] for m in results["metadatas"][0]]

    cursor = conn.cursor()
    placeholders = ",".join("?" * len(ids))
    cursor.execute(
        f"SELECT id, speaker_raw, content, date FROM speeches WHERE id IN ({placeholders})",
        ids,
    )
    speeches = cursor.fetchall()

    context = "\n\n".join(
        [f"Speaker: {s[1]}\nDate: {s[3]}\nContent: {s[2]}" for s in speeches]
    )

    prompt = f"""You are an assistant that answers questions about Malaysian Parliament debates.
        Use the following speeches as context. Always cite the speaker and date.

        Context:
        {context}

        Question: {query}

        Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    answer_text = response.json()["response"]

    return {
        "answer": answer_text,
        "sources": [{"speaker": s[1], "date": s[3]} for s in speeches],
    }
