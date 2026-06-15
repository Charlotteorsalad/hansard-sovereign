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

    # Keyword search
    cursor = conn.cursor()
    keywords = query.replace("?", "").split()
    keyword_conditions = " OR ".join([f"content LIKE '%{k}%'" for k in keywords])
    cursor.execute(
        f"SELECT id FROM speeches WHERE ({keyword_conditions}) AND LENGTH(content) > 100 LIMIT 10"
    )
    keyword_ids = [row[0] for row in cursor.fetchall()]

    # Hybrid search
    all_ids = list(set(ids + keyword_ids))

    placeholders = ",".join("?" * len(all_ids))
    cursor.execute(
        f"SELECT id, speaker_raw, content, date FROM speeches WHERE id IN ({placeholders})",
        all_ids,
    )
    speeches = cursor.fetchall()
    speeches = [s for s in speeches if len(s[2].strip()) > 100]

    # Remove repetition source
    seen = set()
    unique_speeches = []
    for s in speeches:
        if s[0] not in seen:
            seen.add(s[0])
            unique_speeches.append(s)
    speeches = unique_speeches

    numbered_context = "\n\n".join(
        [
            f"[{idx + 1}] Speaker: {s[1]}\nDate: {s[3]}\nContent: {s[2]}"
            for idx, s in enumerate(speeches)
        ]
    )

    prompt = f"""You are an assistant that answers questions about Malaysian Parliament debates.
    Use ONLY the provided speeches as context. Answer in English.
    When you use information from a speech, you MUST cite it inline using square brackets like [1], [2], [3]. Never write "speech 4", always write [4].
    Do NOT repeat or quote the speeches directly. Summarize and cite by number.
    If the context does not contain relevant information, say so clearly.

    Context:
    {numbered_context}

    Question: {query}

    Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    answer_text = response.json()["response"]

    return {
        "answer": answer_text,
        "sources": [
            {"index": idx + 1, "speaker": s[1], "date": s[3], "content": s[2][:200]}
            for idx, s in enumerate(speeches)
        ],
    }
