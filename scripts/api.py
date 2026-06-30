import json
import random
import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import myhansard
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from myhansard.bench import gpu_info, list_models, live_benchmark
from myhansard.embedder import _get_query_model
from myhansard.rag import answer, stream_answer
from pydantic import BaseModel

DB_PATH = Path("data/hansard.db")
CHROMA_PATH = Path("data/chroma")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
collection = myhansard.get_collection(CHROMA_PATH)

# Malay search term in the corpus -> English topic label shown in the question.
# Only topics that actually appear in the speeches are surfaced (checked at startup).
_TOPIC_TERMS = {
    "subsidi": "subsidies",
    "minyak": "fuel",
    "pendidikan": "education",
    "kesihatan": "healthcare",
    "banjir": "flood mitigation",
    "pengangkutan": "public transport",
    "perumahan": "housing",
    "jenayah": "crime",
    "rasuah": "corruption",
    "cukai": "taxes",
    "pekerjaan": "employment",
    "digital": "digitalisation",
    "pertanian": "agriculture",
    "alam sekitar": "the environment",
    "keselamatan": "national security",
    "ekonomi": "the economy",
}
# Procedural/chair roles to exclude when picking a named member.
_ROLE_EXCLUDE = ("Yang di-Pertua", "Pengerusi")
_suggest_cache: dict = {}


def _build_suggestion_facts() -> dict:
    """Read the real topics, members and sitting dates once and cache them."""
    topics = [
        label
        for term, label in _TOPIC_TERMS.items()
        if conn.execute(
            "SELECT 1 FROM speeches WHERE content LIKE ? LIMIT 1", (f"%{term}%",)
        ).fetchone()
    ]

    not_role = " AND ".join(f"speaker_raw NOT LIKE '%{r}%'" for r in _ROLE_EXCLUDE)
    members = [
        (re.sub(r"\s*\[.*$", "", sp).strip(), m.group(1))
        for (sp,) in conn.execute(
            f"SELECT speaker_raw FROM speeches "
            f"WHERE speaker_raw LIKE '%[%]' AND {not_role} "
            f"GROUP BY speaker_raw ORDER BY COUNT(*) DESC LIMIT 40"
        )
        if (m := re.search(r"\[(.*?)\]", sp))
    ]

    dates = [
        d for (d,) in conn.execute("SELECT DISTINCT date FROM speeches WHERE date IS NOT NULL")
    ]

    return {"topics": topics, "members": members, "dates": dates}


def _build_suggestions() -> list[str]:
    facts = _suggest_cache.setdefault("facts", _build_suggestion_facts())
    out: list[str] = []

    topics = random.sample(facts["topics"], min(2, len(facts["topics"])))
    if topics:
        out.append(f"What did members say about {topics[0]}?")
    if len(topics) > 1:
        out.append(f"Any issues raised about {topics[1]}?")

    if facts["members"]:
        name, const = random.choice(facts["members"])
        out.append(f"What did {name} ({const}) say in parliament?")

    if facts["dates"]:
        d = date.fromisoformat(random.choice(facts["dates"]))
        out.append(f"What topics were debated on {d.strftime('%-d %B %Y')}?")

    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_query_model()  # load weights at startup, not on first request
    _suggest_cache["facts"] = _build_suggestion_facts()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.get("/suggestions")
def suggestions():
    return {"suggestions": _build_suggestions()}


@app.post("/query")
def query(request: QueryRequest):
    result = answer(request.query, collection, conn)
    return result


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    def generate():
        for event in stream_answer(request.query, collection, conn):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Live inference benchmark powering the /eval page (real numbers, not canned).
# Real corpus queries, the same kind the chat UI runs.
BENCHMARK_QUERIES = [
    "What did members say about fuel subsidies?",
    "What was discussed about education funding?",
    "What did members say about healthcare and hospitals?",
    "What was raised about flood mitigation?",
    "What did members say about public transport?",
    "What was discussed about affordable housing?",
    "What did members say about corruption?",
    "What was raised about the cost of living and the economy?",
]


class BenchmarkRequest(BaseModel):
    model: str
    query: str


@app.get("/benchmark/info")
def benchmark_info():
    """Hardware, installed models and the query menu, all read live so the page
    reflects this machine rather than hard-coded values."""
    return {
        "hardware": gpu_info(),
        "models": list_models(),
        "queries": BENCHMARK_QUERIES,
    }


@app.post("/benchmark/run")
def benchmark_run(request: BenchmarkRequest):
    def generate():
        try:
            for event in live_benchmark(request.model, request.query, collection, conn):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # surface Ollama / model errors to the UI
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
