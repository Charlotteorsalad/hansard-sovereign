import asyncio
import json
import queue
import random
import re
import sqlite3
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import myhansard
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from myhansard.bench import (
    CONTEXT_N_VALUES,
    context_run,
    gpu_info,
    list_models,
    live_benchmark,
)
from myhansard.embedder import _get_query_model
from myhansard.rag import DEFAULT_MODEL, answer, stream_answer
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
    """Pick 4 example questions from real corpus facts (topics/members/dates).

    Freshly randomised on every call — the frontend controls the cadence by
    calling this once per genuinely new chat (not on every render), so this
    only needs to vary from call to call rather than manage its own rotation
    schedule.
    """
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
    model: str | None = None  # let the UI pick 8B vs the fine-tune; None = default


@app.get("/suggestions")
def suggestions():
    return {"suggestions": _build_suggestions()}


@app.post("/query")
def query(request: QueryRequest):
    result = answer(request.query, collection, conn, model=request.model or DEFAULT_MODEL)
    return result


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    model = request.model or DEFAULT_MODEL

    def generate():
        for event in stream_answer(request.query, collection, conn, model=model):
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
        "production_model": DEFAULT_MODEL,
    }


@app.post("/benchmark/run")
def benchmark_run(request: BenchmarkRequest):
    """Streamed live-benchmark run (SSE) for one model/query.

    live_benchmark() is a synchronous generator doing blocking I/O against
    Ollama, so it runs in a background thread; this async generator relays
    each event to the client. Manually polling Request.is_disconnected() here
    turned out to be unreliable (verified live: it never fired), likely
    fighting Starlette's own disconnect-watcher for the same ASGI receive
    channel. Relying on that built-in mechanism instead — it cancels this
    generator once it detects the client is gone — and catching that in
    `finally` to set cancel_event is what actually and reliably worked in
    testing. Without this, a Stop click (or the client simply giving up) left
    the backend still waiting on Ollama regardless — burning the one
    generation slot this 4 GB GPU has until that request finished or hit its
    own 600s timeout, and wedging every request after it behind it.
    """
    events: queue.Queue = queue.Queue()
    cancel_event = threading.Event()
    _sentinel = object()

    def worker():
        try:
            for event in live_benchmark(
                request.model, request.query, collection, conn, cancel_event
            ):
                events.put(event)
        except Exception as exc:  # surface Ollama / model errors to the UI
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(_sentinel)

    async def generate():
        threading.Thread(target=worker, daemon=True).start()
        try:
            while True:
                try:
                    item = await asyncio.to_thread(events.get, True, 0.25)
                except queue.Empty:
                    continue
                if item is _sentinel:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            # Reached on normal completion too (cancel_event.set() after the
            # worker is already done is a harmless no-op) — and reliably
            # reached when the client disconnects, since Starlette cancels
            # this generator via GeneratorExit/CancelledError at whichever
            # `await`/`yield` it's suspended at.
            cancel_event.set()

    return StreamingResponse(generate(), media_type="text/event-stream")


class ContextRequest(BaseModel):
    model: str
    query: str
    n_results: int


@app.get("/benchmark/context/steps")
def benchmark_context_steps():
    """The N (retrieved-doc count) values the /eval context sweep steps through."""
    return {"n_values": CONTEXT_N_VALUES}


@app.post("/benchmark/context")
def benchmark_context(request: ContextRequest):
    """One point of the context-length curve: prefill/latency for a given N.

    Implemented as a StreamingResponse yielding periodic keep-alive chunks
    and a single final JSON chunk — not because the response is actually
    streamed (the client still just calls .json() on it once), but because
    that's the pattern Starlette reliably detects client disconnection for.
    Polling Request.is_disconnected() on a plain response here was tried
    first and, verified live, never actually fired — this endpoint was the
    one that got stuck and needed a manual Ollama restart. The same logic
    wrapped in an async generator inside a StreamingResponse (as
    benchmark_run() below already does) was verified to detect disconnection
    instantly instead. context_run() is synchronous, so it runs in a
    background thread.

    Note: even with instant detection here, a client that disconnects while
    Ollama is mid-prefill on a long/CPU-spilled prompt won't free the GPU
    slot until that prefill batch finishes — llama.cpp only checks for a
    closed connection between generation steps, not mid-batch. That's a
    limit of the inference engine, not of this cancellation path (verified
    live with debug timestamps: cancel_event fires and closes our connection
    to Ollama the instant the client disconnects).
    """
    cancel_event = threading.Event()
    result: dict = {}

    def worker():
        try:
            result["data"] = context_run(
                request.model, request.query, request.n_results,
                collection, conn, cancel_event,
            )
        except Exception as exc:  # surface Ollama / model errors to the UI
            result["error"] = str(exc)

    async def generate():
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                await asyncio.sleep(0.1)
                yield " "
            yield json.dumps(result.get("data") or {"error": result.get("error")})
        finally:
            cancel_event.set()

    return StreamingResponse(generate(), media_type="application/json")
