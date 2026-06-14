import sqlite3
from pathlib import Path

import myhansard
from fastapi import FastAPI
from myhansard.rag import answer
from pydantic import BaseModel

app = FastAPI()

DB_PATH = Path("data/hansard.db")
CHROMA_PATH = Path("data/chroma")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
collection = myhansard.get_collection(CHROMA_PATH)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query(request: QueryRequest):
    result = answer(request.query, collection, conn)
    return result
