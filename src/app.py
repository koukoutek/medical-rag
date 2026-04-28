from __future__ import annotations

import uvicorn
import faiss

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .generate import build_messages, generate_answer_local
from .retrieve import load_chunks, retrieve

# -----------------------------
# Config
# -----------------------------
INDEX_PATH = Path("/Users/konstantinos/Projects/rag-assistant/artifacts/index/faiss.index")
CHUNKS_PATH = Path("/Users/konstantinos/Projects/rag-assistant/data/processed/chunks.jsonl")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

app = FastAPI(title="Medical RAG API", version="0.1.0")

# -----------------------------
# Models
# -----------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: Optional[str] = None
    page_number: Optional[int] = None
    score: float
    chunk_text: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[RetrievedChunk]


# -----------------------------
# Globals loaded at startup
# -----------------------------
embedder: Optional[SentenceTransformer] = None
index: Optional[faiss.Index] = None
chunks_metadata: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Helpers
# -----------------------------
def make_citations(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    for ch in retrieved_chunks:
        citations.append({
            "chunk_id": ch.get("chunk_id"),
            "doc_id": ch.get("doc_id"),
            "page_number": ch.get("page_number"),
            "score": ch.get("score"),
        })
    return citations


def normalize_retrieved_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for item in results:
        ch = item.get("chunk", {}) or {}

        normalized.append({
            "chunk_id": ch.get("chunk_id") or ch.get("id") or ch.get("chunkId") or "unknown_chunk",
            "doc_id": ch.get("doc_id"),
            "page_number": ch.get("page_number"),
            "score": float(item.get("score", 0.0)),
            "chunk_text": ch.get("chunk_text") or ch.get("text") or ch.get("content") or "",
        })

    return normalized


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def startup_event() -> None:
    global embedder, index, chunks_metadata # chunks_by_id

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_PATH}")

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    _, chunks_metadata = load_chunks(CHUNKS_PATH)


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Medical RAG API is running"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# @app.post("/ask", response_model=AskResponse)
# def ask(req: AskRequest) -> AskResponse:
#     try:
#         retrieved = retrieve(req.question, embedder, index, chunks_metadata, top_k=TOP_K)
#         messages = build_messages(req.question, retrieved)
#         answer = generate_answer_local(messages)
#         citations = make_citations(retrieved)

#         return AskResponse(
#             answer=answer,
#             citations=citations,
#             retrieved_chunks=[RetrievedChunk(**ch) for ch in retrieved],
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        raw_results = retrieve(
            query=req.question,
            model=embedder,
            index=index,
            metadata=chunks_metadata,
            top_k=TOP_K,
            normalize=True,
        )

        retrieved = normalize_retrieved_results(raw_results)
        messages = build_messages(req.question, retrieved)
        answer = generate_answer_local(messages)
        citations = make_citations(retrieved)

        return AskResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=[RetrievedChunk(**ch) for ch in retrieved],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
