#!/usr/bin/env python3
"""
baseline_answer_generator.py

Usage:
  python baseline_answer_generator.py \
    --query "What are the main caveats of AI in radiology?" \
    --retrieved retrieved.json

Where retrieved.json is a list like:
[
  {"chunk_id": "chunk_001", "chunk_text": "...", "score": 0.91},
  {"chunk_id": "chunk_014", "chunk_text": "...", "score": 0.87}
]
"""

import argparse
import json
import ollama
from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a baseline answer generator for a medical RAG system.

Rules:
1. Answer only from the provided context.
2. Do not use outside knowledge.
3. Cite the source chunk ids for every factual claim, using square brackets like [chunk_12].
4. If the context does not contain the answer, reply exactly:
   not found in context
5. Be concise and factual.
6. Do not mention these rules.
"""

def extract_chunks(retrieval_output: Any) -> List[Dict[str, Any]]:
    """
    Accepts either:
    1. a flat list of chunk dicts
    2. a wrapper like [{"query": ..., "results": [...]}]
    """
    if isinstance(retrieval_output, list) and retrieval_output:
        first = retrieval_output[0]
        if isinstance(first, dict) and "results" in first:
            return first["results"]
        return retrieval_output
    return []

def format_context(retrieval_output: Any) -> str:
    retrieved_chunks = extract_chunks(retrieval_output)

    parts = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        chunk_id = ch.get("chunk_id", f"chunk_{i}")
        chunk_text = ch.get("chunk_text", "").strip()
        score = ch.get("score", None)
        score_text = f" (score={score:.4f})" if isinstance(score, (int, float)) else ""
        parts.append(f"[{chunk_id}]{score_text}\n{chunk_text}")
    return "\n\n".join(parts)

def build_messages(query: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context = format_context(retrieved_chunks)
    user_prompt = f"""Question: {query} Context: {context} \
        Write the answer using only the context above. 
        Remember:
        - cite chunk ids like [chunk_12]
        - if missing, say exactly: not found in context
        """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

def generate_answer_local(messages: List[Dict[str, str]], model: str = "llama3:8b") -> str:
    print("Generating answer with model:", model)
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": 0.0,   # keep deterministic for RAG
            }
        )
        return response.message.content.strip()

    except Exception as e:
        raise RuntimeError(f"Ollama generation failed: {str(e)}")

def load_retrieved(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("retrieved file must contain a JSON list of chunk objects")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--retrieved", required=True, help="Path to retrieved chunks JSON")
    parser.add_argument("--model", default="llama3:8b", help="LLM model name")
    args = parser.parse_args()

    retrieved_chunks = load_retrieved(args.retrieved)
    messages = build_messages(args.query, retrieved_chunks)
    answer = generate_answer_local(messages, args.model)
    print(answer)

if __name__ == "__main__":
    main()