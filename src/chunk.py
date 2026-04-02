#!/usr/bin/env python3
"""
Chunk a document corpus into overlapping token windows.

Input:
- metadata JSON file with records containing:
  doc_id, title, page_number, source_path, file_type, text
- one text file per document: <doc_id>.txt

Output:
- JSONL or CSV with:
  chunk_text, doc_id, page_number, chunk_id
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
    - JSON array
    - JSONL (one JSON object per line)
    """
    text = metadata_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    records = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_tokenizer():
    """
    Use tiktoken if available; otherwise fall back to whitespace tokenization.
    """
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")

        def encode(text: str) -> List[int]:
            return enc.encode(text)

        def decode(tokens: List[int]) -> str:
            return enc.decode(tokens)

        return encode, decode
    except Exception:
        # Fallback: approximate tokenization by whitespace
        def encode(text: str) -> List[str]:
            return text.split()

        def decode(tokens: List[str]) -> str:
            return " ".join(tokens)

        return encode, decode


def chunk_tokens(tokens: List[Any], chunk_size: int, overlap: int) -> List[List[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        if end >= len(tokens):
            break
        start += step

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON/JSONL file")
    parser.add_argument("--text_dir", required=True, help="Directory containing <doc_id>.txt files")
    parser.add_argument("--output", required=True, help="Output file path (.jsonl or .csv)")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    text_dir = Path(args.text_dir)
    output_path = Path(args.output)

    records = load_metadata(metadata_path)
    encode, decode = get_tokenizer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()

    rows = []

    for rec in records:
        doc_id = str(rec.get("doc_id", "")).strip()
        if not doc_id:
            continue

        txt_path = text_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            # Skip missing documents, but keep going
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        tokens = encode(text)
        pieces = chunk_tokens(tokens, args.chunk_size, args.overlap)

        page_number = rec.get("page_number", None)

        for i, piece in enumerate(pieces, start=1):
            chunk_text = decode(piece).strip()
            if not chunk_text:
                continue

            row = {
                "chunk_id": f"{doc_id}_chunk_{i:05d}",
                "doc_id": doc_id,
                "page_number": page_number,
                "chunk_text": chunk_text,
            }
            rows.append(row)

    if ext == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif ext == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["chunk_id", "doc_id", "page_number", "chunk_text"],
            )
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("Output file must end with .jsonl or .csv")

    print(f"Wrote {len(rows)} chunks to {output_path}")


if __name__ == "__main__":
    print('Starting chunking process...')
    main()