"""Build a reproducible FAISS index from chunked medical text.

Input
-----
A JSONL file where each line has at least:
    - chunk_id
    - doc_id
    - page_number
    - chunk_text

Outputs
-------
1. A saved FAISS index file
2. A metadata JSONL file that maps vector row ids to source chunks
3. A config JSON file recording the exact indexing setup

Example
-------
python medical_rag_faiss_indexer.py \
  --input chunks.jsonl \
  --output-dir artifacts/index \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 64 \
  --normalize

Notes
-----
- To keep the pipeline reproducible, the script sorts records by chunk_id.
- The metadata file is written in the same order as the FAISS vectors.
- With --normalize enabled, embeddings are L2-normalized and stored in an IP index,
  which makes inner product equivalent to cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class IndexConfig:
    input_path: str
    output_dir: str
    model_name: str
    batch_size: int
    normalize: bool
    sort_key: str
    faiss_index_type: str
    embedding_dim: int


REQUIRED_FIELDS = {"chunk_id", "doc_id", "page_number", "chunk_text"}


def set_reproducible_seed(seed: int = 42) -> None:
    """
    Set deterministic seeds for common Python randomness sources.
    This utility initializes reproducibility settings by seeding:
    - `random` (Python standard library RNG)
    - `numpy.random` (NumPy RNG)
    - `PYTHONHASHSEED` (hash randomization for Python objects)
    Args:
        seed (int, optional): Seed value used across all configured randomness
            sources. Defaults to `42`.
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        path (Path): Path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: List of dictionaries representing the JSONL rows.
    Raises:
        ValueError: If any line in the file is not valid JSON or is missing required fields.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"Missing required fields on line {line_no}: {sorted(missing)}")
            rows.append(row)
    return rows


def stable_sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return a stably sorted copy of row dictionaries by their ``chunk_id`` value.
    The sort key is computed as ``str(row["chunk_id"])``, so all ``chunk_id`` values
    are compared lexicographically as strings. Python's sort is stable, meaning rows
    with equal stringified ``chunk_id`` values keep their original relative order.
    Args:
        rows: A list of dictionaries, each expected to contain a ``"chunk_id"`` key.
    Returns:
        A new list of dictionaries sorted by stringified ``chunk_id``.
    """
    return sorted(rows, key=lambda r: str(r["chunk_id"]))


def batched(iterable: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """
    Yield successive fixed-size batches from a list of dictionaries.
    Args:
        iterable (List[Dict[str, Any]]): The input sequence to split into batches.
        batch_size (int): Number of items per batch.
    Yields:
        List[Dict[str, Any]]: Consecutive slices of `iterable` with at most
        `batch_size` elements each. The final batch may be smaller.
    Raises:
        ValueError: If `batch_size` is less than 1.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def build_embeddings(rows: List[Dict[str, Any]], model_name: str, batch_size: int, normalize: bool) -> np.ndarray:
    """
    Build dense vector embeddings for a list of text chunks using a SentenceTransformer model.
    Args:
        rows (List[Dict[str, Any]]): Input records containing a ``"chunk_text"`` field for each row.
        model_name (str): Name or path of the SentenceTransformer model to load.
        batch_size (int): Number of texts to encode per batch.
        normalize (bool): Whether to L2-normalize embeddings during encoding.
    Returns:
        np.ndarray: A 2D NumPy array of shape ``(len(rows), embedding_dim)`` with dtype ``float32``.
    Raises:
        ValueError: If the produced embeddings are not a 2D array.
    """
    model = SentenceTransformer(model_name)
    texts = [row["chunk_text"] for row in rows]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray, normalize: bool) -> faiss.Index:
    """
    Build and populate a FAISS index from embedding vectors.
    Creates either:
    - `faiss.IndexFlatIP` when `normalize=True` (for cosine-like similarity with pre-normalized vectors), or
    - `faiss.IndexFlatL2` when `normalize=False` (for Euclidean distance).
    The index dimension is inferred from `embeddings.shape[1]`, and all provided embeddings are added to the index.
    Args:
        embeddings (np.ndarray): 2D array of shape `(n_vectors, dim)` containing embedding vectors.
        normalize (bool): If `True`, use inner-product index; otherwise use L2 index.
    Returns:
        faiss.Index: A FAISS index containing the input embeddings.
    """
    dim = embeddings.shape[1]
    if normalize:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def write_metadata(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write per-chunk metadata records to a JSON Lines file.
    Each input row is assigned a sequential `vector_id` based on its position
    in `rows` (starting at 0). For every row, a JSON object is written as a
    single line to `output_path` with the following fields:
    - `vector_id`: Enumerated index of the row.
    - `chunk_id`: Chunk identifier from the row.
    - `doc_id`: Source document identifier from the row.
    - `page_number`: Page number associated with the chunk.
    - `chunk_text`: Text content of the chunk.
    Args:
        rows: List of metadata dictionaries. Each dictionary is expected to
            contain the keys: `chunk_id`, `doc_id`, `page_number`, and
            `chunk_text`.
        output_path: Destination path for the JSONL output file. The file is
            opened in write mode (`"w"`) with UTF-8 encoding and will be
            overwritten if it already exists.
    Returns:
        None.
    """
    with output_path.open("w", encoding="utf-8") as f:
        for vector_id, row in enumerate(rows):
            record = {
                "vector_id": vector_id,
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "page_number": row["page_number"],
                "chunk_text": row["chunk_text"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_config(config: IndexConfig, output_path: Path) -> None:
    """
    Write an ``IndexConfig`` instance to a JSON file.
    Serializes ``config`` using :func:`dataclasses.asdict` and writes it to
    ``output_path`` with UTF-8 encoding, pretty-printed using an indentation of 2
    spaces, followed by a trailing newline.
    Args:
        config: The index configuration object to serialize.
        output_path: Destination path for the generated JSON file.
    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible FAISS index from JSONL chunks.")
    parser.add_argument("--input", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to write the FAISS index and metadata")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name or local path")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings and use FAISS Inner Product for cosine-like retrieval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    rows = stable_sort_rows(rows)

    embeddings = build_embeddings(rows, args.model, args.batch_size, args.normalize)
    index = build_faiss_index(embeddings, args.normalize)

    index_path = output_dir / "faiss.index"
    metadata_path = output_dir / "chunk_metadata.jsonl"
    config_path = output_dir / "index_config.json"

    faiss.write_index(index, str(index_path))
    write_metadata(rows, metadata_path)
    write_config(
        IndexConfig(
            input_path=str(input_path),
            output_dir=str(output_dir),
            model_name=args.model,
            batch_size=args.batch_size,
            normalize=args.normalize,
            sort_key="chunk_id",
            faiss_index_type="IndexFlatIP" if args.normalize else "IndexFlatL2",
            embedding_dim=int(embeddings.shape[1]),
        ),
        config_path,
    )

    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved metadata to:    {metadata_path}")
    print(f"Saved config to:      {config_path}")
    print(f"Indexed chunks:       {len(rows)}")
    print(f"Embedding dim:        {embeddings.shape[1]}")


if __name__ == "__main__":
    main()

