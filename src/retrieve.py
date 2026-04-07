import argparse
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_path):
    """
    Load text chunks and their metadata from a JSONL file.
    Each line in the input file is expected to be a valid JSON object containing
    at least a ``"chunk_text"`` field. The function collects all chunk texts into
    a list and preserves the full JSON object for each line as metadata.
    Args:
        chunks_path (str): Path to a UTF-8 encoded JSON Lines file where each line
            represents one chunk object.
    Returns:
        tuple[list[str], list[dict]]: A tuple containing:
            - texts: List of values from each object's ``"chunk_text"`` key.
            - metadata: List of full parsed JSON objects, one per line.
    """
    texts = []
    metadata = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["chunk_text"])
            metadata.append(obj)

    return texts, metadata


def load_faiss_index(index_path):
    """
    Load and return a FAISS index from disk.
    Args:
        index_path (str): Absolute or relative path to the serialized FAISS index file.
    Returns:
        faiss.Index: The deserialized FAISS index instance.
    """
    return faiss.read_index(index_path)


def embed_query(model, query, normalize=False):
    """
    Generate an embedding vector for a single query string.
    Args:
        model: Embedding model instance exposing an ``encode`` method compatible with
            ``model.encode([query], convert_to_numpy=True)``.
        query (str): Input text to embed.
        normalize (bool, optional): If ``True``, applies L2 normalization in-place
            to the resulting embedding using ``faiss.normalize_L2``. Defaults to ``False``.
    Returns:
        numpy.ndarray: Query embedding as a NumPy array (shape typically ``(1, dim)``),
        optionally L2-normalized.
    """
    embedding = model.encode([query], convert_to_numpy=True)

    if normalize:
        faiss.normalize_L2(embedding)

    return embedding


def search(index, query_vector, top_k):
    """
    Searches a vector index for the nearest neighbors of a query vector.
    Args:
        index: A vector index object exposing a `search(query_vector, top_k)` method
            (e.g., a FAISS index).
        query_vector: The query embedding(s) used for similarity search. Expected to
            be in the shape required by the underlying index (commonly 2D).
        top_k (int): The maximum number of nearest results to retrieve.
    Returns:
        tuple: A pair `(scores, indices)` corresponding to the first query in the
        batch:
            - scores: Similarity/distance scores for the top-k matches.
            - indices: Integer indices of the top-k matched vectors in the index.
    Notes:
        This function returns `scores[0]` and `indices[0]`, so it is intended for
        single-query usage even if a batched query array is provided.
    """
    scores, indices = index.search(query_vector, top_k)
    return scores[0], indices[0]


def retrieve(query, model, index, metadata, top_k=5, normalize=False):
    """
    Retrieve the most relevant metadata chunks for a given query using vector search.
    This function embeds the input query, searches the provided index for the top
    matching vectors, and returns a list of result objects containing similarity
    scores and corresponding metadata entries.
    Args:
        query (str): The input text query to search for.
        model: Embedding model used by `embed_query` to encode the query.
        index: Search index consumed by `search` to perform nearest-neighbor lookup.
        metadata (Sequence[Any]): Collection of chunk metadata aligned by index
            position with vectors stored in `index`.
        top_k (int, optional): Maximum number of nearest neighbors to retrieve.
            Defaults to 5.
        normalize (bool, optional): Whether to normalize the query embedding before
            search. Defaults to False.
    Returns:
        list[dict]: A list of dictionaries, each with:
            - "score" (float): Similarity/distance score returned by the search.
            - "chunk": Metadata entry from `metadata` for the matched index.
    Notes:
        Entries with index `-1` are skipped (treated as invalid/no match).
    """
    query_vec = embed_query(model, query, normalize=normalize)
    scores, indices = search(index, query_vec, top_k)

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "chunk": metadata[idx]
        })

    return results

def save_results_to_json(query, results, output_path="retrieval_results.json"):
    """
    Save a query and its retrieval results to a JSON file.
    This function creates a new entry containing the input query and a normalized
    list of result items (score, chunk metadata, and chunk text). It then appends
    the entry to an existing JSON file if present; otherwise, it creates a new list
    and writes it to disk.
    Args:
        query (str): The user query associated with the retrieved results.
        results (list[dict]): Retrieval results where each item is expected to
            include:
            - "score": Relevance score for the result.
            - "chunk" (dict): Metadata/content dictionary with optional keys:
              "chunk_id", "doc_id", "page_number", and "chunk_text".
        output_path (str, optional): Path to the JSON file used for persistence.
            Defaults to "retrieval_results.json".
    Behavior:
        - Reads existing JSON content from `output_path` if available.
        - Initializes an empty list when the file does not exist or contains invalid JSON.
        - Appends the new entry and rewrites the full JSON array to the file.
        - Prints a confirmation message with the output path.
    """
    entry = {
        "query": query,
        "results": []
    }

    for r in results:
        entry["results"].append({
            "score": r["score"],
            "chunk_id": r["chunk"].get("chunk_id"),
            "doc_id": r["chunk"].get("doc_id"),
            "page_number": r["chunk"].get("page_number"),
            "chunk_text": r["chunk"].get("chunk_text")
        })

    # Append to file if exists
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")

def print_results(query, results, up_to=None):
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Score: {r['score']}")
        print(f"Doc ID: {r['chunk'].get('doc_id')}")
        print(f"Page: {r['chunk'].get('page_number')}")
        print(f"Text:\n{r['chunk'].get('chunk_text')[:up_to]}...") if up_to else print(f"Text:\n{r['chunk'].get('chunk_text')}")
    print("\n")


def run_manual_tests(model, index, metadata, top_k, normalize):
    """Run 10–15 manual test queries"""
    test_queries = [
        "What are the main causes of cardiovascular disease?",
        "How does AI impact radiology workflows?",
        "What are the regulatory challenges for AI in Europe?",
        "Explain dataset shift in medical AI systems",
        "What are common biases in clinical datasets?",
        "How is machine learning used in diagnostic imaging?",
        "What are safety concerns in AI-assisted diagnosis?",
        "What is the role of validation in medical AI?",
        "How do hospital IT systems affect AI deployment?",
        "What are ethical concerns in medical AI?",
        "How does generalization affect model performance?",
        "What are key performance metrics in radiology AI?"
    ]

    for query in test_queries:
        results = retrieve(query, model, index, metadata, top_k, normalize)
        print_results(query, results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index-path", required=True)
    parser.add_argument("--chunks-path", required=True)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="retrieval_results.json")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--normalize", action="store_true",
                        help="Use if FAISS index was built with cosine similarity")

    args = parser.parse_args()

    print("Loading model...")
    model = SentenceTransformer(args.model_name)

    print("Loading FAISS index...")
    index = load_faiss_index(args.index_path)

    print("Loading chunks...")
    _, metadata = load_chunks(args.chunks_path)

    if args.run_tests:
        run_manual_tests(model, index, metadata, args.top_k, args.normalize)

    elif args.query:
        results = retrieve(
            args.query,
            model,
            index,
            metadata,
            args.top_k,
            args.normalize
        )
        save_results_to_json(args.query, results, args.output_path)
        print_results(args.query, results)

    else:
        print("Provide either --query or --run-tests")


if __name__ == "__main__":
    print("Starting retrieval script...")
    main()