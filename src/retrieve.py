import argparse
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_path):
    """
    Load chunks from JSONL file.
    Returns:
        texts: list of chunk_text
        metadata: list of full chunk dicts
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
    """Load FAISS index from disk"""
    return faiss.read_index(index_path)


def embed_query(model, query, normalize=False):
    """Embed a single query"""
    embedding = model.encode([query], convert_to_numpy=True)

    if normalize:
        faiss.normalize_L2(embedding)

    return embedding


def search(index, query_vector, top_k):
    """Search FAISS index"""
    scores, indices = index.search(query_vector, top_k)
    return scores[0], indices[0]


def retrieve(query, model, index, metadata, top_k=5, normalize=False):
    """Full retrieval pipeline"""
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
    Save retrieved results to a JSON file.
    Appends results if file already exists.
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