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
    Extract chunks from retrieval output.
    Handles multiple retrieval output formats:
    - If output is a list with a dict containing "results" key, returns the results list
    - If output is a list, returns it as-is
    - Otherwise returns an empty list
    Args:
        retrieval_output (Any): The output from a retrieval operation, typically a list or nested dict structure.
    Returns:
        List[Dict[str, Any]]: A list of chunk dictionaries extracted from the retrieval output.
    """
    if isinstance(retrieval_output, list) and retrieval_output:
        first = retrieval_output[0]
        if isinstance(first, dict) and "results" in first:
            return first["results"]
        return retrieval_output
    return []

def format_context(retrieval_output: Any) -> str:
    """
    Format retrieved chunks into a readable context string.
    This function extracts chunks from ``retrieval_output`` using
    ``extract_chunks()`` and renders each chunk as a block containing:
    - the chunk identifier in square brackets,
    - an optional relevance score formatted to 4 decimal places,
    - the chunk text.
    Each chunk block is separated by a blank line.
    Args:
        retrieval_output: Raw retrieval result object containing chunk data in a
            format supported by ``extract_chunks()``.
    Returns:
        A single formatted string containing all retrieved chunks. Returns an
        empty string if no chunks are extracted.
    Expected chunk fields:
        Each extracted chunk is expected to be a mapping that may contain:
        - ``chunk_id``: Identifier for the chunk. Defaults to ``"chunk_{i}"``.
        - ``chunk_text``: Text content of the chunk. Defaults to an empty string.
        - ``score``: Numeric relevance score. Included only if it is an ``int`` or
          ``float``.
    Example output:
        [chunk_1] (score=0.9123)
        Some retrieved text here.
        [chunk_2]
        Another retrieved text block.
    """
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
    """
    Build a list of messages for the RAG assistant.
    This function constructs a structured message list suitable for use with
    a chat-based language model API. It formats the retrieved context chunks
    and combines them with the user query into a prompt that instructs the
    model to answer based solely on the provided context.
    Args:
        query (str): The user's input question to be answered.
        retrieved_chunks (List[Dict[str, Any]]): A list of retrieved document
            chunks, where each chunk is a dictionary containing chunk metadata
            and content used to build the context.
    Returns:
        List[Dict[str, str]]: A list of message dictionaries, each containing
            a 'role' (either 'system' or 'user') and 'content' key, formatted
            for use with a chat completion API. The system message contains
            the predefined SYSTEM_PROMPT, while the user message contains
            the query and formatted context.
    Example:
        >>> chunks = [{"id": "chunk_1", "text": "Paris is the capital of France."}]
        >>> messages = build_messages("What is the capital of France?", chunks)
        >>> messages[0]["role"]
        'system'
        >>> messages[1]["role"]
        'user'
    """    
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
    """
    Generate an answer using a local Ollama model.
    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries containing
            the conversation history, where each dictionary has 'role' and 'content' keys.
        model (str, optional): The name of the Ollama model to use for generation.
            Defaults to "llama3:8b".
    Returns:
        str: The generated answer from the model, stripped of leading/trailing whitespace.
    Raises:
        RuntimeError: If the Ollama generation fails for any reason, wrapping the
            original exception message.
    Example:
        >>> messages = [{"role": "user", "content": "What is the capital of France?"}]
        >>> answer = generate_answer_local(messages, model="llama3:8b")
        >>> print(answer)
        'Paris is the capital of France.'
    """
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
    """
    Load retrieved chunks from a JSON file.
    Args:
        path (str): The file path to the JSON file containing the retrieved chunks.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the retrieved chunk objects.
    Raises:
        ValueError: If the JSON file does not contain a list of chunk objects.
        FileNotFoundError: If the file at the specified path does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    Example:
        >>> chunks = load_retrieved("retrieved_chunks.json")
        >>> print(chunks[0])
        {'id': '1', 'text': 'example chunk text', 'metadata': {...}}
    """
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