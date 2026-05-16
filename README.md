# Medical RAG 

## 1. Corpus Ingestion 

### Overview

This script processes a collection of research articles stored in `data/raw/` and converts them into a structured format suitable for downstream tasks such as text splitting, embedding, and retrieval (e.g., RAG systems).

It supports both PDF and plain text files, extracting their contents and standardizing them into clean, machine-readable outputs.

---

### What the Script Does

#### 1. Load Input Files
- Recursively scans `data/raw/` for supported file types:
  - PDF (`.pdf`)
  - Text-based files (`.txt`, `.md`, `.text`)

---

#### 2. Extract Raw Text
- **PDFs**:
  - Reads each page using `pypdf`
  - Extracts text page by page
- **Text files**:
  - Reads full file content directly

---

#### 3. Clean Text
Applies lightweight normalization to improve downstream processing:
- Removes null characters
- Fixes broken words caused by line breaks (e.g., `exam-\nple → example`)
- Normalizes whitespace and line breaks
- Removes excessive empty lines

---

#### 4. Extract Metadata
For each document (or page in PDFs), the script stores:
- `doc_id` → filename (without extension)
- `title` → extracted from PDF metadata or inferred from text
- `page_number` → page index (PDFs only)
- `source_path` → original file location
- `file_type` → file format
- `text` → cleaned content

---

#### 5. Generate Outputs

All outputs are written to `data/processed/`:

##### JSONL File (for pipelines)
- `documents.jsonl`
- One record per:
  - page (PDFs)
  - file (text)
- Ready for chunking and embedding

##### Clean Text Files (for inspection/debugging)
- `data/processed/text/<doc_id>.txt`
- Full document text with page separators for PDFs

---

### Why This Format

- **Page-level granularity** improves retrieval precision
- **JSONL structure** integrates easily with vector databases and LLM pipelines
- **Cleaned text** reduces noise during chunking and embedding
- **Metadata** enables traceability and filtering during retrieval

---

### Usage

```bash
pip install pypdf
python ingest_corpus.py --raw-dir data/raw --out-dir data/processed --mode both
```


## 2. Document Chunking Pipeline

### Overview

This script processes a corpus of documents by splitting extracted text into smaller, overlapping chunks suitable for downstream tasks such as retrieval-augmented generation (RAG), embeddings, or search indexing.

The pipeline assumes:
- A metadata file (JSON or JSONL) describing each document
- A corresponding `.txt` file for each document containing extracted text

---

### Input Data

#### Metadata File

Each record must contain:

- `doc_id` — unique document identifier  
- `title` — document title  
- `page_number` — page reference (if applicable)  
- `source_path` — original file path  
- `file_type` — e.g., pdf, txt  
- `text` — (optional, not used if `.txt` files exist)

Supported formats:
- JSON (array of objects)
- JSONL (one object per line)

#### Text Files

Each document must have a corresponding text file: <doc_id>.txt

Example: 

---

### Chunking Strategy

- **Chunk size:** 500 tokens  
- **Overlap:** 50 tokens  
- **Tokenizer:**  
  - Uses `tiktoken` (if available)  
  - Falls back to whitespace splitting otherwise  

This ensures:
- Context preservation across chunks  
- Compatibility with LLM token limits  

---

### Output Format

The script generates a chunked dataset in:

- **JSONL** (recommended for pipelines), or  
- **CSV**

Each chunk contains:

- `chunk_id` — unique identifier (`doc_id_chunk_XXXXX`)  
- `doc_id` — source document ID  
- `page_number` — original page reference  
- `chunk_text` — chunk content  

#### Example (JSONL)

```json
{"chunk_id":"DOC123_chunk_00001","doc_id":"DOC123","page_number":4,"chunk_text":"..."}
```


### Usage

#### Generate JSONL Output

```bash
python chunk_documents.py --metadata data/raw/metadata.json --text_dir data/raw/texts --output data/processed/chunks.jsonl
```

#### Generate CSV Output

```bash
python chunk_documents.py --metadata data/raw/metadata.json --text_dir data/raw/texts --output data/processed/chunks.csv
```



## 3. Medical RAG FAISS Indexing Pipeline

This repository contains a Python script that turns a JSONL file of text chunks into a reproducible FAISS vector index for medical retrieval-augmented generation (RAG).

### What the script does

The script reads a `chunks.jsonl` file where each record contains:

* `chunk_id`
* `doc_id`
* `page_number`
* `chunk_text`

It then:

1. loads all chunks from the JSONL file
2. sorts them deterministically by `chunk_id`
3. creates an embedding for each chunk
4. stores the vectors in a FAISS index
5. writes a metadata file that maps each vector back to its source chunk
6. saves the exact indexing configuration used for the run

### Why this pipeline is reproducible

The pipeline is designed to be repeatable across runs:

* chunks are sorted before embedding so vector order is stable
* a fixed random seed is set
* the embedding model name is recorded in the config file
* the batch size and normalization setting are saved
* the metadata file is written in the same order as the FAISS vectors

This means vector `0` in the index always corresponds to line `0` in the metadata file, vector `1` to line `1`, and so on.

### Input format

The input must be a JSONL file. Each line should look like this:

```json
{"chunk_id": "c001", "doc_id": "doc_01", "page_number": 3, "chunk_text": "..."}
```

Only the required fields are used by the script, but the chunk text must be present for embedding.

### Output files

The script writes three files into the output directory:

* `faiss.index` — the saved FAISS vector index
* `chunk_metadata.jsonl` — one metadata record per vector
* `index_config.json` — the exact configuration used to build the index

#### `chunk_metadata.jsonl`

Each line contains:

* `vector_id`
* `chunk_id`
* `doc_id`
* `page_number`
* `chunk_text`

This file is what you use to trace retrieval results back to the original chunk.

### Embedding model

The script uses a `SentenceTransformer` model by default:

```bash
sentence-transformers/all-MiniLM-L6-v2
```

This is a strong lightweight baseline for semantic retrieval. It is fast, easy to run locally, and works well for small to medium corpora.

### Similarity setup

The script supports two FAISS modes:

* **L2 distance** when embeddings are not normalized
* **Inner product** when `--normalize` is enabled

When normalization is enabled, inner product behaves like cosine similarity.

### How to run

Install dependencies:

```bash
pip install faiss-cpu sentence-transformers numpy
```

Run the indexing script:

```bash
python medical_rag_faiss_indexer.py \
  --input chunks.jsonl \
  --output-dir artifacts/index \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 64 \
  --normalize
```

### Example output structure

```text
artifacts/index/
├── faiss.index
├── chunk_metadata.jsonl
└── index_config.json
```

### Notes for medical RAG

For medical text, this pipeline gives you a reliable baseline. A stronger domain-specific embedding model may improve retrieval quality, but the overall indexing process stays the same.

### Summary

This script provides a simple, reproducible way to turn chunked medical text into a searchable FAISS index with a metadata file that preserves the link between each vector and its source chunk.
