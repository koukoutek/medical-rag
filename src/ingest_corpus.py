#!/usr/bin/env python3
"""
Ingest a corpus from data/raw/, extract text, and write processed outputs to data/processed/.

Outputs:
- data/processed/documents.jsonl   -> one record per page (PDF) or per file (text)
- data/processed/text/             -> cleaned plain-text files per document

Supported inputs:
- .pdf
- .txt
- .md
- .text

Install:
    pip install pypdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

from pypdf import PdfReader


INPUT_EXTS = {".pdf", ".txt", ".text"}


@dataclass
class Record:
    doc_id: str
    title: str
    page_number: Optional[int]
    source_path: str
    file_type: str
    text: str


def clean_text(text: str) -> str:
    """
    Clean and normalize text for downstream chunking.

    Args:
        text: Raw text string to clean, typically from PDF or document extraction.

    Returns:
        Cleaned and normalized text ready for chunking and downstream NLP tasks.
    """
    text = text.replace("\x00", " ")
    # Fix common PDF hyphenation across line breaks: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Normalize line endings and spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_title_from_pdf(reader: PdfReader, path: Path) -> str:
    """
    Extract or infer a title from a PDF document. If metadata is not available
    or does not contain a valid title, falls back to using the filename stem.
    
    Args:
        reader (PdfReader): A PdfReader object representing the PDF document.
        path (Path): A Path object pointing to the PDF file.
    
    Returns:
        str: The PDF title from metadata, or the filename stem if no title is found.
    
    """
    meta = reader.metadata
    if meta:
        title = getattr(meta, "title", None) or meta.get("/Title")
        if title and str(title).strip():
            return str(title).strip()
    return path.stem


def infer_title_from_text(path: Path, text: str) -> str:
    """
    Extract a title from text content, using the first non-empty line as preference. 
    If no non-empty lines are found, falls back to using the filename stem from the 
    provided path.
    
    Args:
        path: A Path object representing the file, used as fallback for title generation.
        text: The text content to extract a title from.
    
    Returns:
        A string representing the inferred title, truncated to 200 characters maximum
        if derived from text, or the filename stem if no suitable text is found.
    """
    # Use first non-empty line as a lightweight title guess, else filename.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return path.stem


def iter_input_files(raw_dir: Path) -> Iterable[Path]:
    """
    Iterate over input files in a directory recursively.
    
    Args:
        raw_dir: The root directory to search for input files.
    
    Yields:
        Path: Absolute paths to files with extensions in INPUT_EXTS.
    """
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in INPUT_EXTS:
            yield p


def process_pdf(path: Path) -> List[Record]:
    """
    Extract and process text content from a PDF file.
    
    Reads a PDF file from the given path, extracts text from each page,
    cleans the extracted text, and creates Record objects containing the
    processed content along with metadata.
    
    Args:
        path (Path): The file system path to the PDF file to process.
    
    Returns:
        List[Record]: A list of Record objects, one for each page in the PDF.
            Each Record contains:
            - doc_id: Stem of the filename
            - title: Inferred title from PDF metadata or filename
            - page_number: Page number (1-indexed)
            - source_path: Full file path as string
            - file_type: "pdf"
            - text: Cleaned text content from the page
    
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        PdfReadError: If the PDF file cannot be read or is corrupted.
    """
    reader = PdfReader(str(path))
    title = infer_title_from_pdf(reader, path)
    records: List[Record] = []

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        records.append(
            Record(
                doc_id=path.stem,
                title=title,
                page_number=i,
                source_path=str(path),
                file_type="pdf",
                text=text,
            )
        )
    return records


def process_text_file(path: Path) -> List[Record]:
    """
    Process a text file and extract its content into a structured Record.
    
    Reads a text file, cleans its content, infers a title, and returns a list
    containing a single Record object with the file's metadata and processed text.
    
    Args:
        path (Path): The file path to the text file to process.
    
    Returns:
        List[Record]: A list containing a single Record object with:
            - doc_id: The file stem (filename without extension)
            - title: Inferred title from the file content or metadata
            - page_number: None (not applicable for text files)
            - source_path: The string representation of the file path
            - file_type: The file extension (without the leading dot, lowercase)
            - text: The cleaned text content of the file
    
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        UnicodeDecodeError: Potential errors are ignored during encoding.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(raw)
    title = infer_title_from_text(path, raw)
    return [
        Record(
            doc_id=path.stem,
            title=title,
            page_number=None,
            source_path=str(path),
            file_type=path.suffix.lower().lstrip("."),
            text=text,
        )
    ]


def write_jsonl(records: List[Record], out_path: Path) -> None:
    """
    Write a list of records to a JSONL (JSON Lines) file.
    
    Each record is converted to a dictionary and written as a single JSON object
    per line in the output file. The output directory is created if it does not exist.
    
    Args:
        records: A list of Record objects to be written to the file.
        out_path: The Path object specifying the output file location.
    
    Returns:
        None
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_text_files(records: List[Record], out_dir: Path) -> None:
    """
    Write records to text files organized by document ID.
    
    Groups records by their document ID, sorts them by page number (if available),
    and writes the combined text content to individual files in the output directory.
    Pages are prefixed with a header indicating the page number, while non-paged
    text is appended as-is.
    
    Args:
        records: A list of Record objects to be written to files.
        out_dir: The Path object specifying the output directory where text files
                 will be created. The directory is created if it doesn't exist.
    
    Returns:
        None
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    for r in records:
        grouped.setdefault(r.doc_id, []).append(r)

    for doc_id, items in grouped.items():
        # Sort pages first, keep text files as-is
        items = sorted(items, key=lambda x: (x.page_number is None, x.page_number or 0))
        parts = []
        for r in items:
            if r.page_number is not None:
                parts.append(f"\n\n=== PAGE {r.page_number} ===\n\n{r.text}")
            else:
                parts.append(r.text)

        out_file = out_dir / f"{doc_id}.txt"
        out_file.write_text(clean_text("\n".join(parts)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract raw text and metadata from PDFs/text files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Input corpus folder")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output folder")
    parser.add_argument("--mode", choices=["jsonl", "text", "both"], default="both", help="Output format")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir

    if not raw_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {raw_dir}")

    all_records: List[Record] = []

    for path in iter_input_files(raw_dir):
        if path.suffix.lower() == ".pdf":
            all_records.extend(process_pdf(path))
        else:
            all_records.extend(process_text_file(path))

    if args.mode in {"jsonl", "both"}:
        write_jsonl(all_records, out_dir / "documents.jsonl")

    if args.mode in {"text", "both"}:
        write_text_files(all_records, out_dir / "text")

    print(f"Processed {len(all_records)} records into {out_dir}")


if __name__ == "__main__":
    main()