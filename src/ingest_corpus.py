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
    """Basic cleanup suitable for downstream chunking."""
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
    meta = reader.metadata
    if meta:
        title = getattr(meta, "title", None) or meta.get("/Title")
        if title and str(title).strip():
            return str(title).strip()
    return path.stem


def infer_title_from_text(path: Path, text: str) -> str:
    # Use first non-empty line as a lightweight title guess, else filename.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return path.stem


def iter_input_files(raw_dir: Path) -> Iterable[Path]:
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in INPUT_EXTS:
            yield p


def process_pdf(path: Path) -> List[Record]:
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_text_files(records: List[Record], out_dir: Path) -> None:
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