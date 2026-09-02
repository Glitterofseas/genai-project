"""Offline embedding step: the job description PDF into a Chroma index.

    python -m app.modules.embedding.build_index

Run once. Idempotent - a re-run replaces the collection rather than duplicating
it - and the only place embeddings are paid for; the chat loop never re-embeds.
"""
from __future__ import annotations

import argparse
import re

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

from ..config.settings import CHROMA_DIR, JOB_DESCRIPTION_PDF, get_settings

COLLECTION = "job_description"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120


def extract_pages(pdf_path=JOB_DESCRIPTION_PDF) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # The PDF renders bullets as U+FFFD - strip them from the embedded text.
        text = text.replace("�", "-")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(text.strip())
    return pages


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware chunking, falling back to a sliding window."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= size:
            buffer = f"{buffer}\n\n{paragraph}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        if len(paragraph) <= size:
            buffer = paragraph
        else:
            for start in range(0, len(paragraph), size - overlap):
                piece = paragraph[start:start + size].strip()
                if piece:
                    chunks.append(piece)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def build(verbose: bool = True) -> int:
    settings = get_settings()
    if not settings.has_api_key:
        raise SystemExit("OPENAI_API_KEY is not set - add it to .env")

    pages = extract_pages()
    documents, metadatas, ids = [], [], []
    for page_number, page_text in enumerate(pages, start=1):
        for chunk_number, piece in enumerate(chunk(page_text)):
            documents.append(piece)
            metadatas.append(
                {"source": JOB_DESCRIPTION_PDF.name, "page": page_number, "chunk": chunk_number}
            )
            ids.append(f"p{page_number}-c{chunk_number}")

    embeddings = OpenAIEmbeddings(
        model=settings.embed_model, api_key=settings.openai_api_key
    )
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    # Idempotent: drop anything from a previous run before writing.
    existing = store.get().get("ids", [])
    if existing:
        store.delete(ids=existing)
    store.add_texts(texts=documents, metadatas=metadatas, ids=ids)

    if verbose:
        print(f"Embedded {len(documents)} chunks from {len(pages)} pages")
        print(f"  model      : {settings.embed_model}")
        print(f"  collection : {COLLECTION}")
        print(f"  persisted  : {CHROMA_DIR}")
        total_chars = sum(len(d) for d in documents)
        print(f"  ~{total_chars:,} characters (~{total_chars // 4:,} tokens) embedded once")
    return len(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    build(verbose=not args.quiet)
