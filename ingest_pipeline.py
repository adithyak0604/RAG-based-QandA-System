"""
End-to-end ingestion pipeline: file -> pages -> chunks -> embeddings -> vector store.

Usage:
    python ingest_pipeline.py path/to/document.pdf
    python ingest_pipeline.py path/to/document.pdf --query "some question"
"""
import argparse
import sys

from config import settings
from ingestion.loaders import load_document
from processing.chunking import chunk_document
from storage.vector_store import add_chunks, query as query_store


def ingest(filepath: str) -> int:
    print(f"[1/3] Loading {filepath} ...")
    pages = load_document(filepath)
    print(f"      -> {len(pages)} page(s)/section(s) extracted")

    print(f"[2/3] Chunking (target {settings.chunk_size_tokens} tokens, "
          f"{settings.chunk_overlap_tokens} overlap) ...")
    chunks = chunk_document(
        pages,
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    print(f"      -> {len(chunks)} chunk(s) created")

    print("[3/3] Embedding + storing (local sentence-transformers model, "
          "first run downloads the model) ...")
    add_chunks(chunks)
    print(f"      -> done. Stored at {settings.vector_store_path}")

    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to PDF/DOCX/PPTX/TXT file")
    parser.add_argument("--query", help="Optional: test retrieval with a query", default=None)
    parser.add_argument("--answer", action="store_true",
                         help="Generate a direct LLM answer instead of showing raw chunks "
                              "(requires --query, and Ollama running locally)")
    parser.add_argument("--chunk-size", type=int, default=None,
                         help="Override chunk size in tokens (default: 500)")
    parser.add_argument("--overlap", type=int, default=None,
                         help="Override chunk overlap in tokens (default: 75)")
    args = parser.parse_args()

    if args.chunk_size is not None:
        settings.chunk_size_tokens = args.chunk_size
    if args.overlap is not None:
        settings.chunk_overlap_tokens = args.overlap

    ingest(args.filepath)

    if args.query and args.answer:
        from processing.answer_query import answer_query
        print(f"\nQuery: {args.query!r}")
        result = answer_query(args.query)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {result['sources']}")
    elif args.query:
        print(f"\nQuery: {args.query!r}")
        hits = query_store(args.query, n_results=3)
        for i, hit in enumerate(hits, start=1):
            print(f"\n--- Result {i} (distance={hit['distance']:.4f}) ---")
            print(f"Source: {hit['metadata']['source_file']}, "
                  f"page {hit['metadata']['page_number']}")
            print(hit["text"][:300] + ("..." if len(hit["text"]) > 300 else ""))
