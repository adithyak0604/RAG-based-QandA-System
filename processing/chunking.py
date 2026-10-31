"""
Chunking splits page text into overlapping, semantically coherent pieces
for embedding. Important: we do NOT strip stopwords here. Embedding
models are trained on natural language; stopword removal actively hurts
retrieval quality. Stopword removal only belongs in the separate
keyword-extraction pass (see processing/keywords.py, added later).

Chunking respects sentence boundaries so we don't cut a sentence in half
mid-thought, which would produce a worse embedding.
"""
import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    # Good enough for structured documents; swap for spaCy's sentencizer
    # later if you hit messy edge cases (abbreviations, bullet lists).
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _approx_token_count(text: str) -> int:
    # Cheap approximation (~0.75 words per token for English) so we don't
    # need a tokenizer dependency just for chunk sizing.
    return int(len(text.split()) / 0.75)


def chunk_page(
    page: dict, chunk_size_tokens: int = 500, overlap_tokens: int = 75
) -> list[dict]:
    """
    Takes one page dict (from a loader) and returns a list of chunk dicts:
    { "text": str, "page_number": int, "source_file": str, "chunk_index": int }
    """
    sentences = _split_sentences(page["text"])
    chunks = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0

    def flush():
        nonlocal chunk_index
        if not current_sentences:
            return
        chunk_text = " ".join(current_sentences)
        chunks.append(
            {
                "text": chunk_text,
                "page_number": page["page_number"],
                "source_file": page["source_file"],
                "chunk_index": chunk_index,
            }
        )
        chunk_index += 1

    for sentence in sentences:
        sent_tokens = _approx_token_count(sentence)

        if current_tokens + sent_tokens > chunk_size_tokens and current_sentences:
            flush()
            # Build overlap: carry the tail sentences forward so context
            # isn't lost at the boundary.
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                overlap_count += _approx_token_count(s)
                overlap_sentences.insert(0, s)
                if overlap_count >= overlap_tokens:
                    break
            current_sentences = overlap_sentences
            current_tokens = overlap_count

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    flush()
    return chunks


def chunk_document(pages: list[dict], chunk_size_tokens=500, overlap_tokens=75) -> list[dict]:
    """Chunk every page of a document and return a flat list of chunks."""
    all_chunks = []
    for page in pages:
        all_chunks.extend(
            chunk_page(page, chunk_size_tokens, overlap_tokens)
        )
    return all_chunks
