"""
Vector storage. Using ChromaDB in local persistent mode — no server to
run, just writes to disk at settings.vector_store_path.

This gives us a real ingest -> retrieve loop today. When you move to
AWS, this is the piece most likely to change (e.g. to pgvector on RDS,
or a managed Chroma/Pinecone instance) but the interface below
(add_chunks / query) should stay the same so nothing upstream breaks.
"""
import chromadb

from config import settings
from processing.embeddings import embed_texts, embed_query


def get_collection(name: str = "documents"):
    client = chromadb.PersistentClient(path=settings.vector_store_path)
    return client.get_or_create_collection(name=name)


def add_chunks(chunks: list[dict], collection_name: str = "documents") -> None:
    """
    chunks: list of dicts from processing/chunking.py, each with
    "text", "page_number", "source_file", "chunk_index".
    """
    if not chunks:
        return

    collection = get_collection(collection_name)
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    ids = [
        f"{c['source_file']}_p{c['page_number']}_c{c['chunk_index']}"
        for c in chunks
    ]
    metadatas = [
        {
            "source_file": c["source_file"],
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)


def delete_source(source_file: str, collection_name: str = "documents") -> int:
    """Deletes all chunks belonging to a given source file. Returns the
    number of chunks deleted."""
    collection = get_collection(collection_name)
    existing = collection.get(where={"source_file": source_file})
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def get_chunk_by_id(chunk_id: str, collection_name: str = "documents") -> dict | None:
    """Fetches one chunk's full text + metadata by its stored ID — used
    to show 'source reference' for a generated question."""
    collection = get_collection(collection_name)
    result = collection.get(ids=[chunk_id])
    if not result["ids"]:
        return None
    return {
        "id": result["ids"][0],
        "text": result["documents"][0],
        "source_file": result["metadatas"][0]["source_file"],
        "page_number": result["metadatas"][0]["page_number"],
    }


def list_source_files(collection_name: str = "documents") -> list[str]:
    """Returns all distinct source_file names currently stored — used by
    the GUI to populate a 'which document' dropdown."""
    collection = get_collection(collection_name)
    results = collection.get()
    sources = {m["source_file"] for m in results["metadatas"]}
    return sorted(sources)


def get_chunks_by_source(source_file: str, collection_name: str = "documents") -> list[dict]:
    """
    Returns ALL chunks for a given document, in order. Unlike query(),
    this doesn't rank by similarity to a search string — it's for
    systematically walking through a document (e.g. to generate
    questions from every chunk), not for retrieval.
    """
    collection = get_collection(collection_name)
    results = collection.get(where={"source_file": source_file})

    chunks = []
    for i in range(len(results["ids"])):
        meta = results["metadatas"][i]
        chunks.append(
            {
                "id": results["ids"][i],
                "text": results["documents"][i],
                "source_file": meta["source_file"],
                "page_number": meta["page_number"],
                "chunk_index": meta["chunk_index"],
            }
        )
    chunks.sort(key=lambda c: (c["page_number"], c["chunk_index"]))
    return chunks


def query(text: str, n_results: int = 5, collection_name: str = "documents") -> list[dict]:
    """Returns the top-N most relevant chunks for a query string."""
    collection = get_collection(collection_name)
    query_vector = embed_query(text)

    results = collection.query(query_embeddings=[query_vector], n_results=n_results)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return hits
