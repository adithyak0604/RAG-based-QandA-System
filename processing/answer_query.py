"""
This is the "generation" half of RAG. Retrieval alone (storage/vector_store.py)
only returns the most relevant raw chunks — useful for question generation
later, but not a direct answer to a question like "what is the name of the
person?". This module takes those chunks and asks an LLM to actually answer
using only that context.
"""
from llm.client import chat
from storage.vector_store import query as retrieve_chunks

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the answer isn't in the context, say so clearly instead of guessing. "
    "Be direct and concise — a short factual answer, not a summary of the context."
)


def answer_query(question: str, n_chunks: int = 4, collection_name: str = "documents") -> dict:
    hits = retrieve_chunks(question, n_results=n_chunks, collection_name=collection_name)

    if not hits:
        return {"answer": "No relevant content found in the document.", "sources": []}

    context = "\n\n---\n\n".join(h["text"] for h in hits)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    answer = chat(
        task="answer_query",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,  # factual answers, not creative ones
    )

    sources = [
        {"source_file": h["metadata"]["source_file"], "page": h["metadata"]["page_number"]}
        for h in hits
    ]
    return {"answer": answer, "sources": sources}
