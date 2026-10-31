"""
Embeddings layer. Two paths, switched on settings.env:

- development: sentence-transformers, running locally, free.
- production: OpenAI's hosted embeddings API. No local model to load —
  this matters on free-tier hosting (like Streamlit Community Cloud),
  where RAM is limited and loading a torch model in-process is exactly
  the kind of thing that pushes memory over the limit.

IMPORTANT: local and hosted embeddings produce vectors of different
dimensions (384 for all-MiniLM-L6-v2 vs 1536 for text-embedding-3-small).
A ChromaDB collection is locked to whichever dimension first wrote to it,
so you can't mix — a collection built in dev (local) can't be queried in
production (hosted) or vice versa. This is fine in practice: local dev
data and a deployed app's data are separate environments anyway, each
with its own fresh vector store.
"""
from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def _get_local_model():
    # Loaded once and cached — loading the model is the slow part,
    # not the embedding calls themselves. Only imported here (not at
    # module level) so production deployments never pull in torch/
    # sentence-transformers at all.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.local_embedding_model)


@lru_cache(maxsize=1)
def _get_openai_client():
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Hosted embeddings (used in "
            "production) require a separate OpenAI API key — OpenRouter "
            "does not serve an embeddings endpoint."
        )
    return OpenAI(api_key=settings.openai_api_key)


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _get_local_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def _embed_hosted(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
    response = client.embeddings.create(model=settings.hosted_embedding_model, input=texts)
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Takes a list of raw strings, returns a list of embedding vectors
    (plain Python lists of floats, so they're easy to store/serialize).
    """
    if settings.env == "production":
        return _embed_hosted(texts)
    return _embed_local(texts)


def embed_query(text: str) -> list[float]:
    """Convenience wrapper for embedding a single query string."""
    return embed_texts([text])[0]
