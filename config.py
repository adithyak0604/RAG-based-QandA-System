"""
Central configuration for the RAG Q&A system.

Everything that changes between "running on my PC" and "deployed on AWS"
lives here, driven by the ENV variable. Nothing else in the codebase
should hardcode a model name or API base_url.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Loads variables from a local .env file (if present) into the process
# environment. In production (AWS), .env won't exist — env vars are set
# directly by ECS/EC2 instead, and load_dotenv() silently does nothing.
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

ENV = os.getenv("ENV", "development")  # "development" | "production"


@dataclass
class Settings:
    env: str = ENV

    # --- LLM routing ---
    # In dev: cheap/bulk tasks -> local Ollama, quality tasks -> OpenRouter.
    # In production: everything -> OpenRouter (no dependency on a home PC).
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"
    )

    # Tasks that are fine on a local model (bulk, low-stakes)
    cheap_tasks: tuple = ("summarize", "keyword_extract")

    # Manual override for LLM backend, on top of the task-based routing
    # above. "auto" = use the cheap_tasks/env logic as designed.
    # "ollama" or "openrouter" = force that backend for EVERY task,
    # regardless of type — useful when Ollama is too slow for testing,
    # or when you want to force free/local generation for a demo.
    llm_backend_override: str = os.getenv(
        "LLM_BACKEND", "auto"
    )  # "auto" | "ollama" | "openrouter"

    # --- Embeddings ---
    # dev: sentence-transformers running locally, free, no network needed.
    # production: hosted OpenAI embeddings — no local model to load, which
    # also keeps the deployed app's memory footprint small (important on
    # free-tier hosting).
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    hosted_embedding_model: str = os.getenv(
        "HOSTED_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # --- Chunking ---
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75

    # --- Storage ---
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    question_store_path: str = os.getenv(
        "QUESTION_STORE_PATH", "./data/question_sets.db"
    )


settings = Settings()
