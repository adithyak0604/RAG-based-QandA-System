"""
LLM client routing. Both Ollama and OpenRouter speak the OpenAI-compatible
API, so we use one SDK (openai) pointed at different base_urls.

Routing rule (see config.py):
- development: cheap/bulk tasks -> local Ollama, quality-sensitive tasks
  (question generation, grading, answer synthesis) -> OpenRouter.
- production: everything -> OpenRouter (no dependency on your PC being on).

Call sites never construct a client directly — always go through
`chat()` below, passing a `task` name so routing stays centralized.
"""
import time

from functools import lru_cache

from openai import OpenAI

from config import settings


@lru_cache(maxsize=2)
def _get_ollama_client() -> OpenAI:
    return OpenAI(base_url=settings.ollama_base_url, api_key="ollama")  # key unused by Ollama


@lru_cache(maxsize=2)
def _get_openrouter_client() -> OpenAI:
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before using "
            "quality-sensitive tasks (question generation, grading)."
        )
    return OpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)


def _resolve_client_and_model(task: str):
    # Manual override wins over everything, including production routing.
    # This is meant for dev convenience (e.g. switching off a slow local
    # model) — leave it at "auto" for real deployments so production
    # correctly forces OpenRouter everywhere.
    if settings.llm_backend_override == "ollama":
        return _get_ollama_client(), settings.ollama_model
    if settings.llm_backend_override == "openrouter":
        return _get_openrouter_client(), settings.openrouter_model

    if settings.env == "production":
        return _get_openrouter_client(), settings.openrouter_model

    if task in settings.cheap_tasks:
        return _get_ollama_client(), settings.ollama_model

    return _get_openrouter_client(), settings.openrouter_model


def chat(
    task: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> str:
    """
    Single entry point for every LLM call in the app.

    task: a short label like "summarize", "generate_mcq", "answer_query" —
          used only for routing (see config.cheap_tasks), not sent to the model.

    Retries on transient upstream errors (common with free-tier OpenRouter
    models, which sit behind shared, often-saturated capacity). Retrying
    a paid/reliable model rarely triggers this path at all.
    """
    client, model = _resolve_client_and_model(task)

    last_error = None
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        if getattr(response, "choices", None):
            content = response.choices[0].message.content
            if content and content.strip():
                return content

        # No usable choices — likely a transient upstream error (rate
        # limit, provider overload). Back off and retry a few times
        # before giving up.
        error_info = getattr(response, "error", None) or response
        last_error = error_info
        if attempt < max_retries:
            wait = 2 ** attempt  # 2s, 4s, 8s
            print(f"[warn] LLM call attempt {attempt}/{max_retries} failed "
                  f"({error_info}). Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"LLM call for task='{task}' failed after {max_retries} attempts "
        f"(backend={model}). Last error: {last_error}\n"
        "Common causes: invalid/misspelled model name, no OpenRouter "
        "credits, sustained rate limiting on a free-tier model, or "
        "(if using Ollama) the model isn't pulled locally yet "
        "(`ollama pull <model>`). If you're on a `:free` OpenRouter model, "
        "consider switching to a cheap paid model for reliability."
    )
