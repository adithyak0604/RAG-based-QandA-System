# 📚 RAG Q&A System

A Retrieval-Augmented Generation (RAG) application that turns any document
(PDF, DOCX, PPTX, or TXT) into an interactive, auto-graded quiz. Upload a
document, generate MCQ / MSQ / Objective / Subjective questions grounded
in its actual content, answer them in a Streamlit UI, and get instant
scoring with per-question feedback — with every question traceable back
to the exact passage it came from.

Built as an end-to-end systems project: document ingestion, chunking,
local embeddings, vector search, structured LLM generation with schema
validation, and LLM-as-judge grading — with a routing layer that runs
entirely free/local during development and swaps to hosted APIs for
production deployment without touching business logic.

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, TXT, normalized into a
  common chunk/page structure
- **Semantic search** — sentence-aware chunking (no mid-sentence cuts) +
  local embeddings + persistent vector storage (ChromaDB)
- **Four question types** — MCQ, MSQ, Objective, and Subjective, each
  generated with a strict JSON schema and validated before use
- **Automatic grading** — exact match for MCQ/MSQ, normalized-match with
  LLM fallback for Objective, LLM-as-judge rubric scoring for Subjective
- **Source references** — every question links back to the exact chunk
  and page it was generated from
- **Question history** — every generated set is saved and browsable per
  document, so nothing is lost between sessions
- **Dual LLM backend** — local Ollama (free) or OpenRouter (hosted),
  switchable per-task or manually overridden from the UI
- **Streamlit GUI** — upload, generate, take the quiz, and review history,
  all in one app

## Tech stack

| Layer | Choice |
|---|---|
| Document parsing | `pdfplumber`, `python-docx`, `python-pptx` |
| Chunking | Custom sentence-aware splitter with overlap |
| Embeddings | `sentence-transformers` (local, `all-MiniLM-L6-v2`) |
| Vector store | `ChromaDB` (persistent local mode) |
| LLM inference | `Ollama` (local) + `OpenRouter` (hosted), via OpenAI-compatible SDK |
| Question/answer storage | `SQLite` |
| UI | `Streamlit` |

## Project structure

```
RAG-based-QandA-System/
|── app.py                            # Streamlit GUI — main entry point
|── config.py                         # Central settings, env-based routing
|── ingest_pipeline.py                # CLI: ingestion + retrieval
|── generate_questions_pipeline.py    # CLI: question generation driver
|──ingestion/
    └── loaders.py                      # PDF/DOCX/PPTX/TXT -> normalized pages
|── processing/
    └── chunking.py                     # Sentence-aware chunking with overlap
    └── embeddings.py                   # Local embedding generation
    └── question_schemas.py             # Per-type schema + prompt instructions
    └── question_generation.py          # LLM question generation + validation
    └── evaluation.py                   # Per-type answer grading
    └── answer_query.py                 # Direct Q&A over retrieved chunks
    └── json_utils.py                   # Shared LLM JSON-output parsing
|── storage/
    └── vector_store.py                 # ChromaDB wrapper (chunks, retrieval)
    └── question_store.py                # SQLite (persisted question sets)
|── llm/
    └── client.py                       # Ollama/OpenRouter routing + retries
|── data/
    └── sample_docs/                    # Sample document for testing
```

## Architecture

```
Document (PDF/DOCX/PPTX/TXT)
        │
        ▼
  ingestion/loaders.py        →  normalized page dicts
        │
        ▼
  processing/chunking.py      →  sentence-aware chunks with overlap
        │
        ▼
  processing/embeddings.py    →  local embedding vectors
        │
        ▼
  storage/vector_store.py     →  ChromaDB (persistent, per-document)
        │
        ├──► processing/question_generation.py  →  MCQ/MSQ/Objective/Subjective (JSON schema)
        │           │
        │           ▼
        │    storage/question_store.py  →  SQLite, browsable in History tab
        │
        └──► processing/answer_query.py  →  direct Q&A over the document

  User answers (Streamlit UI)
        │
        ▼
  processing/evaluation.py    →  per-type grading + LLM-as-judge → score
```

All LLM calls funnel through `llm/client.py`, which routes each task to
either Ollama or OpenRouter based on environment and task type — this is
the one file that changes when moving from local dev to a hosted
deployment.

## Setup

```bash
git clone https://github.com/adithyak0604/RAG-based-QandA-System.git
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment template and fill in your own values:

```bash
cp .env.example .env
```

At minimum, set `OPENROUTER_API_KEY` in `.env` if you want to use
OpenRouter (get one at https://openrouter.ai/keys). If you're running
Ollama locally instead, install it separately and pull a model:

```bash
ollama pull llama3.1:8b
ollama serve
```

First run downloads the local embedding model (~90MB) — needs internet
once, then works offline.

## Usage

### GUI (recommended)

```bash
streamlit run app.py
```

Opens at `localhost:8501`. Walks through: **Upload & Ingest** → **Generate
Questions** → **Answer Questions**, plus a **History** page in the
sidebar for revisiting past question sets.

### CLI

```bash
# Ingest a document
python3 ingest_pipeline.py path/to/document.pdf

# Ingest and test retrieval
python3 ingest_pipeline.py path/to/document.pdf --query "your question"

# Get a direct synthesized answer instead of raw chunks
python3 ingest_pipeline.py path/to/document.pdf --query "your question" --answer

# Generate a mixed question set (document must already be ingested)
python3 generate_questions_pipeline.py document.pdf --mcq 3 --msq 1 --objective 2 --subjective 1
```


## Design notes

- **No stopword removal for embeddings.** Modern embedding models rely on
  full sentence context; stripping stopwords hurts retrieval quality.
  It's only useful for classical keyword extraction (TF-IDF/KeyBERT),
  which this project doesn't currently do.
- **Chunking respects sentence boundaries**, never cutting mid-sentence,
  which keeps embeddings and generated questions coherent.
- **Environment-aware by design.** `config.py`, `processing/embeddings.py`,
  and `llm/client.py` are structured so that moving from local dev to a
  hosted deployment (e.g. AWS with hosted embeddings + OpenRouter-only
  routing) is a small, localized change — not a rewrite.
- **Grading isn't naive string matching.** Objective answers fall back to
  an LLM judge for phrasing/order differences; Subjective answers are
  scored against a rubric generated alongside the question, not graded
  against a single fixed answer.

## Environment variables

See `.env.example` for the full list. Key ones:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Required for OpenRouter-routed tasks |
| `OPENROUTER_MODEL` | Which OpenRouter model to use (avoid `:free` models — see note below) |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Local Ollama endpoint/model |
| `LLM_BACKEND` | Force `auto` / `ollama` / `openrouter` regardless of task |
| `ENV` | `development` or `production` — production forces all LLM calls through OpenRouter |

**Note on OpenRouter free-tier models:** models with a `:free` suffix sit
behind shared, often-saturated capacity and can fail under load. A cheap
paid model (e.g. `anthropic/claude-3.5-haiku`) is far more reliable for
consistent use.

## Known limitations / roadmap

- Audio answer input (Whisper transcription) — not yet implemented, text
  answers only
- Hosted embeddings for production — `processing/embeddings.py` currently
  only supports local embeddings; a hosted-API branch (OpenAI/Voyage) is
  needed before deploying to a server without local model access
- Local storage (ChromaDB + SQLite) is file-based — a production
  deployment should migrate to RDS/pgvector so data persists across
  container restarts
- No containerization/deployment config yet (Dockerfile, ECS/EC2 setup)

## License

MIT (or update to your preference).
