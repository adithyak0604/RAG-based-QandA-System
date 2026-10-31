"""
Persists generated question sets so they survive app restarts and can be
browsed later, instead of vanishing once the Streamlit session ends.

Uses SQLite (stdlib, zero setup) for now. When you move to AWS, this is
the natural candidate to swap for RDS Postgres — the function signatures
below (save/list/get/delete) are the interface the rest of the app should
keep using, so that swap stays localized to this file.
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = os.getenv("QUESTION_STORE_PATH", "./data/question_sets.db")


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS question_sets (
            id TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            created_at TEXT NOT NULL,
            counts_json TEXT NOT NULL,
            questions_json TEXT NOT NULL
        )
    """)
    return conn


def save_question_set(source_file: str, questions: list[dict]) -> str:
    """Saves a generated question set under its source document.
    Returns the new set's id."""
    counts: dict = {}
    for q in questions:
        counts[q["type"]] = counts.get(q["type"], 0) + 1

    set_id = str(uuid.uuid4())
    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT INTO question_sets (id, source_file, created_at, counts_json, questions_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                set_id,
                source_file,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(counts),
                json.dumps(questions),
            ),
        )
    conn.close()
    return set_id


def list_question_sets(source_file: str | None = None) -> list[dict]:
    """Returns metadata (not full questions) for saved sets, newest
    first. Pass source_file to filter to one document, or None for all."""
    conn = _get_conn()
    if source_file:
        rows = conn.execute(
            "SELECT id, source_file, created_at, counts_json FROM question_sets "
            "WHERE source_file = ? ORDER BY created_at DESC",
            (source_file,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, source_file, created_at, counts_json FROM question_sets "
            "ORDER BY created_at DESC"
        ).fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "source_file": r[1],
            "created_at": r[2],
            "counts": json.loads(r[3]),
            "total": sum(json.loads(r[3]).values()),
        }
        for r in rows
    ]


def get_question_set(set_id: str) -> list[dict] | None:
    """Returns the full list of question dicts for a saved set."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT questions_json FROM question_sets WHERE id = ?", (set_id,)
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def delete_question_set(set_id: str) -> None:
    conn = _get_conn()
    with conn:
        conn.execute("DELETE FROM question_sets WHERE id = ?", (set_id,))
    conn.close()
