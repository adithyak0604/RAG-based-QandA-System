"""
Generates questions from a chunk of source text. One call = one question
type = N questions, all grounded in a single chunk so we can trace every
question back to exactly where it came from (source_chunk_id, page number).

This is a quality-sensitive task, so by default (see config.cheap_tasks)
it routes to OpenRouter, not the local Ollama model.
"""
import json
import re

from llm.client import chat
from processing.question_schemas import TYPE_INSTRUCTIONS, validate_question

SYSTEM_PROMPT = """You are a question-generation engine for an educational \
Q&A system. You generate questions strictly from the provided source text \
— never introduce facts not present in it.

Always respond with ONLY a JSON array. No markdown code fences, no \
preamble, no explanation — just the raw JSON array, parseable by \
json.loads() as-is."""


def _build_user_prompt(chunk_text: str, qtype: str, num_questions: int) -> str:
    instructions = TYPE_INSTRUCTIONS[qtype]
    return f"""Source text:
\"\"\"
{chunk_text}
\"\"\"

Generate exactly {num_questions} question(s) of type "{qtype}" from this text.
{instructions}

Each object in the JSON array must have these fields:
- "type": "{qtype}"
- "question": the question text
- plus the type-specific fields described above

Respond with ONLY the JSON array."""


def _extract_json_array(raw: str) -> list:
    """LLMs sometimes wrap JSON in markdown fences despite instructions.
    Strip those before parsing, and fall back to extracting the first
    [...] block if there's stray text around it."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def generate_questions(chunk: dict, qtype: str, num_questions: int = 1) -> list[dict]:
    """
    chunk: a chunk dict as stored in the vector store, must include
           "text", "chunk_index" (or an id), "source_file", "page_number".
    qtype: one of "MCQ", "MSQ", "Objective", "Subjective"
    Returns a list of validated question dicts. Invalid questions from
    the LLM are dropped (logged), not silently kept.
    """
    if qtype not in TYPE_INSTRUCTIONS:
        raise ValueError(f"Unknown question type: {qtype}")

    prompt = _build_user_prompt(chunk["text"], qtype, num_questions)
    raw_response = chat(
        task=f"generate_{qtype.lower()}",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.5,
    )

    try:
        parsed = _extract_json_array(raw_response)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[warn] Failed to parse LLM output as JSON for {qtype}: {e}")
        print(f"[warn] Raw response was: {raw_response[:300]}")
        return []

    chunk_id = chunk.get("id") or f"{chunk['source_file']}_p{chunk['page_number']}_c{chunk.get('chunk_index', 0)}"

    questions = []
    for q in parsed:
        q.setdefault("type", qtype)
        q["source_chunk_id"] = chunk_id
        q["source_file"] = chunk["source_file"]
        q["page_number"] = chunk["page_number"]
        q["source_excerpt"] = chunk["text"]

        is_valid, error = validate_question(q, qtype)
        if is_valid:
            questions.append(q)
        else:
            print(f"[warn] Dropped invalid {qtype} question: {error}")

    return questions
