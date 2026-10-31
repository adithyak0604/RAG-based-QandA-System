"""
Defines what a valid question looks like for each type, and the prompt
instructions that tell the LLM how to produce it. Keeping schema and
prompt together per type makes it easy to add a new question type later
without touching the generation logic itself.
"""

# Fields every question type shares
BASE_FIELDS = {"type", "question", "source_chunk_id", "source_file", "page_number", "source_excerpt"}

# Fields required per type, on top of BASE_FIELDS
REQUIRED_FIELDS = {
    "MCQ": {"options", "correct_answer"},
    "MSQ": {"options", "correct_answers"},
    "Objective": {"correct_answer"},
    "Subjective": {"rubric"},
}

TYPE_INSTRUCTIONS = {
    "MCQ": (
        "Generate a multiple-choice question with exactly 4 options, "
        "only ONE of which is correct. "
        'Output field "options": array of 4 strings. '
        '"correct_answer": the exact text of the correct option.'
    ),
    "MSQ": (
        "Generate a multiple-select question with 4 to 5 options, where "
        "TWO OR MORE options are correct. "
        'Output field "options": array of strings. '
        '"correct_answers": array of the exact text of all correct options.'
    ),
    "Objective": (
        "Generate a short-answer factual question with one unambiguous "
        "correct answer (a name, date, number, or short phrase — not a "
        "full sentence). "
        '"correct_answer": the short correct answer as a string.'
    ),
    "Subjective": (
        "Generate an open-ended question that requires a multi-sentence "
        "explanation to answer well. "
        '"rubric": array of 3-5 short strings, each describing one thing '
        "a correct answer should mention or explain (used for grading later)."
    ),
}


def validate_question(q: dict, qtype: str) -> tuple[bool, str]:
    """Returns (is_valid, error_message)."""
    if q.get("type") != qtype:
        return False, f"expected type={qtype}, got {q.get('type')!r}"

    missing = (BASE_FIELDS | REQUIRED_FIELDS[qtype]) - q.keys()
    if missing:
        return False, f"missing fields: {missing}"

    if qtype == "MCQ":
        if len(q["options"]) != 4:
            return False, "MCQ must have exactly 4 options"
        if q["correct_answer"] not in q["options"]:
            return False, "correct_answer not found in options"

    if qtype == "MSQ":
        if len(q["options"]) < 4:
            return False, "MSQ must have at least 4 options"
        if len(q["correct_answers"]) < 2:
            return False, "MSQ must have at least 2 correct answers"
        if not set(q["correct_answers"]).issubset(set(q["options"])):
            return False, "some correct_answers not found in options"

    return True, ""
