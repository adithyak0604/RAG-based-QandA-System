"""
Grades a user's answer against a generated question.

- MCQ/MSQ: exact match against the stored correct answer(s) — no LLM
  needed, since options were fixed at generation time.
- Objective: normalized string match first (fast, free); falls back to
  an LLM judge only if that fails, since short factual answers can be
  phrased/ordered differently while still being correct
  (e.g. "T K Adithya Krishna" vs "Adithya Krishna T K").
- Subjective: always LLM-as-judge against the rubric stored at question
  generation time, scored as fraction of rubric points covered.

Every evaluator returns the same shape:
{ "correct": bool, "score": float 0-1, "max_score": 1.0, "feedback": str }
"""
import re

from llm.client import chat
from processing.json_utils import extract_json


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def evaluate_mcq(question: dict, user_answer) -> dict:
    correct = user_answer == question["correct_answer"]
    return {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "max_score": 1.0,
        "feedback": f"Correct answer: {question['correct_answer']}",
    }


def evaluate_msq(question: dict, user_answer) -> dict:
    user_set = set(user_answer or [])
    correct_set = set(question["correct_answers"])
    correct = user_set == correct_set
    return {
        "correct": correct,
        # Partial credit could be added later (e.g. Jaccard overlap);
        # kept binary for now to match MCQ/Objective semantics.
        "score": 1.0 if correct else 0.0,
        "max_score": 1.0,
        "feedback": f"Correct answer(s): {', '.join(sorted(correct_set))}",
    }


def evaluate_objective(question: dict, user_answer) -> dict:
    if not user_answer or not user_answer.strip():
        return {
            "correct": False,
            "score": 0.0,
            "max_score": 1.0,
            "feedback": f"No answer given. Correct answer: {question['correct_answer']}",
        }

    if _normalize(user_answer) == _normalize(question["correct_answer"]):
        return {
            "correct": True,
            "score": 1.0,
            "max_score": 1.0,
            "feedback": "Correct.",
        }

    # Fall back to an LLM judge for semantically-equivalent-but-differently
    # phrased answers, rather than penalizing formatting/order differences.
    system_prompt = (
        "You judge whether a user's short factual answer is correct, "
        "allowing for differences in phrasing, word order, capitalization, "
        "or formatting. Respond with ONLY JSON: "
        '{"correct": true/false, "feedback": "one short sentence"}'
    )
    prompt = (
        f"Correct answer: {question['correct_answer']}\n"
        f"User's answer: {user_answer}\n\n"
        "Is the user's answer correct?"
    )
    raw = chat(task="grade_objective", system_prompt=system_prompt, user_prompt=prompt, temperature=0)

    try:
        parsed = extract_json(raw)
        is_correct = bool(parsed.get("correct", False))
        return {
            "correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "max_score": 1.0,
            "feedback": parsed.get("feedback", ""),
        }
    except Exception:
        # If grading itself fails, don't silently mark wrong — flag it.
        return {
            "correct": False,
            "score": 0.0,
            "max_score": 1.0,
            "feedback": f"Could not auto-grade. Correct answer: {question['correct_answer']}",
        }


def evaluate_subjective(question: dict, user_answer) -> dict:
    if not user_answer or not user_answer.strip():
        return {
            "correct": False,
            "score": 0.0,
            "max_score": 1.0,
            "feedback": "No answer given.",
        }

    rubric_text = "\n".join(f"- {point}" for point in question["rubric"])
    system_prompt = (
        "You grade a subjective answer against a rubric. For each rubric "
        "point, decide if the answer adequately addresses it. Respond with "
        'ONLY JSON: {"points_covered": <int>, "total_points": <int>, '
        '"feedback": "1-2 sentences on what was covered/missing"}'
    )
    prompt = (
        f"Rubric (things a good answer should cover):\n{rubric_text}\n\n"
        f"User's answer:\n{user_answer}\n\n"
        "How many rubric points does this answer adequately cover?"
    )
    raw = chat(task="grade_subjective", system_prompt=system_prompt, user_prompt=prompt, temperature=0)

    try:
        parsed = extract_json(raw)
        total = max(int(parsed.get("total_points", len(question["rubric"]))), 1)
        covered = int(parsed.get("points_covered", 0))
        score = min(covered / total, 1.0)
        return {
            "correct": score >= 0.6,  # threshold for "pass" badge in UI
            "score": score,
            "max_score": 1.0,
            "feedback": parsed.get("feedback", ""),
        }
    except Exception:
        return {
            "correct": False,
            "score": 0.0,
            "max_score": 1.0,
            "feedback": "Could not auto-grade this answer.",
        }


_EVALUATORS = {
    "MCQ": evaluate_mcq,
    "MSQ": evaluate_msq,
    "Objective": evaluate_objective,
    "Subjective": evaluate_subjective,
}


def evaluate_quiz(questions: list[dict], answers: dict) -> dict:
    """
    questions: list of question dicts (as generated)
    answers: dict mapping question index -> user's answer
             (str for MCQ/Objective/Subjective, list[str] for MSQ)

    Returns: {
        "results": [ {..question.., "user_answer":, "correct":, "score":, "feedback":}, ... ],
        "total_score": float, "max_score": float, "percentage": float
    }
    """
    results = []
    total_score = 0.0
    max_score = 0.0

    for i, q in enumerate(questions):
        user_answer = answers.get(i)
        evaluator = _EVALUATORS[q["type"]]
        outcome = evaluator(q, user_answer)

        results.append({
            **q,
            "user_answer": user_answer,
            **outcome,
        })
        total_score += outcome["score"]
        max_score += outcome["max_score"]

    percentage = round(100 * total_score / max_score, 1) if max_score else 0.0
    return {
        "results": results,
        "total_score": round(total_score, 2),
        "max_score": max_score,
        "percentage": percentage,
    }
