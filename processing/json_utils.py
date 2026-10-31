"""
Shared helper for parsing JSON out of LLM responses. LLMs sometimes wrap
JSON in markdown fences or add stray text around it despite instructions
not to — this strips/extracts around that.
"""
import json
import re


def extract_json(raw: str):
    """Parses a JSON object or array from raw LLM text, tolerating
    markdown code fences and minor surrounding text."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
