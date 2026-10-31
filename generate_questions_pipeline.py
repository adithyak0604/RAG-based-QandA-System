"""
Generates a set of questions across all four types for a document that's
already been ingested (run ingest_pipeline.py first).

Usage:
    python3 generate_questions_pipeline.py Resume.pdf --mcq 2 --msq 1 --objective 2 --subjective 1
"""
import argparse
import json

from processing.question_generation import generate_questions
from storage.vector_store import get_chunks_by_source


def generate_question_set(source_file: str, counts: dict, collection_name: str = "documents") -> list[dict]:
    chunks = get_chunks_by_source(source_file, collection_name)
    if not chunks:
        raise ValueError(
            f"No chunks found for '{source_file}'. Did you run "
            f"ingest_pipeline.py on it first?"
        )

    print(f"Found {len(chunks)} chunk(s) for {source_file}")

    all_questions = []
    for qtype, count in counts.items():
        if count <= 0:
            continue
        print(f"\nGenerating {count} {qtype} question(s)...")
        generated = 0
        chunk_i = 0
        # Spread questions across different chunks rather than piling
        # them all onto chunk 0, so questions cover the whole document.
        while generated < count and chunk_i < len(chunks) * 2:
            chunk = chunks[chunk_i % len(chunks)]
            qs = generate_questions(chunk, qtype, num_questions=1)
            all_questions.extend(qs)
            generated += len(qs)
            chunk_i += 1
        print(f"  -> got {generated}/{count}")

    return all_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help="Filename as stored (e.g. Resume.pdf)")
    parser.add_argument("--mcq", type=int, default=0)
    parser.add_argument("--msq", type=int, default=0)
    parser.add_argument("--objective", type=int, default=0)
    parser.add_argument("--subjective", type=int, default=0)
    parser.add_argument("--output", default="questions_output.json")
    args = parser.parse_args()

    counts = {
        "MCQ": args.mcq,
        "MSQ": args.msq,
        "Objective": args.objective,
        "Subjective": args.subjective,
    }

    questions = generate_question_set(args.source_file, counts)

    with open(args.output, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nSaved {len(questions)} question(s) to {args.output}")
