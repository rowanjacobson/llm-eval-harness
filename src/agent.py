import json
import os
import re
from typing import Any, Dict, List, Tuple

import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DOCS_PATH = "docs/documents.json"
QUESTIONS_PATH = "evals/questions.json"
MODEL_NAME = "claude-sonnet-4-6"
TOP_K = 3


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def score_doc(question: str, doc: Dict[str, str]) -> int:
    question_tokens = set(tokenize(question))
    doc_tokens = set(tokenize(doc["title"] + " " + doc["text"]))
    return len(question_tokens.intersection(doc_tokens))


def retrieve_docs(question: str, documents: List[Dict[str, str]], top_k: int = 3) -> List[Tuple[Dict[str, str], int]]:
    scored = []
    for doc in documents:
        score = score_doc(question, doc)
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def build_context(retrieved_docs: List[Tuple[Dict[str, str], int]]) -> str:
    parts = []
    for doc, score in retrieved_docs:
        parts.append(
            f"<document>\n"
            f"<doc_id>{doc['doc_id']}</doc_id>\n"
            f"<title>{doc['title']}</title>\n"
            f"<text>{doc['text']}</text>\n"
            f"</document>"
        )
    return "\n\n".join(parts)


def ask_claude(question: str, context: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set in your environment.")

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = (
        "You are an internal knowledge assistant.\n"
        "Answer only from the provided documents.\n"
        "If the answer is not supported by the documents, say you do not know.\n"
        "Return valid JSON with this exact schema:\n"
        "{\n"
        '  "answer": "string",\n'
        '  "citations": ["doc_id"],\n'
        '  "abstain": true_or_false,\n'
        '  "uncertainty": "low|medium|high"\n'
        "}"
    )

    user_prompt = (
        f"Here are the documents:\n\n{context}\n\n"
        f"Question: {question}"
    )

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=500,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    # Anthropic returns content blocks; for a simple text response this is usually the first text block.
    return response.content[0].text


RESULTS_PATH = "results.json"

def main() -> None:
    documents = load_json(DOCS_PATH)
    questions = load_json(QUESTIONS_PATH)

    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)

    results = []

    for i, question_obj in enumerate(questions, start=1):
        question = question_obj["question"]

        retrieved_docs = retrieve_docs(question, documents, top_k=TOP_K)
        context = build_context(retrieved_docs)
        answer = ask_claude(question, context)

        results.append({
            "question_number": i,
            "question_id": question_obj["question_id"],
            "question": question,
            "answer": answer,
        })

        print(f"\n--- Question {i} ({question_obj['question_id']}) ---")
        print(f"Q: {question}")
        print(f"A: {answer}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()