import json
import os
from typing import Any, Dict, List, Tuple

import anthropic
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent.parent / ".env")

DOCS_PATH = "docs/documents.json"
QUESTIONS_PATH = "evals/questions.json"
MODEL_NAME = "claude-sonnet-4-6"
TOP_K = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def build_doc_index(documents: List[Dict[str, str]], model: SentenceTransformer) -> np.ndarray:
    corpus = [f"{doc['title']}: {doc['text']}" for doc in documents]
    return embed_texts(corpus, model)


def retrieve_docs(
    question: str,
    documents: List[Dict[str, str]],
    doc_embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
) -> List[Tuple[Dict[str, str], float]]:
    question_embedding = embed_texts([question], model)
    similarities = cosine_similarity(question_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(documents[i], float(similarities[i])) for i in top_indices]


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

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    doc_embeddings = build_doc_index(documents, model)
    print(f"Indexed {len(documents)} documents.\n")

    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)

    results = []

    for i, question_obj in enumerate(questions, start=1):
        question = question_obj["question"]

        retrieved_docs = retrieve_docs(question, documents, doc_embeddings, model, top_k=TOP_K)
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