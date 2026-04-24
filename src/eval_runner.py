import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from retrieve import load_json, build_doc_index, retrieve_docs, build_context, EMBEDDING_MODEL
from agent import ask_claude
from evals import eval_retrieval, eval_citations, eval_grounding

load_dotenv(Path(__file__).parent.parent / ".env")

DOCS_PATH = Path(__file__).parent.parent / "docs/documents.json"
QUESTIONS_PATH = Path(__file__).parent.parent / "evals/questions.json"
EVALS_DIR = Path(__file__).parent.parent / "evals"
RUN_LABEL = "baseline"
TOP_K = 3


def parse_answer(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def next_version(label: str) -> str:
    pattern = re.compile(rf"results_{re.escape(label)}_v(\d+)\.json")
    versions = []
    for f in EVALS_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            versions.append(int(m.group(1)))
    return f"{label}_v{max(versions) + 1}" if versions else f"{label}_v1"


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    documents = load_json(DOCS_PATH)
    questions = load_json(QUESTIONS_PATH)

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    doc_embeddings = build_doc_index(documents, embed_model)
    print(f"Indexed {len(documents)} documents.\n")

    run_label = next_version(RUN_LABEL)
    results = []

    for i, q_obj in enumerate(questions, start=1):
        question = q_obj["question"]
        gold_doc_ids = q_obj.get("source_doc_ids", [])

        retrieved = retrieve_docs(question, documents, doc_embeddings, embed_model, top_k=TOP_K)
        retrieved_doc_ids = [doc["doc_id"] for doc, _ in retrieved]
        context = build_context(retrieved)

        raw_answer = ask_claude(question, context, client)

        try:
            parsed = parse_answer(raw_answer)
            answer_text = parsed.get("answer", "")
            citations = parsed.get("citations", [])
            abstain = parsed.get("abstain", False)
            uncertainty = parsed.get("uncertainty", "")
        except (json.JSONDecodeError, KeyError):
            parsed = {}
            answer_text = raw_answer
            citations = []
            abstain = False
            uncertainty = ""

        retrieval_eval = eval_retrieval(retrieved_doc_ids, gold_doc_ids)
        citation_eval = eval_citations(citations, retrieved_doc_ids)
        grounding_eval = eval_grounding(question, answer_text, context, client)

        result = {
            "question_number": i,
            "question_id": q_obj["question_id"],
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "abstain": abstain,
            "uncertainty": uncertainty,
            "retrieved_doc_ids": retrieved_doc_ids,
            "gold_doc_ids": gold_doc_ids,
            "retrieval_eval": retrieval_eval,
            "citation_eval": citation_eval,
            "grounding_eval": grounding_eval,
        }
        results.append(result)

        print(f"--- Q{i} ({q_obj['question_id']}) ---")
        print(f"Q: {question}")
        print(f"A: {answer_text}")
        print(f"  retrieval hit={retrieval_eval['hit']}  recall={retrieval_eval['recall']}")
        print(f"  citation_score={citation_eval['citation_score']}  hallucinated={citation_eval['hallucinated_citations']}")
        print(f"  grounded={grounding_eval['grounded']}  reason={grounding_eval['reason']}\n")

    # Summary metrics
    answerable = [r for r in results if r["gold_doc_ids"]]
    hit_rate = sum(1 for r in answerable if r["retrieval_eval"]["hit"]) / len(answerable) if answerable else 0
    avg_recall = sum(r["retrieval_eval"]["recall"] for r in answerable) / len(answerable) if answerable else 0
    hallucination_rate = sum(1 for r in results if not r["grounding_eval"]["grounded"]) / len(results)
    avg_citation_score = sum(r["citation_eval"]["citation_score"] for r in results) / len(results)

    summary = {
        "run_label": run_label,
        "num_questions": len(results),
        "retrieval_hit_rate": round(hit_rate, 3),
        "avg_retrieval_recall": round(avg_recall, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "avg_citation_score": round(avg_citation_score, 3),
    }

    output = {"summary": summary, "results": results}
    out_path = EVALS_DIR / f"results_{run_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=" * 50)
    print(f"Run: {run_label}")
    print(f"Retrieval hit rate:   {summary['retrieval_hit_rate']}")
    print(f"Avg retrieval recall: {summary['avg_retrieval_recall']}")
    print(f"Hallucination rate:   {summary['hallucination_rate']}")
    print(f"Avg citation score:   {summary['avg_citation_score']}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
