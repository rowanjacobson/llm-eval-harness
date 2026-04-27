import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import retrieve_vector
from retrieve import load_json, build_doc_index, retrieve_docs, build_context, EMBEDDING_MODEL
from agent import ask_claude
from evals import eval_retrieval, eval_citations, eval_grounding

load_dotenv(Path(__file__).parent.parent / ".env")

DOCS_PATH = Path(__file__).parent.parent / "docs/documents.json"
QUESTIONS_PATH = Path(__file__).parent.parent / "evals/questions.json"
EVALS_DIR = Path(__file__).parent.parent / "evals"
CHROMA_PATH = str(Path(__file__).parent.parent / "vector_db")
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


def build_context_from_chroma(retrieved_docs: list) -> str:
    parts = []
    for doc in retrieved_docs:
        parts.append(
            f"<document>\n"
            f"<doc_id>{doc['doc_id']}</doc_id>\n"
            f"<title>{doc['title']}</title>\n"
            f"<text>{doc['text']}</text>\n"
            f"</document>"
        )
    return "\n\n".join(parts)


def run_pipeline(question: str, gold_doc_ids: list, context: str, retrieved_doc_ids: list, client: anthropic.Anthropic) -> dict:
    raw_answer = ask_claude(question, context, client)

    try:
        parsed = parse_answer(raw_answer)
        answer_text = parsed.get("answer", "")
        citations = parsed.get("citations", [])
        abstain = parsed.get("abstain", False)
        uncertainty = parsed.get("uncertainty", "")
    except (json.JSONDecodeError, KeyError):
        answer_text = raw_answer
        citations = []
        abstain = False
        uncertainty = ""

    return {
        "retrieved_doc_ids": retrieved_doc_ids,
        "answer": answer_text,
        "citations": citations,
        "abstain": abstain,
        "uncertainty": uncertainty,
        "retrieval_eval": eval_retrieval(retrieved_doc_ids, gold_doc_ids),
        "citation_eval": eval_citations(citations, retrieved_doc_ids),
        "grounding_eval": eval_grounding(question, answer_text, context, client),
    }


def compute_summary(results: list, key: str) -> dict:
    answerable = [r for r in results if r["gold_doc_ids"]]
    hit_rate = sum(1 for r in answerable if r[key]["retrieval_eval"]["hit"]) / len(answerable) if answerable else 0
    avg_recall = sum(r[key]["retrieval_eval"]["recall"] for r in answerable) / len(answerable) if answerable else 0
    hallucination_rate = sum(1 for r in results if not r[key]["grounding_eval"]["grounded"]) / len(results)
    avg_citation_score = sum(r[key]["citation_eval"]["citation_score"] for r in results) / len(results)
    return {
        "retrieval_hit_rate": round(hit_rate, 3),
        "avg_retrieval_recall": round(avg_recall, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "avg_citation_score": round(avg_citation_score, 3),
    }


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    documents = load_json(DOCS_PATH)
    questions = load_json(QUESTIONS_PATH)

    print("Loading numpy embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    doc_embeddings = build_doc_index(documents, embed_model)

    print("Initialising ChromaDB retriever...")
    retrieve_vector.CHROMA_PATH = CHROMA_PATH
    retrieve_vector.init()

    print(f"Indexed {len(documents)} documents.\n")

    run_label = next_version(RUN_LABEL)
    results = []

    for i, q_obj in enumerate(questions, start=1):
        question = q_obj["question"]
        gold_doc_ids = q_obj.get("source_doc_ids", [])

        # Numpy pipeline
        numpy_retrieved = retrieve_docs(question, documents, doc_embeddings, embed_model, top_k=TOP_K)
        numpy_doc_ids = [doc["doc_id"] for doc, _ in numpy_retrieved]
        numpy_context = build_context(numpy_retrieved)
        numpy_result = run_pipeline(question, gold_doc_ids, numpy_context, numpy_doc_ids, client)

        # Chroma pipeline
        chroma_retrieved = retrieve_vector.retrieve(question, k=TOP_K)
        chroma_doc_ids = chroma_retrieved["retrieved_doc_ids"]
        chroma_context = build_context_from_chroma(chroma_retrieved["retrieved_docs"])
        chroma_result = run_pipeline(question, gold_doc_ids, chroma_context, chroma_doc_ids, client)

        result = {
            "question_number": i,
            "question_id": q_obj["question_id"],
            "question": question,
            "gold_doc_ids": gold_doc_ids,
            "numpy": numpy_result,
            "chroma": chroma_result,
        }
        results.append(result)

        print(f"--- Q{i} ({q_obj['question_id']}) ---")
        print(f"Q: {question}")
        print(f"  [numpy]  hit={numpy_result['retrieval_eval']['hit']}  answer: {numpy_result['answer'][:80]}")
        print(f"  [chroma] hit={chroma_result['retrieval_eval']['hit']}  answer: {chroma_result['answer'][:80]}\n")

    numpy_summary = compute_summary(results, "numpy")
    chroma_summary = compute_summary(results, "chroma")

    output = {
        "summary": {
            "run_label": run_label,
            "num_questions": len(results),
            "numpy": numpy_summary,
            "chroma": chroma_summary,
        },
        "results": results,
    }

    out_path = EVALS_DIR / f"results_{run_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=" * 50)
    print(f"Run: {run_label}")
    print(f"{'Metric':<30} {'numpy':>10} {'chroma':>10}")
    print(f"{'Retrieval hit rate':<30} {numpy_summary['retrieval_hit_rate']:>10} {chroma_summary['retrieval_hit_rate']:>10}")
    print(f"{'Avg retrieval recall':<30} {numpy_summary['avg_retrieval_recall']:>10} {chroma_summary['avg_retrieval_recall']:>10}")
    print(f"{'Hallucination rate':<30} {numpy_summary['hallucination_rate']:>10} {chroma_summary['hallucination_rate']:>10}")
    print(f"{'Avg citation score':<30} {numpy_summary['avg_citation_score']:>10} {chroma_summary['avg_citation_score']:>10}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
