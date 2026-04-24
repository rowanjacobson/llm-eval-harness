import json
import re
from typing import Dict, List

import anthropic

JUDGE_MODEL = "claude-haiku-4-5"


def eval_retrieval(retrieved_doc_ids: List[str], gold_doc_ids: List[str]) -> Dict:
    if not gold_doc_ids:
        return {"hit": None, "recall": None, "missed_docs": []}
    hit = any(doc_id in retrieved_doc_ids for doc_id in gold_doc_ids)
    matched = [doc_id for doc_id in gold_doc_ids if doc_id in retrieved_doc_ids]
    missed = [doc_id for doc_id in gold_doc_ids if doc_id not in retrieved_doc_ids]
    recall = len(matched) / len(gold_doc_ids)
    return {"hit": hit, "recall": recall, "missed_docs": missed}


def eval_citations(citations: List[str], retrieved_doc_ids: List[str]) -> Dict:
    if not citations:
        return {"citation_score": 0.0, "hallucinated_citations": []}
    valid = [c for c in citations if c in retrieved_doc_ids]
    hallucinated = [c for c in citations if c not in retrieved_doc_ids]
    citation_score = len(valid) / len(citations)
    return {"citation_score": citation_score, "hallucinated_citations": hallucinated}


def eval_grounding(
    question: str, answer: str, context: str, client: anthropic.Anthropic
) -> Dict:
    prompt = (
        "You are an evaluation judge. Given a question, a context of documents, and an answer, "
        "determine whether the answer is grounded in the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Return valid JSON with this exact schema:\n"
        '{"grounded": true_or_false, "reason": "short explanation"}'
    )

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)
