import json
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


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
    top_k: int = TOP_K,
) -> List[Tuple[Dict[str, str], float]]:
    question_embedding = embed_texts([question], model)
    similarities = cosine_similarity(question_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(documents[i], float(similarities[i])) for i in top_indices]


def build_context(retrieved_docs: List[Tuple[Dict[str, str], float]]) -> str:
    parts = []
    for doc, _ in retrieved_docs:
        parts.append(
            f"<document>\n"
            f"<doc_id>{doc['doc_id']}</doc_id>\n"
            f"<title>{doc['title']}</title>\n"
            f"<text>{doc['text']}</text>\n"
            f"</document>"
        )
    return "\n\n".join(parts)
