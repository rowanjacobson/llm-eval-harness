import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "vector_db"

_model = None
_collection = None


def init():
    global _model, _collection
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = client.get_collection(name="internal_docs")


def retrieve(question, k=5):
    query_embedding = _model.encode(question).tolist()

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    retrieved_docs = []
    for doc_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved_docs.append({
            "doc_id": doc_id,
            "title": metadata.get("title", ""),
            "text": text,
            "score": distance
        })

    return {
        "retrieved_docs": retrieved_docs,
        "retrieved_doc_ids": [doc["doc_id"] for doc in retrieved_docs],
        "scores": [doc["score"] for doc in retrieved_docs]
    }
