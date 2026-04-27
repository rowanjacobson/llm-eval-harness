import json
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_PATH = "docs/documents.json"
CHROMA_PATH = "vector_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="internal_docs"
)

with open(DOCS_PATH, "r") as f:
    documents = json.load(f)

ids = []
texts = []
metadatas = []
embeddings = []

for doc in documents:
    text = f"{doc['title']}\n{doc['text']}"

    ids.append(doc["doc_id"])
    texts.append(text)
    metadatas.append({
        "title": doc.get("title", ""),
        "department": doc.get("department", ""),
        "source_type": doc.get("source_type", ""),
        "last_updated": doc.get("last_updated", "")
    })
    embeddings.append(model.encode(text).tolist())

collection.upsert(
    ids=ids,
    documents=texts,
    metadatas=metadatas,
    embeddings=embeddings
)

print(f"Indexed {len(documents)} documents into Chroma.")