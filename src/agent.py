#!/opt/anaconda3/bin/python3
import json
from pathlib import Path
import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

def load_json(path):
    with open(ROOT / path) as f:
        return json.load(f)

def main():
    docs = load_json("docs/documents.json")
    questions = load_json("evals/questions.json")
    question = questions[0]["question"]

    docs_text = "\n\n".join(
        f"[{doc['title']}]\n{doc['text']}" for doc in docs
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": f"You are a helpful assistant. Answer questions using only the documents below.\n\n{docs_text}",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": question}],
    )

    answer = next(b.text for b in response.content if b.type == "text")
    print(f"Q: {question}")
    print(f"A: {answer}")

if __name__ == "__main__":
    main()
