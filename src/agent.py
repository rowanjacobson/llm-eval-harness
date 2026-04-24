import anthropic

MODEL_NAME = "claude-sonnet-4-6"


def ask_claude(question: str, context: str, client: anthropic.Anthropic) -> str:
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
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.content[0].text
