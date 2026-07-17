"""
qa_engine.py
------------
The "Generation" half of RAG - calling OpenRouter (free tier), with:

Document facts are grounded in retrieved text. Recent conversation is used
only to resolve follow-up references such as "what does that mean?".
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Pinned to a specific, verified-free, general-purpose model - NOT the
# "openrouter/free" auto-router, which unpredictably picks a different
# model each request (that's what caused the "User Safety: safe" bug
# earlier). Using the 20b variant - the 120b variant was pulled from the
# free tier (confirmed via a live 404 error from OpenRouter itself).
MODEL_NAME = "openai/gpt-oss-20b:free"

MAX_HISTORY_TURNS = 6

ACKNOWLEDGEMENTS = {
    "ok", "okay", "thanks", "thank you", "got it", "understood",
    "cool", "great", "nice", "perfect",
}
FOLLOW_UP_MARKERS = (
    "as you said", "you said", "what does that mean", "what do you mean",
    "what does it mean", "what do you mean by", "explain that", "explain it",
    "tell me more", "elaborate", "how so", "why is that", "continue",
)

SYSTEM_PROMPT = (
    "You are a helpful assistant embedded in a PDF chat tool. The user has "
    "uploaded one or more documents. Document context is the only evidence "
    "for factual claims: do not use general knowledge or make unsupported "
    "inferences. Recent chat history may be used only to understand what a "
    "follow-up pronoun or phrase refers to; it is not factual evidence. If "
    "the supplied document context does not support an answer, say: "
    "'I couldn't find that in the uploaded documents.' A short acknowledgement "
    "such as 'okay' can receive a brief, natural conversational reply.\n\n"
    "Formatting rules: write in plain conversational text, like a chat "
    "message. Do NOT use markdown headers (no #, ##, ###) - they render as "
    "giant page titles in this interface. DO use a markdown table whenever "
    "the question asks for a comparison, difference, or lists multiple "
    "items side by side - tables are much clearer than prose for that. "
    "IMPORTANT table rule: this renderer does NOT support <br> tags or line "
    "breaks inside table cells - they show up as literal broken text. Keep "
    "each table cell to a single short line; if a cell needs multiple "
    "points, separate them with semicolons on one line instead of line "
    "breaks or bullets within the cell. Use bullet points for straightforward "
    "lists outside of tables, and bold for key terms.\n\n"
    "Length rules: match your answer's length to the question. A specific "
    "factual question deserves a short, direct answer. A broad request "
    "like 'summarize this document' deserves a thorough answer that "
    "actually covers the material. Always finish your last sentence "
    "completely - never cut off mid-thought."
)


def classify_message(question: str, chat_history: list) -> str:
    """Classify a turn so follow-ups can reuse the previous cited context."""
    normalized = " ".join(question.lower().strip().split()).strip(".!?")
    if normalized in ACKNOWLEDGEMENTS:
        return "acknowledgement"
    if chat_history and any(marker in normalized for marker in FOLLOW_UP_MARKERS):
        return "follow_up"
    return "new_question"


def select_context(question: str, vector_store, chat_history: list, top_k: int = 8):
    """Return context and mode, avoiding a new vector search for follow-ups."""
    mode = classify_message(question, chat_history)
    if mode == "follow_up":
        return chat_history[-1].get("sources", []), mode
    if mode == "acknowledgement":
        return [], mode
    return vector_store.search(question, top_k=top_k), mode


def build_messages(question: str, retrieved_chunks: list, chat_history: list) -> list:
    """
    Builds messages with recent conversational context and source chunks.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in chat_history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    if retrieved_chunks:
        context_text = "\n\n".join(
            f"[From {chunk['source']}, page {chunk['page']}]: {chunk['text']}"
            for chunk in retrieved_chunks
        )
        user_content = (
            f"Document context for this turn:\n{context_text}\n\nQuestion: {question}"
        )
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})
    return messages


def answer_question(question: str, retrieved_chunks: list, chat_history: list = None) -> str:
    """
    Ties everything together: build conversation-aware messages, call
    OpenRouter, return the generated answer text.
    """
    if chat_history is None:
        chat_history = []

    if not OPENROUTER_API_KEY:
        return (
            "No OpenRouter API key found. Create a .env file in the project "
            "root with a line like: OPENROUTER_API_KEY=sk-or-v1-..."
        )

    messages = build_messages(question, retrieved_chunks, chat_history)

    try:
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.2,
            },
            timeout=45,  # bigger model, allow more time
        )
    except requests.exceptions.RequestException as e:
        return f"Network error reaching OpenRouter: {e}"

    if response.status_code == 429:
        return (
            "OpenRouter's free tier rate limit was hit (this happens with "
            "heavy testing on free models). Wait a minute and try again."
        )

    if response.status_code == 404 and "unavailable for free" in response.text.lower():
        return (
            f"The model '{MODEL_NAME}' was just pulled from OpenRouter's "
            f"free tier (this rotates without notice). Update MODEL_NAME "
            "in qa.py to a currently available OpenRouter model."
        )

    if response.status_code != 200:
        return f"OpenRouter API error ({response.status_code}). Please try again shortly."

    try:
        data = response.json()
    except ValueError:
        return "OpenRouter returned an invalid response. Please try again."

    try:
        content = data["choices"][0]["message"].get("content")
    except (KeyError, IndexError):
        return f"Unexpected response format from OpenRouter: {data}"

    if not content:
        return "The model returned an empty response. Please try asking again."

    return content.strip()
