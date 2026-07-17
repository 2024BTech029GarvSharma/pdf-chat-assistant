# AI-Powered PDF Chat Assistant (RAG)

A Streamlit Retrieval-Augmented Generation (RAG) application for asking questions about one or more uploaded PDFs. It extracts page-level text, creates embeddings locally, searches with FAISS, and asks an OpenRouter LLM to answer from the retrieved PDF content.

## Features

- Upload and query multiple text-based PDFs in one session.
- Page-aware citations: each answer shows the source file and page numbers used.
- Local extraction, chunking, embeddings, and FAISS vector search.
- Conversational follow-ups: questions such as “what does that mean?” reuse the sources from the preceding answer instead of performing an unnecessary new search.
- Natural acknowledgement handling: messages such as “okay” receive a brief response without document retrieval.
- Grounded answers: factual claims must be supported by supplied PDF context; unsupported questions receive a clear fallback.
- Responsive Streamlit interface with desktop and mobile styling.
- User-facing errors for corrupt PDFs and PDFs with no extractable text.

## Architecture

```text
PDF upload -> pdfplumber text extraction -> page-level chunks
          -> SentenceTransformer embeddings -> FAISS index
Question -> new search OR prior cited context -> OpenRouter LLM -> answer + sources
```

## Requirements

- Python 3.10 or newer.
- An OpenRouter API key. The configured `openai/gpt-oss-20b:free` model is free when available, but availability and response time can vary.
- Internet access for the initial embedding-model download and for OpenRouter generation.

## Installation

```bash
git clone <your-repository-url>
cd pdf-chat-assistant
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root. Do not commit it.

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Start the app:

```bash
python -m streamlit run app.py
```

`python -m streamlit` is recommended because it works even if the `streamlit` command is not on your PATH.

## How to use

1. Open the local URL displayed by Streamlit.
2. Upload one or more PDFs from the sidebar.
3. Wait for the “file(s) ready” confirmation.
4. Ask a question, for example: `What is the difference between Mitra and admin?`
5. Ask a follow-up, for example: `As you said, what does per-candidate cash mean?` The app reuses the previous answer’s cited pages.
6. Expand **Sources used for this answer** to verify the file and page numbers.

## Testing

Run the automated conversation tests:

```bash
python -m unittest discover -s tests -v
```

The tests confirm that new questions search the vector store, follow-ups reuse previous citations, acknowledgements do not search, and the LLM receives both conversational context and document evidence.

## Limitations and safety

- Only text-based PDFs are supported. Scanned/image-only PDFs require OCR, which is not currently included.
- Vectors and chat history are kept only in the active Streamlit session; nothing is persisted after the session ends.
- Follow-ups reuse the immediately preceding answer’s sources. A follow-up that changes topic should be phrased as a new question so it triggers a fresh search.
- Retrieval reduces hallucination risk but cannot guarantee correctness. Always check the displayed source pages for important decisions.
- Free OpenRouter models can be rate-limited, slow, or removed. For production, use a stable paid model and add authentication, rate limiting, persistent storage, and observability.

## Project structure

```text
app.py                 Streamlit UI and session management
utils/pdf_utils.py     PDF extraction and validation
utils/vector_store.py  Chunking, embedding, and FAISS retrieval
utils/qa.py            Conversation routing and OpenRouter generation
tests/                 Automated behavior tests
requirements.txt       Python dependencies
```

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `OPENROUTER_API_KEY` error | Check that `.env` is in the project root and contains a non-empty key. Run `python check_env.py`. |
| `streamlit` is not recognized | Use `python -m streamlit run app.py`. |
| No text can be extracted | The PDF is likely scanned or protected; use an OCR-enabled version. |
| Slow or failed responses | Free OpenRouter capacity may be limited; retry later or configure a stable model. |
| Follow-up has the wrong subject | State the entity again, e.g. “For Mitra, explain per-candidate cash.” |
