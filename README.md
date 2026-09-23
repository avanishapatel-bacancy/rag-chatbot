# 🧠 DocuMind - AI Document Assistant - a free, end-to-end RAG Chatbot

Upload documents, ask questions, get grounded answers with citations -
built entirely on free tools, runnable identically on your laptop or on a
free public URL anyone can open.

**Live pipeline:** upload → parse → chunk → embed → hybrid retrieve → cited,streamed answer.

## Why these choices

| Stage | Choice | Why |
|---|---|---|
| Chat LLM | **Google Gemini** (`gemini-flash-latest` etc., via the `google-genai` SDK) | Free API tier, no credit card, fast; the `-latest` alias tracks whatever Google currently recommends so the app doesn't break when a model is retired |
| Embeddings | **Local model via `fastembed`** (`BAAI/bge-small-en-v1.5`, ONNX runtime) | Runs entirely on-device - no API key, no rate limit, no usage quota, so processing documents is never blocked by a provider's billing tier. Lighter than `sentence-transformers`+`torch` (no GPU deps), so it still installs fast on free hosting |
| Vector search | **In-memory hybrid (cosine + BM25)**, per browser session | No database to host; one visitor's documents are never visible to another |
| Frontend | **Streamlit** | One codebase runs identically with `streamlit run app.py` locally *and* on Streamlit Community Cloud - no separate frontend/backend deploy |

## Features

- **Multi-format ingestion**: PDF, DOCX, TXT, Markdown, multiple files at once.
- **Paragraph-aware chunking** with configurable size/overlap and sliding overlap so context isn't cut mid-thought.
- **Hybrid retrieval**: blends semantic (embedding) similarity with BM25 keyword search - a slider lets you tune the mix live.
- **Grounded, cited answers**: the model is instructed to answer only from retrieved context and cite `[1] [2]…`; it says so explicitly when the answer isn't in your documents (reduces hallucination).
- **Multi-turn conversation**: follow-up questions are rewritten into standalone queries using chat history before retrieval.
- **Streaming responses** with a live-updating cursor and a source-citation viewer.
- **Session-isolated storage**: your uploaded documents exist only in your browser session's memory - never written to disk, never shared with other visitors.
- **Bring-your-own-key**: visitors can optionally paste their own free Gemini key so a busy public deployment doesn't burn through the owner's free-tier quota; a per-session rate limit protects the shared quota either way. Document processing itself never needs a key at all, since embeddings run locally.

## Architecture

```
Browser (Streamlit UI)
  │
  ├─ Upload → document_loader.py (pypdf / python-docx) → chunking.py
  │                                                           │
  │                                       embeddings.py (local fastembed, no API)
  │                                                           │
  │                                          vectorstore.py (in-memory, per session)
  │
  └─ Question → llm.py: condense → retrieve (hybrid search) → stream_answer (Gemini)
```

## Run it locally

```bash
git clone <your-repo-url>
cd RAG_Chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your free Gemini API key
# (get one in 30 seconds at https://aistudio.google.com/apikey)

streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

> The first time you process a document, `fastembed` downloads its embedding
> model (~130MB, one-time, from Hugging Face) and caches it locally - this
> takes ~20-30s. Every embedding after that is instant local CPU inference,
> with no API key and no rate limit.

## Deploy it for free - get a public URL anyone can use

This same repo deploys unchanged to **Streamlit Community Cloud**:

1. Push this project to a GitHub repository (see below).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub (free).
3. Click **"New app"**, pick your repo/branch, and set the main file to `app.py`.
4. Under **"Advanced settings" → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-free-gemini-key"
   ```
5. Click **Deploy**. You'll get a public URL like `https://your-app.streamlit.app` that anyone can open - no login required for visitors.

> Alternative: this Streamlit app also runs unchanged on **[Hugging Face Spaces](https://huggingface.co/spaces)** (choose the "Streamlit" SDK, add the same secret under Space settings) if you'd prefer HF's hosting and built-in container logs.

### Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: DocuMind - AI Document Assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Configuration reference

| Setting | Where | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | secrets/env | *(required)* | Owner's fallback API key |

## Tech stack

Python · Streamlit · Google Gemini (chat, free tier) · fastembed (local, free,
unlimited embeddings) · BM25 (`rank-bm25`) · NumPy.

## Possible next steps

- Add reranking (e.g. a cross-encoder) on top of the hybrid retriever for higher precision.
- Persist a user's session vector store to disk (opt-in) so it survives a page refresh.
