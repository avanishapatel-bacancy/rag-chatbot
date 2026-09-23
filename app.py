"""DocuMind - a free, end-to-end RAG chatbot.

Upload documents -> parse -> chunk -> embed (local, free) -> hybrid retrieve
-> grounded, cited, streamed answers (Gemini). Every visitor's documents
live only in their own browser session (never persisted or shared).
"""
from __future__ import annotations

import os
import time

import numpy as np
import streamlit as st

from rag.chunking import chunk_documents
from rag.document_loader import SUPPORTED_EXTENSIONS, parse_document
from rag.embeddings import embed_documents, embed_query
from rag.llm import NOT_FOUND_PHRASE, condense_question, stream_answer
from rag.vectorstore import HybridVectorStore

MODEL_CHOICES = ["models/gemini-flash-lite-latest", "models/gemini-flash-latest", "models/gemini-2.5-flash"]
MAX_QUERIES_PER_MINUTE = 10
BOT_AVATAR = "🧠"
USER_AVATAR = "🙂"
FILE_ICONS = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝", "markdown": "📝"}

st.set_page_config(page_title="DocuMind - RAG Chatbot", page_icon="🧠", layout="wide")

# ------------------------------------------------------------- theme setup --
# A single, fixed "Aurora" theme: a bright, airy canvas with an indigo ->
# violet -> cyan accent gradient. No light/dark toggle - one considered
# palette, tuned for strong text contrast throughout.
THEME = {
    "bg-app": "#f6f7fc",
    "bg-sidebar": "#ffffff",
    "bg-surface": "#ffffff",
    "bg-surface-hover": "#f0f0fd",
    "bg-input": "#ffffff",
    "text-primary": "#161a2b",
    "text-secondary": "#5b6273",
    "border-color": "rgba(15, 23, 42, 0.12)",
    "accent-1": "#4f46e5",
    "accent-2": "#9333ea",
    "accent-3": "#0891b2",
    "pill-bg": "rgba(79, 70, 229, 0.1)",
    "pill-text": "#4338ca",
    "shadow-sm": "0 1px 2px rgba(15, 23, 42, 0.06)",
    "shadow-md": "0 8px 24px rgba(15, 23, 42, 0.09)",
    "bubble-user-bg": "linear-gradient(120deg, #4f46e5 0%, #7c3aed 100%)",
    "bubble-user-text": "#ffffff",
    "bubble-bot-bg": "#ffffff",
    "bubble-bot-text": "#161a2b",
    "scrollbar-thumb": "#d5d8e3",
}


def _vars(colors: dict) -> str:
    return "\n".join(f"--{k}: {v};" for k, v in colors.items())


_theme_vars = f":root {{ {_vars(THEME)} }}"

# ---------------------------------------------------------------- styling --
# NOTE: every line below starts at column 0 (no leading indentation). Streamlit's
# markdown renderer treats a 4-space-indented line as a CommonMark indented code
# block, which would make the raw HTML/CSS show up as literal escaped text.
_STYLE_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

{_theme_vars}

html, body, .stApp,
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] {{
background: var(--bg-app);
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* Clears the chat input's own fixed overlay below the disclaimer, which
(being out of normal flow - see below) doesn't reserve space for itself. */
.block-container {{ max-width: 900px; padding-top: 1.5rem; padding-bottom: 90px; box-sizing: border-box; }}

.block-container :is(h1, h2, h3, h4, h5, h6, p, span, label):not(.documind-hero *),
[data-testid="stSidebar"] :is(h1, h2, h3, h4, h5, h6, p, span, label) {{
color: var(--text-primary);
}}
[data-testid="stCaptionContainer"] {{ color: var(--text-secondary) !important; }}

/* keep two-up column rows (doc pill + remove button, example-question
grid, etc.) vertically centered instead of top-aligned */
[data-testid="stHorizontalBlock"] {{ align-items: center; }}

[data-testid="stSidebar"] {{
background: var(--bg-sidebar);
border-right: 1px solid var(--border-color);
}}

.documind-hero {{
display: flex; align-items: center; gap: 1rem;
padding: 1.5rem 1.8rem; border-radius: 16px; margin-bottom: 1.1rem;
background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 55%, var(--accent-3) 100%);
color: white; box-shadow: var(--shadow-md);
}}
.documind-hero .hero-icon {{
display: flex; align-items: center; justify-content: center;
width: 3rem; height: 3rem; border-radius: 12px; flex-shrink: 0;
background: rgba(255, 255, 255, 0.16); font-size: 1.6rem;
}}
.documind-hero h1 {{ margin: 0; font-size: 1.55rem; font-weight: 800; color: #fff; letter-spacing: -.01em; }}
.documind-hero p {{ margin: .3rem 0 0; opacity: .94; font-size: .92rem; color: #fff; }}

.status-line {{
display: flex; align-items: center; gap: .4rem;
font-size: .85rem; color: var(--text-secondary); margin: 0 0 1rem 0;
}}

.doc-pill {{
display: inline-block; padding: 3px 11px; border-radius: 999px;
background: var(--pill-bg); color: var(--pill-text); font-size: .78rem;
margin: 2px 4px 2px 0; border: 1px solid var(--border-color);
}}
.source-card {{
border: 1px solid var(--border-color); border-radius: 10px;
background: var(--bg-surface); color: var(--text-primary);
padding: .55rem .75rem; margin-bottom: .5rem; font-size: .87rem;
}}
.source-card b {{ color: var(--accent-2); }}
.relevance-track {{
height: 4px; border-radius: 2px; background: var(--border-color);
margin: .3rem 0 .45rem 0; overflow: hidden;
}}
.relevance-fill {{ height: 100%; background: linear-gradient(90deg, var(--accent-2), var(--accent-3)); }}

/* Three things need to be true for the disclaimer to sit right above the
chat input at all times: (1) stay glued to the viewport regardless of
scroll position, (2) line up with the sidebar-aware content column
rather than the full window, (3) not depend on it being the "last"
element, since Streamlit's rerun/reconciliation doesn't reliably keep it
there as messages accumulate (a DOM-order trick like sticky or flex
"last child" intermittently targets a stale mid-list position instead).

"position: fixed" gives (1) and (3) for free - but by default it centers
on the full window for (2), ignoring the sidebar. A transform on an
ancestor can redefine its containing block to fix that, but only if that
ancestor is the plain, non-scrolling content frame (stAppViewContainer's
main-column child) - putting the transform on the scrollable pane inside
it (stAppScrollToBottomContainer) very nearly works, except a "position:
fixed" descendant would still be clipped/scrolled by that pane's own
overflow (only escaping AT the ancestor the containing block resolves
to, not before it) - which is what caused visible repaint glitches
during scroll in that version. This one skips past the scrolling pane
entirely. */
[data-testid="stAppViewContainer"] > div:has([data-testid="stAppScrollToBottomContainer"]) {{
transform: translateZ(0);
}}
.disclaimer {{
position: fixed; left: 50%; transform: translateX(-50%); bottom: 76px;
z-index: 20; width: 100%; max-width: 900px; box-sizing: border-box;
text-align: center; font-size: .78rem; color: var(--text-secondary);
margin: 0; padding: 8px 1.5rem 6px; background: var(--bg-app);
}}

/* buttons & inputs */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {{
border-radius: 10px; background: var(--bg-surface); color: var(--text-primary);
border: 1px solid var(--border-color); transition: background .15s ease, border-color .15s ease, color .15s ease;
}}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {{
border-color: var(--accent-2); color: var(--accent-2); background: var(--bg-surface-hover);
}}
/* Streamlit's visible border lives on this wrapper, not on the <input>
itself - targeting the input alone leaves the wrapper's own (hardcoded
white) border in place, invisible on a light background. Sized and
rounded to match the chat input pill instead of Streamlit's default box. */
[data-testid="stTextInputRootElement"] {{
background: var(--bg-input) !important; border: 1px solid var(--border-color) !important;
border-radius: 12px !important; height: 2.35rem !important; min-height: 2.35rem !important;
}}
[data-testid="stTextInputRootElement"] input {{
background: transparent !important; color: var(--text-primary) !important;
padding-top: 0 !important; padding-bottom: 0 !important;
}}
[data-testid="stSelectbox"] [role="group"] {{
background: var(--bg-input) !important; border: 1px solid var(--border-color) !important;
border-radius: 10px !important;
}}
[data-testid="stSelectbox"] input {{ color: var(--text-primary) !important; background: transparent !important; }}
[data-testid="stSelectboxVirtualDropdown"] {{
background: var(--bg-surface) !important; color: var(--text-primary) !important;
border: 1px solid var(--border-color) !important;
}}
[data-testid="stSelectboxVirtualDropdown"] * {{ background: transparent !important; color: var(--text-primary) !important; }}
[data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"] {{ background: var(--bg-surface-hover) !important; }}
[data-testid="stExpander"] {{
background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 12px;
}}
[data-testid="stFileUploaderDropzone"] {{
background: var(--bg-surface); border: 1px dashed var(--border-color); border-radius: 12px;
}}

/* realtime chat bubbles */
[data-testid="stChatMessage"] {{
border-radius: 18px; padding: .85rem 1rem; margin-bottom: .6rem;
border: 1px solid var(--border-color); background: var(--bubble-bot-bg);
box-shadow: var(--shadow-sm); animation: documind-fade-in .25s ease-out;
}}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] * {{
color: var(--bubble-bot-text);
}}
/* Streamlit gives each chat bubble a distinct wrapper, so positional
selectors like :nth-child can't tell user/assistant apart -- the
aria-label Streamlit puts on the content div is the reliable signal. */
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{
background: var(--bubble-user-bg); border: none;
}}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] * {{
color: var(--bubble-user-text);
}}
[data-testid="stChatMessageAvatarCustom"] {{
background: var(--bg-surface); border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);
}}
@keyframes documind-fade-in {{
from {{ opacity: 0; transform: translateY(4px); }}
to {{ opacity: 1; transform: translateY(0); }}
}}

[data-testid="stChatInput"], [data-testid="stChatInput"] > div {{
background: var(--bg-input) !important; border-radius: 14px;
}}
[data-testid="stChatInput"] {{ border: 1px solid var(--border-color); }}
[data-testid="stChatInput"] > div {{ padding: 6px 14px !important; }}
[data-testid="stChatInputTextArea"] {{ color: var(--text-primary); min-height: 20px !important; }}
[data-testid="stChatInputTextArea"]::placeholder {{ color: var(--text-secondary); opacity: 1; }}
[data-testid="stChatInputSubmitButton"] {{ color: var(--accent-2); }}
/* Streamlit pads the whole fixed footer generously by default and leaves
it full-width; tighten the padding and cap it to the same 900px column
as .block-container so the input lines up with the hero/buttons above
it instead of stretching edge-to-edge. */
[data-testid="stBottomBlockContainer"] {{
padding: 10px 0 18px !important; max-width: 900px !important; margin: 0 auto !important;
}}
/* Streamlit wraps the footer in an unlabeled div (no data-testid) that's
hardcoded white, showing as a mismatched band behind the input on this
theme - it's the sole direct child of stBottom, so target it structurally. */
[data-testid="stBottom"] > div {{ background: var(--bg-app) !important; }}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--scrollbar-thumb); border-radius: 8px; }}
"""

st.markdown(f"<style>{_STYLE_CSS}</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="documind-hero">'
    '<div class="hero-icon">🧠</div>'
    "<div>"
    "<h1>DocuMind</h1>"
    "<p>Ask questions about your documents and get grounded, cited answers. "
    "Your documents stay private to this browser session.</p>"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------- session state --
defaults = {
    "messages": [],
    "store": HybridVectorStore(),
    "processed_signatures": set(),
    "user_api_key": "",
    "query_timestamps": [],
    "pending_question": None,
    "regenerate_question": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def resolve_api_key() -> str | None:
    own_key = st.session_state.user_api_key.strip()
    if own_key:
        return own_key
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        secret_key = None
    return secret_key or os.environ.get("GEMINI_API_KEY")


def file_icon(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return FILE_ICONS.get(ext, "📄")


# --------------------------------------------------------------- sidebar --
with st.sidebar:
    st.subheader("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or Markdown",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )
    process_clicked = st.button("⚙️ Process documents", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            new_files = [
                f for f in uploaded_files
                if (f.name, f.size) not in st.session_state.processed_signatures
            ]
            if not new_files:
                st.info("All uploaded files are already processed.")
            else:
                # Chunks/vectors from every file in this batch are collected
                # here and added to the store in a single call below -
                # HybridVectorStore.add() rebuilds its whole BM25 index from
                # scratch on every call, so adding per-file would rebuild it
                # once per file over an ever-growing corpus instead of once.
                batch_chunks = []
                batch_vectors = []
                indexed = []
                for f in new_files:
                    try:
                        with st.spinner(f"Parsing {f.name}..."):
                            sections = parse_document(f.name, f.getvalue())
                        with st.spinner(f"Chunking {f.name}..."):
                            chunks = chunk_documents({f.name: sections})
                        if not chunks:
                            st.warning(f"No extractable text found in {f.name}.")
                            continue
                        progress = st.progress(0.0, text=f"Embedding {f.name}...")

                        def _cb(done, total):
                            progress.progress(done / total, text=f"Embedding {f.name}... ({done}/{total})")

                        vectors = embed_documents([c.text for c in chunks], progress_cb=_cb)
                        progress.empty()
                        batch_chunks.extend(chunks)
                        batch_vectors.append(vectors)
                        st.session_state.processed_signatures.add((f.name, f.size))
                        indexed.append((f.name, len(chunks)))
                    except Exception as exc:
                        st.error(f"Failed to process {f.name}: {exc}")

                if batch_chunks:
                    st.session_state.store.add(batch_chunks, np.vstack(batch_vectors))
                    for name, n in indexed:
                        st.success(f"Indexed {name} ({n} chunks).")

    store: HybridVectorStore = st.session_state.store
    if store.sources:
        st.caption(f"{len(store)} chunks indexed across {len(store.sources)} document(s):")
        for source in store.sources:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f'<span class="doc-pill">{file_icon(source)} {source}</span>', unsafe_allow_html=True)
            if col2.button("×", key=f"remove_{source}", help="Remove this document"):
                store.remove_source(source)
                st.session_state.processed_signatures = {
                    sig for sig in st.session_state.processed_signatures if sig[0] != source
                }
                st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.subheader("🔑 Gemini API Key")
    st.caption("Only needed for answering questions - document processing runs fully locally and needs no key.")
    st.session_state.user_api_key = st.text_input(
        "Use your own free key (optional)",
        type="password",
        help=(
            "Get a free key at aistudio.google.com/apikey. If left blank, the app's "
            "shared key is used - supplying your own avoids sharing the free-tier quota "
            "with other visitors."
        ),
    )

    st.divider()
    with st.expander("⚙️ Retrieval & model settings"):
        model_name = st.selectbox(
            "Gemini model", MODEL_CHOICES, index=0,
            help="flash-lite is fastest; flash and 2.5-flash trade some speed for stronger reasoning.",
        )
        top_k = st.slider("Chunks retrieved per question", 1, 10, 5)
        alpha = st.slider("Semantic ↔ keyword blend", 0.0, 1.0, 0.6, help="1.0 = pure semantic search, 0.0 = pure keyword (BM25)")
        temperature = st.slider("Answer creativity (temperature)", 0.0, 1.0, 0.3)

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        transcript = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "⬇️ Download transcript", transcript, file_name="documind_chat.txt",
            use_container_width=True,
        )

    st.divider()
    st.caption(
        "🔒 Privacy: documents are kept only in this browser session's memory - never "
        "written to disk or shared with other visitors."
    )

# ----------------------------------------------------------- status line --
if store.sources:
    st.markdown(
        f'<div class="status-line">🟢 {len(store.sources)} document(s) ready - ask away.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-line">⚪ No documents yet - upload one in the sidebar to get grounded, '
        'cited answers (or just say hi).</div>',
        unsafe_allow_html=True,
    )


def render_sources(sources) -> None:
    with st.expander(f"📚 {len(sources)} source(s) used"):
        max_score = max((sc.score for sc in sources), default=1.0) or 1.0
        for j, sc in enumerate(sources, start=1):
            page_info = f" · page {sc.chunk.page}" if sc.chunk.page else ""
            bar_pct = max(4, round(100 * sc.score / max_score))
            snippet = sc.chunk.text[:400] + ("..." if len(sc.chunk.text) > 400 else "")
            st.markdown(
                f'<div class="source-card"><b>[{j}] {file_icon(sc.chunk.source)} {sc.chunk.source}{page_info}</b>'
                f'<div class="relevance-track"><div class="relevance-fill" style="width:{bar_pct}%"></div></div>'
                f'{snippet}</div>',
                unsafe_allow_html=True,
            )


def generate_reply(question: str, history: list[dict]) -> dict:
    """Runs condense -> retrieve -> stream, rendering progress live in the
    current chat_message container. Returns the finished message dict."""
    api_key = resolve_api_key()
    error_text = None
    scored = []
    display_sources = []
    full_text = ""
    placeholder = st.empty()
    placeholder.markdown("🔎 _Searching your documents…_")
    try:
        standalone_q = condense_question(question, history, api_key)
        store: HybridVectorStore = st.session_state.store
        if store.chunks:
            q_vector = embed_query(standalone_q)
            scored = store.search(q_vector, standalone_q, top_k=top_k, alpha=alpha)
        placeholder.markdown("✍️ _Writing answer…_")
        for delta in stream_answer(standalone_q, scored, history, api_key, model_name, temperature):
            full_text += delta
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
        if NOT_FOUND_PHRASE not in full_text.lower():
            display_sources = scored
    except Exception as exc:
        error_text = str(exc)
        if "503" in error_text or "UNAVAILABLE" in error_text:
            reason = "Gemini's servers are temporarily overloaded"
        elif "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            reason = "the free-tier rate limit was hit"
        else:
            reason = "of an unexpected error"
        full_text = f"Sorry, something went wrong because {reason}. Please try again in a few seconds."
        placeholder.markdown(full_text)

    if display_sources:
        render_sources(display_sources)
    return {
        "role": "assistant", "content": full_text, "sources": display_sources,
        "question": question, "error": error_text,
    }


# --------------------------------------------------------------- chat area --
for i, message in enumerate(st.session_state.messages):
    avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])
        if message["role"] == "assistant" and message.get("error"):
            with st.expander("🔧 Technical details"):
                st.code(message["error"])
        is_last = i == len(st.session_state.messages) - 1
        if message["role"] == "assistant" and is_last:
            retry_label = "🔄 Retry" if message.get("error") else "🔄 Regenerate"
            if st.button(retry_label, key=f"regen_{i}"):
                st.session_state.messages.pop()
                st.session_state.regenerate_question = message["question"]
                st.rerun()

if not st.session_state.messages:
    st.markdown("#### What would you like to know?")
    examples = [
        ("📝", "Summarize the key points in bullet form."),
        ("🔎", "What is this document about?"),
        ("❓", "What questions should I ask about this content?"),
        ("📌", "List any important dates, numbers, or names mentioned."),
    ]
    cols = st.columns(2)
    for idx, (icon, example) in enumerate(examples):
        if cols[idx % 2].button(f"{icon}  {example}", use_container_width=True, key=f"example_{idx}"):
            st.session_state.pending_question = example

question = st.chat_input("Ask a question about your documents...")
st.markdown(
    '<p class="disclaimer">DocuMind can make mistakes. Double-check important information '
    "against your source documents.</p>",
    unsafe_allow_html=True,
)

is_regenerate = False
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
elif st.session_state.regenerate_question:
    question = st.session_state.regenerate_question
    st.session_state.regenerate_question = None
    is_regenerate = True

if question:
    now = time.time()
    st.session_state.query_timestamps = [t for t in st.session_state.query_timestamps if now - t < 60]
    if len(st.session_state.query_timestamps) >= MAX_QUERIES_PER_MINUTE:
        st.warning("Rate limit reached to keep the shared free API quota fair - please wait a minute.")
    elif not resolve_api_key():
        st.error("No Gemini API key configured. Add your free key in the sidebar to chat.")
    else:
        st.session_state.query_timestamps.append(now)
        if not is_regenerate:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(question)

        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            reply = generate_reply(question, history)
        st.session_state.messages.append(reply)
        st.rerun()
