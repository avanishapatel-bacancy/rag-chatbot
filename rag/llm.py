"""Answer generation via the free Gemini API, grounded in retrieved chunks."""
from __future__ import annotations

import time

from google import genai
from google.genai import types

from rag.vectorstore import ScoredChunk

_TRANSIENT_MARKERS = ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
_MAX_STREAM_RETRIES = 3

NOT_FOUND_PHRASE = "i couldn't find this in the uploaded documents"

SYSTEM_PROMPT = f"""You are a precise, helpful assistant that answers questions using ONLY \
the numbered CONTEXT excerpts below, which come from documents the user uploaded.

Rules:
- Base your answer strictly on the CONTEXT. Never use outside knowledge to fill gaps.
- Synthesizing counts as "in the CONTEXT": for meta-questions like "what is this about", \
"summarize this", or "give an overview", describe the topic/purpose of the CONTEXT in your \
own words even if no single excerpt states it explicitly. You are not restricted to quoting \
or paraphrasing exact wording.
- Only use the fallback below when the CONTEXT has nothing plausibly relevant to the question \
at all (e.g. it asks about a topic, fact, or entity the CONTEXT never touches on). If in doubt, \
prefer answering from what IS there over refusing.
- If the answer is truly not contained in the CONTEXT, say plainly: \
"{NOT_FOUND_PHRASE.capitalize()}." Do not guess or speculate.
- Cite every claim with the matching bracketed source number, e.g. [1] or [2][3].
- Be concise and well-structured. Use short paragraphs or bullet points.
- If the user's message is purely conversational (a greeting, thanks, etc.) and unrelated \
to the documents, respond naturally without forcing citations.
"""

CONDENSE_PROMPT = """Given the chat history and a follow-up message, rewrite the follow-up \
as a standalone question that contains all the context needed to search a document \
database. If the follow-up is already standalone, or is just a greeting/small talk, \
return it unchanged. Reply with ONLY the rewritten question, no preamble.

Chat history:
{history}

Follow-up message: {question}

Standalone question:"""


def _format_history(history: list[dict], max_turns: int = 4) -> str:
    trimmed = history[-max_turns:]
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in trimmed) or "(none)"


_REFERENTIAL_WORDS = {
    "it", "this", "that", "these", "those", "they", "them", "he", "she",
    "previous", "above", "earlier", "again", "more", "further", "same",
}


def _needs_condensing(question: str) -> bool:
    """Most follow-ups are already standalone. Only pay for the extra Gemini
    round-trip when the question is short or leans on a pronoun/reference
    that likely points back at the conversation -- this is what actually
    makes multi-turn chat feel slow, so skip it whenever we safely can."""
    words = [w.strip(".,!?").lower() for w in question.split()]
    if len(words) <= 6:
        return True
    return any(w in _REFERENTIAL_WORDS for w in words)


def condense_question(question: str, history: list[dict], api_key: str, model_name: str) -> str:
    if not history or not _needs_condensing(question):
        return question
    client = genai.Client(api_key=api_key)
    prompt = CONDENSE_PROMPT.format(history=_format_history(history), question=question)
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        rewritten = (response.text or "").strip()
        return rewritten or question
    except Exception:
        return question


def build_context_block(scored_chunks: list[ScoredChunk]) -> str:
    blocks = []
    for i, sc in enumerate(scored_chunks, start=1):
        page_info = f", page {sc.chunk.page}" if sc.chunk.page else ""
        blocks.append(f"[{i}] (source: {sc.chunk.source}{page_info})\n{sc.chunk.text}")
    return "\n\n".join(blocks)


def stream_answer(
    question: str,
    scored_chunks: list[ScoredChunk],
    history: list[dict],
    api_key: str,
    model_name: str,
    temperature: float = 0.3,
):
    """Yields text deltas from Gemini, grounded in the given context chunks.

    Retries transient server errors (overload/rate-limit) by restarting the
    stream from scratch, but only while nothing has been yielded to the
    caller yet -- once partial text has been shown, a retry could duplicate
    it, so at that point the error is simply raised.
    """
    client = genai.Client(api_key=api_key)
    context_block = build_context_block(scored_chunks) if scored_chunks else "(no relevant context found)"

    convo = []
    for turn in history[-6:]:
        role = "user" if turn["role"] == "user" else "model"
        convo.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))

    prompt = f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"
    delay = 1.0

    for attempt in range(_MAX_STREAM_RETRIES):
        chat = client.chats.create(
            model=model_name,
            history=convo,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
            ),
        )
        emitted_any = False
        try:
            for chunk in chat.send_message_stream(prompt):
                if chunk.text:
                    emitted_any = True
                    yield chunk.text
            return
        except Exception as exc:
            is_transient = any(marker in str(exc) for marker in _TRANSIENT_MARKERS)
            if emitted_any or not is_transient or attempt == _MAX_STREAM_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
