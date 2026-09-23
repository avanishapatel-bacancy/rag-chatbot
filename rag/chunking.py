"""Paragraph-aware recursive text chunking.

Splits on paragraph boundaries first (keeps ideas intact), falls back to
sentence/character splitting for oversized paragraphs, and stitches small
pieces back together up to chunk_size with a sliding overlap so context
never gets cut off at a hard boundary.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from rag.document_loader import RawSection

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_oversized(paragraph: str, chunk_size: int) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(paragraph)
    pieces, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                pieces.append(current)
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size):
                    pieces.append(sentence[i : i + chunk_size])
                current = ""
            else:
                current = sentence
    if current:
        pieces.append(current)
    return pieces


def _merge_with_overlap(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer}\n\n{piece}".strip() if buffer else piece
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            overlap_tail = buffer[-chunk_overlap:] if chunk_overlap and buffer else ""
            buffer = f"{overlap_tail}\n\n{piece}".strip() if overlap_tail else piece
    if buffer:
        merged.append(buffer)
    return merged


def chunk_documents(
    sections_by_source: dict[str, list[RawSection]],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """sections_by_source: {filename: [RawSection, ...]}"""
    all_chunks: list[Chunk] = []
    for source, sections in sections_by_source.items():
        for section in sections:
            paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]
            atomic_pieces: list[str] = []
            for paragraph in paragraphs:
                if len(paragraph) <= chunk_size:
                    atomic_pieces.append(paragraph)
                else:
                    atomic_pieces.extend(_split_oversized(paragraph, chunk_size))
            merged = _merge_with_overlap(atomic_pieces, chunk_size, chunk_overlap)
            for text in merged:
                all_chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=text,
                        source=source,
                        page=section.page,
                        chunk_index=len(all_chunks),
                    )
                )
    return all_chunks
