"""Parsers that turn uploaded files into plain-text pages/sections.

Each loader returns a list of dicts: {"text": str, "page": int | None}.
Keeping the interface uniform lets the chunker stay format-agnostic.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader
import docx


@dataclass
class RawSection:
    text: str
    page: int | None = None


def load_pdf(file_bytes: bytes) -> list[RawSection]:
    reader = PdfReader(io.BytesIO(file_bytes))
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            sections.append(RawSection(text=text, page=i))
    return sections


def load_docx(file_bytes: bytes) -> list[RawSection]:
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    return [RawSection(text=text, page=None)] if text else []


def load_text(file_bytes: bytes) -> list[RawSection]:
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    return [RawSection(text=text, page=None)] if text else []


_LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "txt": load_text,
    "md": load_text,
    "markdown": load_text,
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS.keys())


def parse_document(filename: str, file_bytes: bytes) -> list[RawSection]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return loader(file_bytes)
