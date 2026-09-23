"""In-memory hybrid retriever: dense (Gemini embeddings, cosine) + sparse (BM25).

Deliberately NOT a shared/persistent database. Each Streamlit session builds
its own store in st.session_state, so one visitor's uploaded documents are
never visible to another visitor. Fine at the scale of a single user's
document set (brute-force numpy cosine similarity), and avoids extra infra
(no FAISS/Chroma service to run or deploy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridVectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        seen = []
        for c in self.chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        if self._embeddings is None:
            self._embeddings = norm
        else:
            self._embeddings = np.vstack([self._embeddings, norm])
        self.chunks.extend(chunks)
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

    def remove_source(self, source: str) -> None:
        keep_idx = [i for i, c in enumerate(self.chunks) if c.source != source]
        self.chunks = [self.chunks[i] for i in keep_idx]
        if self._embeddings is not None and keep_idx:
            self._embeddings = self._embeddings[keep_idx]
        elif not keep_idx:
            self._embeddings = None
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks]) if self.chunks else None

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        span = scores.max() - scores.min()
        if span < 1e-10:
            return np.zeros_like(scores)
        return (scores - scores.min()) / span

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.6,
    ) -> list[ScoredChunk]:
        """alpha weights dense-similarity vs BM25 lexical score (0=lexical only, 1=dense only)."""
        if not self.chunks:
            return []
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        dense_scores = self._embeddings @ q_norm
        lexical_scores = np.array(self._bm25.get_scores(_tokenize(query_text)))

        fused = alpha * self._normalize(dense_scores) + (1 - alpha) * self._normalize(lexical_scores)
        top_idx = np.argsort(-fused)[:top_k]
        return [ScoredChunk(chunk=self.chunks[i], score=float(fused[i])) for i in top_idx]
