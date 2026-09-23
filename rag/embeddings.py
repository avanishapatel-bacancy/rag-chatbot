"""Local, free, unlimited embeddings via fastembed (ONNX runtime, no torch/GPU).

Running the embedding model on-device -- instead of calling a hosted
embedding API -- means document processing never depends on any provider's
free-tier quota or billing plan. The model weights (~130MB) download once
from Hugging Face on first use and are cached locally after that; every
embedding call afterwards is local CPU inference with no rate limit.
"""
from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_BATCH_SIZE = 32

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _model


def embed_documents(texts: list[str], progress_cb=None) -> np.ndarray:
    """Embed chunk texts for storage. Returns an (N, D) float32 array."""
    model = _get_model()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        vectors.extend(list(model.embed(batch)))
        if progress_cb:
            progress_cb(min(i + _BATCH_SIZE, len(texts)), len(texts))
    return np.array(vectors, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    model = _get_model()
    vec = list(model.query_embed([text]))[0]
    return np.array(vec, dtype="float32")
