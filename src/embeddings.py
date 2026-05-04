"""Local embedding model wrapper.

Uses ``sentence-transformers`` (``all-MiniLM-L6-v2`` by default) to produce
384-dimensional vectors entirely on the user's machine. The model is
downloaded once on first use and cached under ``~/.cache/huggingface``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from . import config


@lru_cache(maxsize=1)
def get_embedder():
    """Lazy-load the sentence-transformers model exactly once per process."""
    # Imported lazily so the rest of the codebase can be inspected without
    # paying the (~80 MB) model download just to read --help.
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def embed_texts(texts: Iterable[str], *, batch_size: int = 64) -> list[list[float]]:
    """Embed an iterable of strings into a list of float lists."""
    model = get_embedder()
    arr = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity becomes dot product
    )
    return arr.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
