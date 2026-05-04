"""High-level retrieval orchestration.

Glues the router and the vector store together: classify the query, decide
which partition(s) to hit, embed once, and merge results.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, vector_store
from .embeddings import embed_query
from .router import Routing, route


@dataclass
class RetrievalResult:
    routing: Routing
    chunks: list[dict]   # ranked, deduped, ready for the LLM prompt


def retrieve(query: str, *, top_k: int = config.TOP_K) -> RetrievalResult:
    """Retrieve the most relevant chunks for ``query``.

    The number of chunks returned is approximately ``top_k`` overall — for
    "both" queries we pull from each partition separately and merge.
    """
    routing = route(query)
    q_emb = embed_query(query)

    if routing.type == "person":
        chunks = vector_store.query(q_emb, top_k=top_k, type_filter="person")
    elif routing.type == "place":
        chunks = vector_store.query(q_emb, top_k=top_k, type_filter="place")
    elif routing.type == "both":
        per_type = max(2, config.TOP_K_PER_TYPE_WHEN_BOTH)
        chunks = vector_store.query(q_emb, top_k=per_type, type_filter="person")
        chunks += vector_store.query(q_emb, top_k=per_type, type_filter="place")
        # Stable sort by similarity desc
        chunks.sort(key=lambda c: c["similarity"], reverse=True)
    else:  # "unknown"
        chunks = vector_store.query(q_emb, top_k=top_k, type_filter=None)

    # Deduplicate by chunk id while preserving order.
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in chunks:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        deduped.append(c)

    return RetrievalResult(routing=routing, chunks=deduped[:top_k])
