"""Chroma-backed vector store.

We use a single Chroma collection (``wiki_rag``) and tag each chunk with a
``type`` metadata field of either ``"person"`` or ``"place"``. This is the
"Option B" design from the assignment — chosen because it keeps the storage
layer simple while letting us answer mixed queries about both people and
places without merging across collections at query time.

Cosine similarity is configured at the collection level. Embeddings are
normalised in :mod:`src.embeddings`, so cosine and dot-product rankings agree.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import chromadb
from chromadb.config import Settings

from . import config
from .chunker import Chunk
from .embeddings import embed_texts


# ---------------------------------------------------------------------------
# Client / collection helpers
# ---------------------------------------------------------------------------


def get_client() -> chromadb.api.client.Client:
    """Return a persistent Chroma client rooted at ``data/chroma_db/``."""
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_or_create_collection(client: chromadb.api.client.Client | None = None):
    """Get (or create) the single shared collection."""
    client = client or get_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def index_chunks(chunks: Sequence[Chunk], *, batch_size: int = 64) -> int:
    """Embed and upsert ``chunks`` into the Chroma collection.

    Returns the number of chunks written.
    """
    if not chunks:
        return 0

    coll = get_or_create_collection()
    n = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [c.chunk_id for c in batch]
        documents = [c.text for c in batch]
        metadatas = [
            {
                "entity": c.entity,
                "type": c.type,
                "title": c.title,
                "url": c.url,
                "chunk_index": c.chunk_index,
                "word_count": c.word_count,
            }
            for c in batch
        ]
        embeddings = embed_texts(documents)
        coll.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        n += len(batch)
    return n


def reset_collection() -> None:
    """Drop and recreate the collection (used by the 'Reset' UI button)."""
    client = get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def collection_stats() -> dict:
    """Return small bookkeeping stats for the UI."""
    coll = get_or_create_collection()
    total = coll.count()
    if total == 0:
        return {"total": 0, "by_type": {}, "entities": []}

    # Pull metadata only (cheap) to count by type and list distinct entities.
    sample = coll.get(include=["metadatas"])
    by_type: dict[str, int] = {}
    entities: set[str] = set()
    for meta in sample["metadatas"] or []:
        by_type[meta.get("type", "unknown")] = by_type.get(meta.get("type", "unknown"), 0) + 1
        if "entity" in meta:
            entities.add(meta["entity"])
    return {
        "total": total,
        "by_type": by_type,
        "entities": sorted(entities),
    }


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


def query(
    query_embedding: list[float],
    *,
    top_k: int = config.TOP_K,
    type_filter: str | None = None,
) -> list[dict]:
    """Run a similarity search and return a flat list of result dicts.

    ``type_filter`` may be ``"person"``, ``"place"``, or ``None`` (no filter).
    """
    coll = get_or_create_collection()
    where = {"type": type_filter} if type_filter else None

    res = coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    out: list[dict] = []
    if not res["ids"] or not res["ids"][0]:
        return out
    for i, _id in enumerate(res["ids"][0]):
        out.append(
            {
                "id": _id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                # Chroma returns cosine *distance* (1 - cosine_similarity);
                # we expose a similarity for nicer UI display.
                "distance": res["distances"][0][i],
                "similarity": 1.0 - res["distances"][0][i],
            }
        )
    return out
