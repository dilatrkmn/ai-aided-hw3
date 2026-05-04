"""End-to-end ingestion script.

Usage:
    python -m scripts.ingest_data            # ingest, chunk, embed, index
    python -m scripts.ingest_data --reset    # drop existing collection first
    python -m scripts.ingest_data --refetch  # re-download Wikipedia even if cached
"""

from __future__ import annotations

import argparse
import sys

from src import chunker, ingest, vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia data into the local RAG store.")
    parser.add_argument("--reset", action="store_true", help="Wipe the Chroma collection before indexing.")
    parser.add_argument("--refetch", action="store_true", help="Re-download articles even if cached.")
    args = parser.parse_args()

    if args.reset:
        print("Resetting Chroma collection...")
        vector_store.reset_collection()

    print("Step 1/3: Fetching Wikipedia articles...")
    docs = ingest.ingest_all(skip_existing=not args.refetch)

    if not docs:
        print("No newly ingested documents; loading existing local documents...")
        docs = ingest.load_all_documents()

    if not docs:
        print("No documents available. Aborting.", file=sys.stderr)
        return 1

    print(f"\nStep 2/3: Chunking {len(docs)} documents...")
    chunks = chunker.chunk_documents(docs)
    print(f"  produced {len(chunks)} chunks")

    print("\nStep 3/3: Embedding + indexing chunks (this loads the embedding model)...")
    n = vector_store.index_chunks(chunks)
    print(f"  indexed {n} chunks into collection {vector_store.config.COLLECTION_NAME!r}")

    stats = vector_store.collection_stats()
    print(
        f"\nDone. Collection now contains {stats['total']} chunks "
        f"(by type: {stats['by_type']}) across {len(stats['entities'])} entities."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
