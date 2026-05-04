"""Minimal CLI fallback chat for the local Wikipedia RAG assistant.

Run with:
    python cli.py

Type ``/clear`` to reset the conversation, ``/context`` to toggle showing
retrieved chunks, or ``/quit`` to exit.
"""

from __future__ import annotations

import sys

from src import config, llm, vector_store
from src.rag import answer_stream


def main() -> int:
    print("=" * 60)
    print("Local Wikipedia RAG Assistant (CLI)")
    print("=" * 60)

    ok, msg = llm.is_available(model=config.LLM_MODEL_NAME)
    if not ok:
        print(f"[!] {msg}")
        print("    Run `ollama serve` and `ollama pull llama3.2:3b` and try again.")
        return 1

    stats = vector_store.collection_stats()
    if stats["total"] == 0:
        print("[!] Vector store is empty. Run: python -m scripts.ingest_data")
        return 1
    print(
        f"Loaded {stats['total']} chunks "
        f"({stats['by_type'].get('person', 0)} person / "
        f"{stats['by_type'].get('place', 0)} place) "
        f"across {len(stats['entities'])} entities."
    )
    print("Commands: /clear, /context, /quit")
    print("-" * 60)

    show_context = False

    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not q:
            continue
        if q in {"/quit", "/exit"}:
            return 0
        if q == "/clear":
            print("(history cleared — this CLI is stateless across turns anyway)")
            continue
        if q == "/context":
            show_context = not show_context
            print(f"(retrieved-context display = {show_context})")
            continue

        retrieval, stream = answer_stream(q, model=config.LLM_MODEL_NAME)
        print("Assistant: ", end="", flush=True)
        for tok in stream:
            sys.stdout.write(tok)
            sys.stdout.flush()
        print()

        if show_context:
            print(f"\n  routing = {retrieval.routing.type} — {retrieval.routing.rationale}")
            for i, ch in enumerate(retrieval.chunks, start=1):
                meta = ch["metadata"]
                print(
                    f"  [Source {i}] {meta.get('title', '?')} "
                    f"(type={meta.get('type')}, sim={ch['similarity']:.3f})"
                )


if __name__ == "__main__":
    raise SystemExit(main())
