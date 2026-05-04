"""RAG pipeline: prompt assembly + grounded answer generation.

The pipeline is intentionally small: it composes a system prompt that
instructs the model to ground answers in the supplied context and to say
"I don't know." when the context doesn't contain the answer. Each retrieved
chunk is passed in as a numbered ``[Source N]`` block so the LLM can cite
sources in its reply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from . import config, llm
from .retriever import RetrievalResult, retrieve


SYSTEM_PROMPT = """You are a careful assistant that answers questions about famous people and famous places.

You MUST follow these rules:
1. Answer ONLY using facts that appear in the provided <context> blocks.
2. If the context does not contain enough information to answer the question, reply exactly: "I don't know."
3. Do NOT invent facts, dates, names, or numbers.
4. Prefer concise answers (2-5 sentences) unless the user explicitly asks for more detail.
5. When helpful, cite sources inline by their number, like "[Source 2]".
6. If the user asks a comparison question, structure the answer so each entity is covered."""


@dataclass
class Answer:
    text: str
    retrieval: RetrievalResult


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no context retrieved)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        header = f"[Source {i}] {meta.get('title', meta.get('entity', '?'))} ({meta.get('type', '?')})"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


def _build_prompt(query: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    return (
        f"<context>\n{context}\n</context>\n\n"
        f"User question: {query}\n\n"
        "Answer the question using only the context above. "
        "If the answer is not in the context, reply exactly: I don't know."
    )


def answer(
    query: str,
    *,
    model: str | None = None,
    top_k: int = config.TOP_K,
    temperature: float = 0.2,
) -> Answer:
    """Full RAG: retrieve, prompt, generate. Returns an :class:`Answer`."""
    retrieval = retrieve(query, top_k=top_k)

    # Short-circuit: if retrieval returned nothing, don't even bother the LLM.
    if not retrieval.chunks:
        return Answer(text="I don't know.", retrieval=retrieval)

    prompt = _build_prompt(query, retrieval.chunks)
    text = llm.generate(
        prompt,
        model=model,
        system=SYSTEM_PROMPT,
        temperature=temperature,
    )
    return Answer(text=text, retrieval=retrieval)


def answer_stream(
    query: str,
    *,
    model: str | None = None,
    top_k: int = config.TOP_K,
    temperature: float = 0.2,
) -> tuple[RetrievalResult, Iterator[str]]:
    """Streaming variant. Returns the retrieval result up-front and a token iterator."""
    retrieval = retrieve(query, top_k=top_k)

    if not retrieval.chunks:
        def _idk() -> Iterator[str]:
            yield "I don't know."
        return retrieval, _idk()

    prompt = _build_prompt(query, retrieval.chunks)
    stream = llm.generate_stream(
        prompt,
        model=model,
        system=SYSTEM_PROMPT,
        temperature=temperature,
    )
    return retrieval, stream
