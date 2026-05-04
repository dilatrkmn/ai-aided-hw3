"""Custom chunker.

Implemented in pure Python rather than via a third-party text-splitter
library. The strategy is:

1. Strip Wikipedia section markers like ``== References ==`` so we don't
   index navigation/boilerplate text.
2. Split the cleaned text into sentences with a small regex-based splitter
   (good enough for English Wikipedia prose).
3. Greedily pack sentences into chunks of approximately ``chunk_size_words``
   words, with a sliding overlap of ``overlap_words`` words so that concepts
   that span a chunk boundary remain retrievable from either side.

A word-based budget keeps chunks within a comfortable embedding context
window without depending on a tokenizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from . import config

# Wikipedia plain-text dumps embed section headings like "== History ==".
# We treat the bibliographic / navigational tail as boilerplate we want to
# drop — these sections rarely contain factual answers and pollute retrieval.
_BOILERPLATE_HEADINGS = {
    "see also",
    "references",
    "further reading",
    "external links",
    "notes",
    "citations",
    "bibliography",
    "sources",
    "footnotes",
}

_HEADING_RE = re.compile(r"^\s*={2,}\s*(.+?)\s*={2,}\s*$", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_wikipedia_text(text: str) -> str:
    """Remove boilerplate sections and section markers from a plain-text dump."""

    # Walk through line by line, dropping anything inside a boilerplate section.
    lines = text.splitlines()
    cleaned: list[str] = []
    skip = False
    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            heading = m.group(1).strip().lower()
            skip = heading in _BOILERPLATE_HEADINGS
            # Drop the heading line itself either way — it's structure, not content.
            continue
        if skip:
            continue
        cleaned.append(line)
    # Collapse runs of blank lines.
    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter.

    Splits on whitespace following a ``.``, ``!`` or ``?`` that's followed by
    an uppercase letter or opening punctuation. Good enough for Wikipedia
    English prose; preserves abbreviations like "U.S." and "Dr." reasonably
    well because the next character isn't whitespace.
    """
    # Normalize line breaks within paragraphs but keep paragraph breaks.
    paragraphs = re.split(r"\n{2,}", text)
    sentences: list[str] = []
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_END_RE.split(para) if s.strip())
    return sentences


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single chunk of text together with bookkeeping metadata."""

    chunk_id: str   # globally unique, used as the Chroma id
    entity: str
    type: str       # "person" / "place"
    title: str
    url: str
    text: str
    chunk_index: int
    word_count: int


def chunk_document(
    *,
    entity: str,
    type: str,
    title: str,
    url: str,
    text: str,
    chunk_size_words: int = config.CHUNK_SIZE_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Chunk a single Wikipedia document into overlapping passages.

    Sentences are kept intact: we accumulate sentences until adding the next
    one would push the running word count past ``chunk_size_words``, then
    emit a chunk and slide back ``overlap_words`` words of context for the
    next chunk's prefix. This is similar to the LangChain
    RecursiveCharacterTextSplitter strategy but implemented from scratch.
    """

    cleaned = clean_wikipedia_text(text)
    sentences = split_sentences(cleaned)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0
    idx = 0

    for sent in sentences:
        sent_words = sent.split()
        n = len(sent_words)

        # If a single sentence is enormous, hard-split it on word boundaries.
        if n > chunk_size_words:
            # Flush any pending buffer first.
            if buffer:
                chunks.append(_make_chunk(buffer, idx, entity, type, title, url))
                idx += 1
                buffer, buffer_words = _carry_overlap(buffer, overlap_words)

            for start in range(0, n, chunk_size_words - overlap_words):
                window = sent_words[start : start + chunk_size_words]
                if not window:
                    break
                chunks.append(
                    _make_chunk(
                        [" ".join(window)], idx, entity, type, title, url
                    )
                )
                idx += 1
            continue

        if buffer_words + n > chunk_size_words and buffer:
            # Emit current chunk, then start a new one with overlap context.
            chunks.append(_make_chunk(buffer, idx, entity, type, title, url))
            idx += 1
            buffer, buffer_words = _carry_overlap(buffer, overlap_words)

        buffer.append(sent)
        buffer_words += n

    # Flush trailing buffer
    if buffer:
        chunks.append(_make_chunk(buffer, idx, entity, type, title, url))

    return chunks


def _carry_overlap(buffer: list[str], overlap_words: int) -> tuple[list[str], int]:
    """Take the trailing ``overlap_words`` words of ``buffer`` as the next prefix."""
    if overlap_words <= 0 or not buffer:
        return [], 0
    # Concatenate, then take the last N words.
    joined = " ".join(buffer)
    words = joined.split()
    if len(words) <= overlap_words:
        return [joined], len(words)
    tail = " ".join(words[-overlap_words:])
    return [tail], overlap_words


def _make_chunk(
    sentences: list[str],
    idx: int,
    entity: str,
    type: str,
    title: str,
    url: str,
) -> Chunk:
    text = " ".join(sentences).strip()
    word_count = len(text.split())
    slug = entity.lower().replace(" ", "_").replace("/", "_").replace("'", "")
    chunk_id = f"{type}__{slug}__{idx:04d}"
    return Chunk(
        chunk_id=chunk_id,
        entity=entity,
        type=type,
        title=title,
        url=url,
        text=text,
        chunk_index=idx,
        word_count=word_count,
    )


def chunk_documents(documents: Iterable) -> list[Chunk]:
    """Chunk a collection of ``ingest.Document`` objects."""
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(
            chunk_document(
                entity=doc.entity,
                type=doc.type,
                title=doc.title,
                url=doc.url,
                text=doc.text,
            )
        )
    return chunks
