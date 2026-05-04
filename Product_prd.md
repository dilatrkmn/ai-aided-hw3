# Product PRD — Local Wikipedia RAG Assistant

This PRD describes the system in enough detail for an AI coding assistant (or a fresh engineer) to rebuild it from scratch.

## 1. Problem statement

Build a fully local, ChatGPT-style assistant that can answer factual questions about a curated set of famous people and famous places, grounded in Wikipedia content. The system must run on a single laptop with no external LLM, embedding, or retrieval API.

## 2. Goals

1. Ingest Wikipedia content for **at least 20 famous people and 20 famous places** including a fixed required set (see `src/config.py`).
2. Index that content in a local vector database so each user query can be answered from retrieved context.
3. Generate the final answer with a local LLM, refusing to answer (with `"I don't know."`) when retrieval doesn't surface a supporting fact.
4. Provide a chat UI that exposes the retrieved sources and lets the user reset state.

## 3. Non-goals

- Real-time / scheduled refresh of Wikipedia content.
- Multi-user concurrency, authentication, or cloud deployment.
- Open-domain QA outside the curated entity list.
- Multi-modal input (images, audio).

## 4. Users

- The course instructor running the project from the README.
- The student during demo and development.

## 5. User journeys

**J1 — First-time setup.** User clones the repo, installs dependencies, runs `ollama pull llama3.2:3b` and `python -m scripts.ingest_data`, then `streamlit run app.py`. The app loads with a populated index and a green "Ollama OK" indicator.

**J2 — Person query.** User asks "Why is Nikola Tesla famous?". The system routes to the person partition, retrieves 5 chunks about Tesla, and the LLM produces a 2–4 sentence grounded summary with `[Source N]` citations.

**J3 — Place query.** User asks "Where is Mount Everest?". Routed to the place partition.

**J4 — Mixed query.** User asks "Compare Albert Einstein and Nikola Tesla." Both names match the people list; the router emits `type=both` (since both are people). User asks "Which famous place is located in Turkey?": no entity match, place cues fire → `type=place`, corpus search returns Hagia Sophia / Cappadocia chunks.

**J5 — Out-of-corpus query.** User asks "Who is the president of Mars?". Retrieval returns chunks with low similarity but Chroma still returns them; the system prompt's "I don't know." rule causes the LLM to refuse.

## 6. Functional requirements

### F1. Ingestion

- Source: Wikipedia REST/MediaWiki API (`api.php?action=query&prop=extracts&explaintext=1&redirects=1`).
- Persist raw text + canonical title + URL as `data/raw/<type>__<slug>.json`.
- Bookkeeping table in `data/wikirag.sqlite` listing each ingested entity, type, char count, and timestamp.
- Re-runs are idempotent — already-cached files are skipped unless `--refetch`.

### F2. Chunking

- Pure-Python sentence-aware chunker.
- Default: ~220 words per chunk, 40-word overlap.
- Drops boilerplate sections (References, See also, Further reading, External links, Notes, Citations, Bibliography, Sources, Footnotes).
- Hard-splits sentences longer than the chunk budget on word boundaries.

### F3. Embedding & storage

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 dim, normalized).
- Vector store: a single persistent Chroma collection `wiki_rag` with cosine distance.
- Chunk metadata: `entity`, `type` (`person`|`place`), `title`, `url`, `chunk_index`, `word_count`.

### F4. Retrieval & routing

- Rule-based router classifies each query as `person`, `place`, `both`, or `unknown`.
  - Direct entity-name match (full name or distinctive last name) is a strong signal.
  - Cue-word match scores augment the decision (e.g. "where", "located" → place).
- Retriever embeds the query once and runs a Chroma similarity search with `where={"type": ...}` when applicable.
- For `type=both`, retrieves `top_k_per_type` chunks from each partition then merges by similarity desc.
- Returns up to `top_k` chunks (default 5).

### F5. Generation

- Local model via Ollama HTTP API. Default `llama3.2:3b`. User-configurable via env var or sidebar.
- System prompt enforces grounded answers, `"I don't know."` fallback, and `[Source N]` citation style.
- Streaming token-by-token rendering in the UI.
- Empty-retrieval short-circuit: skip the LLM entirely and return `"I don't know."`.

### F6. Chat interface

- Streamlit primary UI (`app.py`):
  - Chat-style transcript with user / assistant bubbles.
  - Sidebar: model selector, `top_k`, temperature, "show retrieved context" toggle, Ollama / index status, "clear chat history" button.
  - Expandable "Retrieved context & routing" panel under each assistant turn showing routing decision, rationale, and each source chunk (title, type, similarity, link, text).
- CLI fallback (`cli.py`) with `/context`, `/clear`, `/quit` commands.

## 7. Non-functional requirements

- **Locality.** No outbound LLM/embedding API. Wikipedia is fetched only at ingest time.
- **Reproducibility.** A fresh clone + the README's three commands (`pip install`, `ollama pull`, `python -m scripts.ingest_data`) must yield a working app.
- **Latency target.** End-to-end answer in < 10 s on a modern laptop with `llama3.2:3b` (warm cache). Embedding + retrieval together < 200 ms.
- **Footprint.** Index for 40 entities × ~30 chunks fits comfortably in memory (~5 MB of vectors).

## 8. Acceptance criteria

1. `python -m scripts.ingest_data` succeeds and reports ≥ 20 people and ≥ 20 places.
2. `streamlit run app.py` shows green Ollama status and non-zero index stats.
3. Each example query in the README produces a grounded answer that cites at least one source — except the failure-case queries, which produce `"I don't know."`.
4. Toggling "Show retrieved context" reveals chunk text and metadata in the UI.

## 9. Out-of-scope (deferred)

- Re-ranking with a cross-encoder.
- Multilingual support.
- Conversational memory across turns (each turn is currently independent).
- Eval harness with ground-truth Q&A pairs.

## 10. Tech stack

- Python 3.10+
- requests, sentence-transformers, chromadb, streamlit
- Ollama (separate process)
