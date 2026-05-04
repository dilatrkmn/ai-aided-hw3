# Production Deployment Recommendations

The implementation in this repo is a single-laptop, single-user prototype. This document describes what would change to take it to a robust shared deployment.

## 1. Move the LLM and embedder behind a service

A laptop-bound Ollama process is fine for the demo but doesn't scale. In production:

- Run inference as a managed service. Either keep Ollama and put it behind a small autoscaling group, or move to a hosted inference engine (vLLM, TGI, llama.cpp server) on GPU instances. Front it with a load balancer and a request queue so a slow request can't head-of-line block faster ones.
- Move the embedder to the same service tier. Embedding on the application pod adds latency and binds the pod's memory to the model size — a dedicated embedding service can batch across users for higher throughput.
- Pin model versions explicitly. The current code accepts whatever tag is locally available; in production every deploy should specify the exact model digest so two pods never disagree on what `llama3.2:3b` means.

## 2. Replace single-file Chroma with a managed vector DB

Persistent Chroma writes a SQLite + flat-file store under `data/chroma_db/`. That works for a 40-entity demo but breaks down on three axes:

- **Concurrency.** SQLite locks during writes. Two ingest jobs at once will fight.
- **Operational hygiene.** No replication, no backups, no point-in-time restore.
- **Scale.** HNSW in-process is fine for tens of thousands of vectors but you want a dedicated service past that.

Move to one of: Chroma in client/server mode, Qdrant, Weaviate, Pinecone, or PGVector if Postgres is already in your stack. Whatever you pick, run it as a stateful service with replication and snapshot backups.

## 3. Treat ingestion as a pipeline, not a script

`scripts/ingest_data.py` is a synchronous batch. In production:

- Run ingestion on a schedule (Airflow / Dagster / a cron job in K8s) so the corpus stays fresh.
- Make every step idempotent and resumable. The current code is mostly there (raw cache + upsert) but doesn't track Wikipedia revision IDs, so it can't tell when an article actually changed.
- Capture provenance — store each chunk's source URL, fetch timestamp, and Wikipedia revision ID alongside the embedding. The UI already shows the URL; surfacing the revision lets you say "this answer is based on the version of the article from <date>".
- Consider a streaming ingestion path for users who paste their own URLs.

## 4. Improve retrieval quality

The current retriever does pure cosine similarity over a single embedding model. For better answer quality:

- **Hybrid retrieval.** Combine BM25 over the same chunks with the dense vectors. Sparse retrieval is much better at exact-name matches (e.g. "Hagia Sophia" rather than a paraphrase).
- **Re-ranking.** Pass the top ~20 candidates through a small cross-encoder (`bge-reranker`, `ms-marco-MiniLM`) before sending the top 5 to the LLM. This is the single highest-leverage retrieval upgrade.
- **Query rewriting.** Use the LLM to expand the query (HyDE-style or simple paraphrase) before embedding. Especially helps comparison queries.
- **Metadata-filtered retrieval as a function of routed entities.** Today the router decides `type=person|place`. Better: when an entity name matches, scope the search to that exact `entity` first and only fall back to the partition if low-similarity.

## 5. Replace rule-based routing with a small classifier

The current router is keyword-based and works well on the demo set, but it'll miss queries phrased unusually (e.g. "the painter who lost his ear"). Two practical paths:

- A small classifier (logistic regression on TF-IDF, or a fine-tuned distilled model) trained on a few hundred labelled questions.
- LLM-based routing — a tiny prompt to the local model returning JSON `{type, entities}`. Adds latency but is more robust to phrasing.

Either way, keep the rule-based path as a fast / deterministic fallback.

## 6. Productionise the API surface

Right now the app has only a Streamlit UI and a CLI. For shared use:

- Wrap `src.rag.answer` (and a streaming variant) in a FastAPI service with `/ask` and `/sources/{chunk_id}` endpoints. This decouples the model/data layer from the front-end.
- Add an OpenAPI schema, request validation, and proper auth (OAuth or signed JWTs) if the corpus is anything other than fully public.
- Cache by `(query, model)` so the same question doesn't re-pay generation cost. A small Redis with a 24-hour TTL is plenty.

## 7. Observability

- Structured request logs with: query, routing decision, chunk IDs retrieved, similarity scores, model name, generation latency, tokens in/out, and a stable request ID. `logger = logging.getLogger("wikirag")` already gets you 80% of the way there.
- Trace each request end-to-end (OpenTelemetry, exported to Jaeger/Tempo).
- Track quality metrics offline: a periodic batch that runs a fixed eval set (golden Q&A pairs) and tracks (a) retrieval recall@k against gold chunks and (b) answer faithfulness via an LLM-as-judge or string-overlap heuristic.

## 8. Safety & content controls

The corpus is curated, so abuse is limited, but:

- Always filter out chunks whose `type` doesn't match the routing decision when the user explicitly scoped the question (e.g. "what places…" should never quote a person chunk in the answer).
- Add a moderation pass — a small classifier flagging prompt-injection-like content from Wikipedia (rare for the entities here, but cheap to add).
- Treat the assistant's output as untrusted until verified — the system prompt's `"I don't know."` rule plus showing sources is the user's verification path; don't suppress the source panel.

## 9. Cost & capacity planning

- Embedding 40 articles is free; embedding millions is not. Budget for it explicitly and consider a smaller model (`bge-small`, `gte-small`) if you scale the corpus by 100×.
- LLM cost is the dominant variable. Decide up-front whether you're running a self-hosted GPU pool (CapEx, predictable cost) or paying per token. The current code's clean separation between `src.llm` and the rest means swapping providers is a one-file change.

## 10. Testing

The repo has no unit tests today. The minimum production set:

- Unit tests for `chunker` (chunk size respected, overlap correct, boilerplate stripped).
- Unit tests for `router` (each example query → expected `type`).
- An offline eval harness with ground-truth (question, expected entities, expected answer keywords) that can be run in CI.
- A smoke test that ingests two entities, asks one question per type, and asserts on the routing decision and that retrieval returned a chunk for the right entity.
