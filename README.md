# Local Wikipedia RAG Assistant — BLG483E HW3

A small ChatGPT-style system that answers questions about famous people and famous places using **only local resources**: a local embedding model, a local vector database, and a local LLM. No external API is called for retrieval, embedding, or generation.

The pipeline:

```
Wikipedia REST API  ──►  raw text  ──►  custom chunker  ──►  sentence-transformers
                                                              ▼
                                                          Chroma collection
                                                          (type=person|place)
                                                              ▼
        user query  ──►  rule-based router  ──►  retriever  ──►  Ollama (llama3.2:3b)  ──►  answer
```

## Repository layout

```
.
├── app.py                    # Streamlit chat UI (main entry)
├── cli.py                    # Minimal CLI fallback chat
├── scripts/
│   └── ingest_data.py        # Fetch + chunk + embed + index
├── src/
│   ├── config.py             # Entity lists, paths, model names, hyper-params
│   ├── ingest.py             # Wikipedia REST API ingestion
│   ├── chunker.py            # Sentence-aware overlap chunker (pure Python)
│   ├── embeddings.py         # sentence-transformers wrapper
│   ├── vector_store.py       # Chroma persistent collection
│   ├── router.py             # Rule-based person/place/both/unknown router
│   ├── retriever.py          # Routing + similarity search orchestration
│   ├── llm.py                # Ollama HTTP client (blocking + streaming)
│   └── rag.py                # Prompt assembly + grounded generation
├── data/                     # Created on first run (gitignored)
│   ├── raw/                  #   raw Wikipedia JSON dumps
│   ├── chroma_db/            #   persistent vector store
│   └── wikirag.sqlite        #   ingestion bookkeeping
├── requirements.txt
├── Product_prd.md            # PRD describing what to build (for AI)
├── recommendation.md         # Production deployment recommendations
└── README.md
```

## 1. Install dependencies

You need Python 3.10+ and [Ollama](https://ollama.com/) installed.

```bash
# Clone the repo, then from the project root:
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first time the embedder runs, `sentence-transformers` will download `all-MiniLM-L6-v2` (~90 MB) into `~/.cache/huggingface/`.

## 2. Run the local LLM

In a separate terminal:

```bash
ollama serve                       # starts the local LLM server on :11434
ollama pull llama3.2:3b            # ~2 GB, used as the default generation model
```

Other supported defaults: `phi3`, `mistral`. Override with the `WIKIRAG_LLM_MODEL` env var or change it in the Streamlit sidebar at runtime.

## 3. Ingest the data

```bash
python -m scripts.ingest_data
```

This will:

1. Download Wikipedia plain-text extracts for every entity in `src/config.py` (20+ people, 20+ places). Cached in `data/raw/` so re-runs are instant.
2. Clean each article (drops References / See also / External links sections), split it into ~220-word overlapping chunks.
3. Embed every chunk with `sentence-transformers/all-MiniLM-L6-v2` and upsert into a single Chroma collection with `type=person` or `type=place` metadata.

Useful flags:

```bash
python -m scripts.ingest_data --reset      # wipe the Chroma collection first
python -m scripts.ingest_data --refetch    # re-download articles even if cached
```

## 4. Start the chat application

### Streamlit (recommended)

```bash
python -m streamlit run app.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`). The sidebar shows index / Ollama health, lets you swap models, change `top_k`, toggle retrieved-context display, and clear chat history.

### CLI

```bash
python cli.py
```

Commands inside the CLI: `/context` toggles showing the retrieved chunks, `/clear` resets, `/quit` exits.

## Example queries

**People**

- Who was Albert Einstein and what is he known for?
- What did Marie Curie discover?
- Why is Nikola Tesla famous?
- Compare Lionel Messi and Cristiano Ronaldo.
- What is Frida Kahlo known for?

**Places**

- Where is the Eiffel Tower located?
- Why is the Great Wall of China important?
- What is Machu Picchu?
- What was the Colosseum used for?
- Where is Mount Everest?

**Mixed**

- Which famous place is located in Turkey?
- Which person is associated with electricity?
- Compare Albert Einstein and Nikola Tesla.
- Compare the Eiffel Tower and the Statue of Liberty.

**Failure cases (should answer "I don't know.")**

- Who is the president of Mars?
- Tell me about a random unknown person John Doe.

## Design notes

**Vector store layout — single collection with a `type` metadata field (Option B).** Compared to two separate collections (Option A), one collection keeps the codebase simpler, makes mixed/comparison queries trivial (pull `top_k` from each partition with a `where={"type": ...}` filter and merge), and avoids duplicating ID spaces. The trade-off is a slightly larger HNSW graph, which at 40 entities × ~30 chunks each is negligible.

**Chunking — sentence-aware, ~220 words with 40-word overlap.** Implemented from scratch in `src/chunker.py` rather than via a library text-splitter. Boilerplate sections (References, See also, External links, etc.) are stripped before chunking so retrieval focuses on prose. The overlap preserves coreferences across chunk boundaries (a sentence ending one chunk and continuing the next stays retrievable from either side).

**Routing — rule-based, deliberately simple.** Direct entity-name match against `config.PEOPLE` / `config.PLACES`, augmented with cue-word scoring (e.g. "where", "located" → place; "who", "discovered" → person). On ties or no-match it falls back to a corpus-wide search. The router's classification + rationale are surfaced in the UI for transparency.

**Generation — grounded, cited, with an explicit "I don't know" path.** The system prompt forbids inventing facts and instructs the model to reply exactly `"I don't know."` when the context is insufficient. If retrieval returns zero chunks the LLM is bypassed entirely and we return `"I don't know."` directly.

For more detail see `Product_prd.md`. For thoughts on what would change in production see `recommendation.md`.

## Demo video

> _Add the Loom or unlisted-YouTube link here once recorded._

## Troubleshooting

- **"Ollama not reachable"** — make sure `ollama serve` is running and `OLLAMA_HOST` (default `http://localhost:11434`) is correct.
- **"Model llama3.2:3b is not pulled"** — run `ollama pull llama3.2:3b` (or whatever model you've selected in the sidebar).
- **"Vector store is empty"** — run `python -m scripts.ingest_data`. The first run takes ~1–2 minutes (mostly model download + Wikipedia fetches).
- **Slow first query** — the embedder is loaded lazily and the LLM warms up its KV cache on the first request. Subsequent queries are noticeably faster.
