"""Streamlit chat UI for the local Wikipedia RAG assistant.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time

import streamlit as st

from src import config, llm, vector_store
from src.rag import SYSTEM_PROMPT, answer_stream

st.set_page_config(page_title="Local Wikipedia RAG", page_icon=":books:", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: status + controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Wikipedia RAG")
    st.caption("Fully local: sentence-transformers + Chroma + Ollama")

    # Model selection
    model = st.text_input(
        "Ollama model",
        value=st.session_state.get("model", config.LLM_MODEL_NAME),
        help="Any model you've pulled with `ollama pull <name>`. Default: llama3.2:3b",
    )
    st.session_state["model"] = model

    top_k = st.slider("Chunks retrieved (top_k)", 1, 10, value=config.TOP_K)
    temperature = st.slider("Temperature", 0.0, 1.0, value=0.2, step=0.05)
    show_context = st.checkbox("Show retrieved context", value=True)

    st.divider()

    # Ollama status
    ok, msg = llm.is_available(model=model)
    if ok:
        st.success(f"Ollama: {msg}")
    else:
        st.error(f"Ollama: {msg}")

    # Index status
    try:
        stats = vector_store.collection_stats()
    except Exception as e:  # pragma: no cover - defensive
        stats = {"total": 0, "by_type": {}, "entities": []}
        st.error(f"Vector store error: {e}")

    if stats["total"] == 0:
        st.warning(
            "Vector store is empty. Run `python -m scripts.ingest_data` "
            "from a terminal first."
        )
    else:
        st.info(
            f"Indexed: **{stats['total']}** chunks "
            f"({stats['by_type'].get('person', 0)} person / "
            f"{stats['by_type'].get('place', 0)} place) "
            f"across {len(stats['entities'])} entities."
        )
        with st.expander("Indexed entities"):
            st.write(", ".join(stats["entities"]))

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {role, content, retrieval?}

st.title(":books: Local Wikipedia RAG Assistant")
st.caption(
    "Ask about any of the indexed people or places. Answers are grounded in "
    "Wikipedia content stored locally — the model will say *I don't know* if "
    "the answer isn't in the corpus."
)

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_context and msg.get("retrieval"):
            _render_retrieval = True  # always render for assistant messages
            with st.expander("Retrieved context & routing"):
                routing = msg["retrieval"]["routing"]
                st.markdown(f"**Routing:** `{routing['type']}` — {routing['rationale']}")
                for i, ch in enumerate(msg["retrieval"]["chunks"], start=1):
                    meta = ch["metadata"]
                    st.markdown(
                        f"**[Source {i}]** {meta.get('title', '?')} "
                        f"_(type={meta.get('type', '?')}, sim={ch['similarity']:.3f})_  \n"
                        f"[Wikipedia]({meta.get('url', '#')})"
                    )
                    st.text(ch["text"][:1200] + ("..." if len(ch["text"]) > 1200 else ""))


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask a question…")
if prompt:
    if stats["total"] == 0:
        st.error("The vector store is empty. Ingest data first.")
        st.stop()
    if not ok:
        st.error("Ollama is not reachable. Start it with `ollama serve` and pull the model.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        start = time.time()
        retrieval, token_iter = answer_stream(
            prompt, model=model, top_k=top_k, temperature=temperature
        )
        full = ""
        try:
            for token in token_iter:
                full += token
                placeholder.markdown(full + "▌")
        except Exception as e:
            full = f"_(generation failed: {e})_"
        elapsed = time.time() - start
        placeholder.markdown(full)
        st.caption(f"Generated in {elapsed:.1f}s · {len(retrieval.chunks)} chunks retrieved")

        retrieval_payload = {
            "routing": {
                "type": retrieval.routing.type,
                "rationale": retrieval.routing.rationale,
            },
            "chunks": retrieval.chunks,
        }

        if show_context:
            with st.expander("Retrieved context & routing"):
                st.markdown(
                    f"**Routing:** `{retrieval.routing.type}` — "
                    f"{retrieval.routing.rationale}"
                )
                for i, ch in enumerate(retrieval.chunks, start=1):
                    meta = ch["metadata"]
                    st.markdown(
                        f"**[Source {i}]** {meta.get('title', '?')} "
                        f"_(type={meta.get('type', '?')}, "
                        f"sim={ch['similarity']:.3f})_  \n"
                        f"[Wikipedia]({meta.get('url', '#')})"
                    )
                    st.text(ch["text"][:1200] + ("..." if len(ch["text"]) > 1200 else ""))

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": full,
            "retrieval": retrieval_payload,
        }
    )
