"""Streamlit interface for the AI Course Assistant.

    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from course_assistant.config import Settings
from course_assistant.factory import build_assistant, build_store
from course_assistant.ingest import iter_folder, load_and_chunk

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "sample_docs"

st.set_page_config(page_title="AI Course Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_store(embedder: str):
    settings = Settings()
    settings.embedder = embedder
    return settings, build_store(settings)


def index_file(store, settings, name: str, data: bytes | None = None, path: Path | None = None) -> int:
    chunks = load_and_chunk(path or name, settings.chunk_size, settings.chunk_overlap, data=data)
    store.remove_source(name)
    return store.add(chunks)


# --- Sidebar: settings and documents ------------------------------------------

with st.sidebar:
    st.header("📚 Course material")
    options = ["sentence-transformers", "hashing"]
    default = Settings().embedder
    embedder = st.selectbox("Embeddings", options, index=options.index(default) if default in options else 0,
                            help="'hashing' runs fully offline with no model download, but matches on "
                                 "shared words rather than meaning.")
    try:
        settings, store = get_store(embedder)
    except Exception as e:  # missing package, or the model can't be downloaded
        st.error(f"Couldn't load the '{embedder}' embeddings: {e}\n\n"
                 "Install requirements and check your internet connection, or switch to 'hashing'.")
        st.stop()

    uploads = st.file_uploader("Upload documents", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded files", type="primary", width="stretch"):
        with st.spinner("Chunking and embedding..."):
            n = sum(index_file(store, settings, f.name, data=f.getvalue()) for f in uploads)
        st.success(f"Indexed {n} chunks from {len(uploads)} file(s).")

    if st.button("Load sample course notes", width="stretch"):
        with st.spinner("Indexing sample notes..."):
            n = sum(index_file(store, settings, p.name, path=p) for p in iter_folder(SAMPLE_DIR))
        st.success(f"Indexed {n} chunks.")

    sources = store.sources()
    st.caption(f"{store.count()} chunks from {len(sources)} document(s)")
    for s in sources:
        st.markdown(f"- {s}")
    if sources and st.button("Clear all documents", width="stretch"):
        store.reset()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    top_k = st.slider("Passages to retrieve", 1, 8, settings.top_k)
    if settings.llm_api_key:
        st.caption(f"LLM: `{settings.llm_model}`")
    else:
        st.warning("No `LLM_API_KEY` set. Answers will be the best-matching passage instead of a "
                   "generated summary. See the README to connect an LLM.")

settings.top_k = top_k
assistant = build_assistant(settings, store)

# --- Chat ---------------------------------------------------------------------

st.title("AI Course Assistant")
st.caption("Ask questions about your course material. Answers are grounded in your documents, with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources, cited):
    if not sources:
        return
    with st.expander(f"Retrieved passages ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            badge = "✅ cited" if i in cited else "not cited"
            st.markdown(f"**[{i}] {s.citation}** · similarity {1 - s.distance:.2f} · {badge}")
            st.text(s.text)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []), msg.get("cited", []))

if question := st.chat_input("e.g. What is the difference between TCP and UDP?"):
    if store.count() == 0:
        st.info("Upload documents or load the sample notes from the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.chat_message("assistant"):
            with st.spinner("Searching the material..."):
                try:
                    answer = assistant.ask(question, history)
                except Exception as e:  # surface API errors instead of crashing the app
                    st.error(f"Error while answering: {e}")
                    st.stop()
            st.markdown(answer.text)
            render_sources(answer.sources, answer.cited)
        st.session_state.messages.append({"role": "assistant", "content": answer.text,
                                          "sources": answer.sources, "cited": answer.cited})
