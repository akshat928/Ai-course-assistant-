"""Tests run fully offline: the hashing embedder replaces sentence-transformers and
a fake LLM replaces the API call."""

from pathlib import Path

import pytest

from course_assistant.embeddings import HashingEmbedder
from course_assistant.ingest import chunk_document, iter_folder, load_and_chunk, load_pages
from course_assistant.rag import (NO_ANSWER, SYSTEM_PROMPT, CourseAssistant, build_messages,
                                  extract_citations, extractive_llm, format_context)
from course_assistant.store import Retrieved, VectorStore

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = VectorStore(HashingEmbedder(), str(tmp_path_factory.mktemp("chroma")), "test")
    for path in iter_folder(SAMPLES):
        s.add(load_and_chunk(path, chunk_size=600, chunk_overlap=80))
    return s


# --- Ingestion ----------------------------------------------------------------

def test_chunks_respect_size_and_keep_metadata():
    pages = [(None, "Intro paragraph about networks.\n\n" + ("TCP is reliable. " * 200))]
    chunks = chunk_document("notes.md", pages, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.source == "notes.md" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_ids_are_stable():
    pages = [(1, "Same text on page one. " * 50)]
    a = chunk_document("x.pdf", pages, 200, 20)
    b = chunk_document("x.pdf", pages, 200, 20)
    assert [c.id for c in a] == [c.id for c in b]
    assert a[0].metadata["page"] == 1


def test_load_pages_from_bytes_and_rejects_unknown_types():
    assert load_pages("notes.txt", data=b"hello world")[0] == (None, "hello world")
    with pytest.raises(ValueError):
        load_pages("slides.pptx", data=b"")


def test_pdf_loading(tmp_path):
    from pypdf import PdfWriter

    path = tmp_path / "blank.pdf"
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        w.write(f)
    pages = load_pages(path)
    assert [p for p, _ in pages] == [1, 2]


# --- Retrieval ----------------------------------------------------------------

@pytest.mark.parametrize("question, expected_source", [
    ("Explain the TCP three-way handshake SYN ACK", "01_computer_networks.md"),
    ("How should passwords be stored with salt and bcrypt?", "02_cryptography.md"),
    ("What is the difference between precision and recall?", "03_machine_learning.md"),
    ("How many usable hosts are in a /26 subnet?", "01_computer_networks.md"),
])
def test_retrieval_finds_the_right_document(store, question, expected_source):
    hits = store.search(question, k=3)
    assert hits[0].source == expected_source


def test_upsert_does_not_duplicate(tmp_path):
    s = VectorStore(HashingEmbedder(), str(tmp_path), "dup")
    chunks = load_and_chunk(SAMPLES / "02_cryptography.md")
    s.add(chunks)
    s.add(chunks)
    assert s.count() == len(chunks)
    s.remove_source("02_cryptography.md")
    assert s.count() == 0


def test_max_distance_filters_irrelevant_chunks(store):
    assert store.search("recipe for chocolate cake with frosting", k=4, max_distance=0.94) == []


# --- Prompting and answers ----------------------------------------------------

HITS = [Retrieved("TCP uses a three-way handshake.", "net.md", None, 0.2),
        Retrieved("UDP has an 8 byte header.", "net.pdf", 3, 0.3)]


def test_context_is_numbered_with_citations():
    ctx = format_context(HITS)
    assert "[1] (source: net.md)" in ctx
    assert "[2] (source: net.pdf, p. 3)" in ctx


def test_build_messages_keeps_recent_history_only():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    msgs = build_messages("new question", HITS, history)
    assert len(msgs) == 7
    assert msgs[-1]["content"].endswith("Question: new question")


def test_extract_citations_ignores_out_of_range():
    assert extract_citations("Fact [1]. Other [2][2]. Bogus [9].", 2) == [1, 2]


def test_ask_passes_context_and_tracks_citations(store):
    seen = {}

    def fake_llm(system, messages):
        seen["system"], seen["messages"] = system, messages
        return "TCP needs a handshake while UDP does not [1][2]."

    assistant = CourseAssistant(store, fake_llm, top_k=3, max_distance=0.94)
    ans = assistant.ask("What is the difference between TCP and UDP?")
    assert seen["system"] == SYSTEM_PROMPT
    assert "Context passages" in seen["messages"][-1]["content"]
    assert ans.cited == [1, 2]
    assert ans.grounded
    assert len(ans.sources) == 3


def test_ask_refuses_without_calling_llm_when_nothing_relevant(store):
    def exploding_llm(system, messages):
        raise AssertionError("LLM should not be called")

    ans = CourseAssistant(store, exploding_llm, max_distance=0.94).ask("best pizza toppings in Naples")
    assert ans.text == NO_ANSWER
    assert not ans.grounded


def test_extractive_fallback_returns_top_passage(store):
    ans = CourseAssistant(store, extractive_llm, max_distance=0.94).ask("What does a DNS MX record do?")
    assert "[1]" in ans.text
    assert ans.cited == [1]
    assert ans.sources[0].text.split()[0] in ans.text


def test_empty_question():
    ans = CourseAssistant(VectorStore(HashingEmbedder(), None, "empty"), extractive_llm).ask("   ")
    assert ans.text == "Please enter a question."


def test_openai_llm_sends_system_prompt(monkeypatch):
    import openai

    from course_assistant.rag import openai_llm

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = type("M", (), {"content": "Answer [1]"})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    class FakeClient:
        def __init__(self, api_key, base_url=None):
            captured["api_key"], captured["base_url"] = api_key, base_url
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    llm = openai_llm("sk-test", "test-model", base_url="http://localhost:11434/v1")
    out = llm("SYS", [{"role": "user", "content": "hi"}])
    assert out == "Answer [1]"
    assert captured["model"] == "test-model"
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["base_url"] == "http://localhost:11434/v1"


def test_factory_uses_embedder_default_threshold(tmp_path, monkeypatch):
    from course_assistant.config import Settings
    from course_assistant.factory import build_assistant

    monkeypatch.delenv("MAX_DISTANCE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings()
    s.embedder, s.persist_dir = "hashing", str(tmp_path)
    a = build_assistant(s)
    assert a.max_distance == 0.94
    assert a.llm is extractive_llm
