"""Build a ready-to-use assistant from Settings."""

from course_assistant.config import Settings
from course_assistant.embeddings import DEFAULT_MAX_DISTANCE, get_embedder
from course_assistant.rag import CourseAssistant, extractive_llm, openai_llm
from course_assistant.store import VectorStore


def build_store(settings: Settings) -> VectorStore:
    embedder = get_embedder(settings.embedder, settings.embedding_model)
    # Separate collections per embedder: vectors from different models aren't comparable.
    return VectorStore(embedder, settings.persist_dir, f"{settings.collection}_{settings.embedder.replace('-', '_')}")


def build_assistant(settings: Settings, store: VectorStore | None = None) -> CourseAssistant:
    store = store or build_store(settings)
    if settings.llm_api_key:
        llm = openai_llm(settings.llm_api_key, settings.llm_model, settings.llm_base_url, settings.temperature)
    else:
        llm = extractive_llm
    max_distance = settings.max_distance
    if max_distance is None:
        max_distance = DEFAULT_MAX_DISTANCE.get(settings.embedder, 0.8)
    return CourseAssistant(store, llm, top_k=settings.top_k, max_distance=max_distance)
