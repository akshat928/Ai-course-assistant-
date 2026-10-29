"""Settings, read from environment variables (or a .env file) with sensible defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # Vector store
    persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", str(ROOT / "chroma_db")))
    collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "course_material"))

    # Embeddings: "sentence-transformers" (default) or "hashing" (offline, no model download)
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "sentence-transformers"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    # Chunking
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))

    # Retrieval
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    # Chunks with cosine distance above this are treated as irrelevant (0 = identical, 2 = opposite).
    # Unset means the embedder's default (see DEFAULT_MAX_DISTANCE in embeddings.py).
    max_distance: float | None = field(
        default_factory=lambda: float(os.environ["MAX_DISTANCE"]) if os.getenv("MAX_DISTANCE") else None)

    # LLM: any OpenAI-compatible API (OpenAI, Groq, Together, local Ollama, ...)
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    llm_base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
