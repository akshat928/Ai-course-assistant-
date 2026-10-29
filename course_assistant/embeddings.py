"""Embedding functions for ChromaDB.

- SentenceTransformerEmbedder: semantic embeddings (default, downloads the model once)
- HashingEmbedder: dependency-free bag-of-words vectors for offline use and tests
"""

import hashlib
import math
import re

from chromadb import Documents, EmbeddingFunction, Embeddings


class SentenceTransformerEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input), normalize_embeddings=True).tolist()

    @staticmethod
    def name() -> str:
        return "course-assistant-sentence-transformers"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "SentenceTransformerEmbedder":
        return SentenceTransformerEmbedder(config["model_name"])


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was "
    "what when where which who why will with does do can".split()
)


class HashingEmbedder(EmbeddingFunction):
    """Hashes words and word pairs into a fixed-size vector, then L2-normalizes it.

    It matches on shared vocabulary rather than meaning, so results are weaker than
    sentence-transformers, but it needs no downloads and is fully deterministic.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        vec = [0.0] * self.dim
        for f in features:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(t) for t in input]

    @staticmethod
    def name() -> str:
        return "course-assistant-hashing"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbedder":
        return HashingEmbedder(config.get("dim", 4096))


# Default cosine-distance cut-off per embedder: retrieved chunks further than this are
# treated as irrelevant. Hashing vectors only overlap on shared words, so even good
# matches sit at higher distances than with semantic embeddings.
DEFAULT_MAX_DISTANCE = {"sentence-transformers": 0.8, "hashing": 0.94}


def get_embedder(kind: str, model_name: str = "all-MiniLM-L6-v2") -> EmbeddingFunction:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedder '{kind}'. Use 'sentence-transformers' or 'hashing'.")
