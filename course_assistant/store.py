"""ChromaDB vector store wrapper."""

from dataclasses import dataclass

import chromadb
from chromadb import EmbeddingFunction

from course_assistant.ingest import Chunk


@dataclass
class Retrieved:
    text: str
    source: str
    page: int | None
    distance: float

    @property
    def citation(self) -> str:
        return f"{self.source}, p. {self.page}" if self.page else self.source


class VectorStore:
    def __init__(self, embedder: EmbeddingFunction, persist_dir: str | None = None,
                 collection: str = "course_material"):
        self.client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], batch_size: int = 256) -> int:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )
        return len(chunks)

    def remove_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def search(self, query: str, k: int = 4, max_distance: float | None = None) -> list[Retrieved]:
        if self.count() == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, self.count()))
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if max_distance is not None and dist > max_distance:
                continue
            hits.append(Retrieved(text=text, source=meta["source"], page=meta.get("page"), distance=float(dist)))
        return hits

    def sources(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        name, ef = self.collection.name, self.collection._embedding_function
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, embedding_function=ef, metadata={"hnsw:space": "cosine"})
