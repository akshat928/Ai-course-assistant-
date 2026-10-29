"""Load documents and split them into overlapping chunks for embedding."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".md", ".txt"}


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int

    @property
    def metadata(self) -> dict:
        meta = {"source": self.source, "chunk_index": self.chunk_index}
        if self.page is not None:
            meta["page"] = self.page
        return meta


def load_pages(path: str | Path, data: bytes | None = None) -> list[tuple[int | None, str]]:
    """Return [(page_number, text), ...]. Text files are a single page with number None.

    Pass `data` to read from bytes (e.g. a Streamlit upload) instead of the file system.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type {suffix}. Use one of: {', '.join(sorted(SUPPORTED))}")

    if suffix == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data) if data is not None else str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    text = data.decode("utf-8", errors="replace") if data is not None else path.read_text(encoding="utf-8", errors="replace")
    return [(None, text)]


def chunk_document(source: str, pages: list[tuple[int | None, str]],
                   chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    """Split pages into chunks. Splits on headings and paragraphs first, then sentences."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < 20:  # skip fragments like stray headers or page numbers
                continue
            idx = len(chunks)
            # Stable ID: re-indexing the same file updates chunks instead of duplicating them.
            digest = hashlib.sha1(f"{source}|{page}|{idx}|{piece}".encode()).hexdigest()[:16]
            chunks.append(Chunk(id=digest, text=piece, source=source, page=page, chunk_index=idx))
    return chunks


def load_and_chunk(path: str | Path, chunk_size: int = 800, chunk_overlap: int = 120,
                   data: bytes | None = None) -> list[Chunk]:
    path = Path(path)
    return chunk_document(path.name, load_pages(path, data), chunk_size, chunk_overlap)


def iter_folder(folder: str | Path):
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p
