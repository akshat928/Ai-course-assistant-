# AI Course Assistant

A retrieval-augmented generation (RAG) assistant that answers natural-language questions about your course material, grounded in the documents you upload, with a citation for every claim.

Upload lecture notes, PDFs or textbook chapters, ask a question, and the assistant retrieves the most relevant passages from a ChromaDB vector store, sends them to an LLM with strict grounding instructions, and shows you exactly which passages the answer came from.

## How it works

```
 Documents (PDF / MD / TXT)
        │
        ▼
 Chunking ── LangChain RecursiveCharacterTextSplitter (headings → paragraphs → sentences)
        │
        ▼
 Embedding ── Sentence-Transformers (all-MiniLM-L6-v2)
        │
        ▼
 ChromaDB ── persistent vector store, cosine similarity
        │
 Question ──► embed ──► top-k nearest chunks ──► relevance filter
                                                       │
                              nothing relevant? ◄──────┤
                              "I couldn't find this"   │
                                                       ▼
                               LLM with numbered passages + grounding rules
                                                       │
                                                       ▼
                               Answer with [1][2] citations + retrieved chunks
```

### Reducing hallucination

- **Grounding prompt**: the model is told to answer only from the numbered passages, cite each claim, and reply with a fixed refusal when the answer isn't there.
- **Relevance filter**: chunks above a cosine-distance cut-off are dropped. If nothing relevant is left, the assistant refuses **without calling the LLM**, so it can't make something up.
- **Citation tracking**: the app parses `[n]` markers and labels each retrieved passage as cited or not, so you can check the answer against its sources.

## Features

- Upload PDF, Markdown and text files from the browser, or index a folder from the command line
- Chat with follow-up questions (recent turns are sent as history)
- Inspect the retrieved chunks behind every answer, with similarity scores
- Works with any OpenAI-compatible API: OpenAI, Groq, Together, or a local model through Ollama
- Runs without an API key too: it returns the best-matching passage instead of a generated answer
- Offline `hashing` embedder for machines that can't download models
- Re-indexing a file replaces its chunks instead of duplicating them

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add your API key
streamlit run app.py
```

In the sidebar, click **Load sample course notes** (computer networks, cryptography, machine learning) or upload your own files, then ask a question such as *"Why should ECB mode not be used?"*

The first run downloads the embedding model (about 90 MB).

## Connecting an LLM

Set these in `.env`:

| Provider | `LLM_BASE_URL` | `LLM_MODEL` example |
|---|---|---|
| OpenAI | *(leave empty)* | `gpt-4o-mini` |
| Groq (free tier) | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| Ollama (local, no key needed) | `http://localhost:11434/v1` | `llama3.1` |

For Ollama, set `LLM_API_KEY=ollama` (any non-empty value).

## Command line

```bash
python -m course_assistant.cli index data/sample_docs --reset
python -m course_assistant.cli ask "Why should ECB mode not be used?"
python -m course_assistant.cli chat
```

Example output without an LLM key (extractive mode):

```
Most relevant passage from the course material:

> ## Symmetric Encryption
> ...
> A block cipher needs a mode of operation to encrypt data longer than one block. ECB mode
> encrypts each block independently, so identical plaintext blocks produce identical
> ciphertext blocks and leak patterns; it should never be used. ... [1]

  *[1] 02_cryptography.md  (distance 0.83)
   [2] 02_cryptography.md  (distance 0.89)
   [3] 01_computer_networks.md  (distance 0.94)
```

The `*` marks passages the answer cites.

## Configuration

All settings come from environment variables or `.env`. See [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDER` | `sentence-transformers` | Or `hashing` for offline use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any Sentence-Transformers model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Characters per chunk and overlap |
| `TOP_K` | `4` | Passages retrieved per question |
| `MAX_DISTANCE` | per embedder | Cosine-distance cut-off for relevance |
| `CHROMA_DIR` | `./chroma_db` | Where the vector store is saved |

## Project layout

```
app.py                          Streamlit interface
course_assistant/ingest.py      Loading PDFs/text and chunking
course_assistant/embeddings.py  Sentence-Transformers and offline hashing embedders
course_assistant/store.py       ChromaDB wrapper: upsert, search, delete by source
course_assistant/rag.py         Prompt building, LLM call, citation parsing, refusal logic
course_assistant/factory.py     Builds the assistant from settings
course_assistant/cli.py         Command-line interface
data/sample_docs/               Sample course notes
tests/                          pytest suite (runs offline)
```

## Tests

```bash
pytest -q
```

The 19 tests run fully offline. They use the hashing embedder and a fake LLM, and they cover chunking, PDF loading, retrieval accuracy on the sample notes, de-duplication, the relevance filter, citation parsing, the refusal path, and the OpenAI client wrapper.

## Future work

- Hybrid search (BM25 keyword + vector) and a cross-encoder re-ranker
- Streaming responses in the chat
- An evaluation set measuring answer faithfulness and retrieval recall

## License

MIT. See [LICENSE](LICENSE).
