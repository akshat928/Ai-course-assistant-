"""Retrieval-augmented generation: retrieve chunks, build a grounded prompt, call the LLM."""

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from course_assistant.store import Retrieved, VectorStore

NO_ANSWER = "I couldn't find this in the course material."

SYSTEM_PROMPT = f"""You are a teaching assistant for a technical course.
Answer the student's question using ONLY the numbered context passages provided.

Rules:
- Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].
- If the passages don't contain the answer, reply exactly: "{NO_ANSWER}"
- Do not use outside knowledge, even if you know the answer.
- Be concise and clear. Use short paragraphs or bullet points."""


class LLM(Protocol):
    def __call__(self, system: str, messages: list[dict]) -> str: ...


@dataclass
class Answer:
    text: str
    sources: list[Retrieved] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)  # 1-based passage numbers the answer cites

    @property
    def grounded(self) -> bool:
        return bool(self.sources) and self.text.strip() != NO_ANSWER


def format_context(hits: list[Retrieved]) -> str:
    return "\n\n".join(f"[{i}] (source: {h.citation})\n{h.text}" for i, h in enumerate(hits, start=1))


def build_messages(question: str, hits: list[Retrieved], history: list[dict] | None = None) -> list[dict]:
    """Recent chat turns, then the context and the new question as the final user message."""
    messages = [m for m in (history or [])[-6:] if m["role"] in ("user", "assistant")]
    messages.append({
        "role": "user",
        "content": f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}",
    })
    return messages


def extract_citations(text: str, n_sources: int) -> list[int]:
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    return sorted(n for n in nums if 1 <= n <= n_sources)


def openai_llm(api_key: str, model: str, base_url: str | None = None, temperature: float = 0.1) -> LLM:
    """Client for any OpenAI-compatible chat completions API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def call(system: str, messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return resp.choices[0].message.content or ""

    return call


def extractive_llm(system: str, messages: list[dict]) -> str:
    """Fallback when no API key is set: returns the best passage verbatim, cited.

    Lets the app run end to end without an LLM. Answers are excerpts, not summaries.
    """
    content = messages[-1]["content"]
    first = re.search(r"\[1\] \(source: [^)]*\)\n(.*?)(?:\n\n\[2\]|\n\nQuestion:)", content, re.S)
    if not first:
        return NO_ANSWER
    quoted = "\n".join(f"> {line}" if line.strip() else ">" for line in first.group(1).strip().splitlines())
    return f"Most relevant passage from the course material:\n\n{quoted} [1]"


class CourseAssistant:
    def __init__(self, store: VectorStore, llm: LLM | Callable = extractive_llm,
                 top_k: int = 4, max_distance: float | None = 0.8):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_distance = max_distance

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        question = question.strip()
        if not question:
            return Answer(text="Please enter a question.")
        hits = self.store.search(question, k=self.top_k, max_distance=self.max_distance)
        if not hits:
            # Nothing relevant was retrieved: refuse instead of letting the LLM guess.
            return Answer(text=NO_ANSWER)
        text = self.llm(SYSTEM_PROMPT, build_messages(question, hits, history)).strip()
        return Answer(text=text, sources=hits, cited=extract_citations(text, len(hits)))
