"""Command-line interface.

    python -m course_assistant.cli index data/sample_docs
    python -m course_assistant.cli ask "What is the difference between TCP and UDP?"
    python -m course_assistant.cli chat
"""

import argparse
import sys

from course_assistant.config import Settings
from course_assistant.factory import build_assistant, build_store
from course_assistant.ingest import iter_folder, load_and_chunk


def cmd_index(settings: Settings, paths: list[str], reset: bool) -> None:
    store = build_store(settings)
    if reset:
        store.reset()
    files = []
    for p in paths:
        files.extend(iter_folder(p) if __import__("pathlib").Path(p).is_dir() else [p])
    total = 0
    for f in files:
        chunks = load_and_chunk(f, settings.chunk_size, settings.chunk_overlap)
        store.remove_source(chunks[0].source if chunks else str(f))
        total += store.add(chunks)
        print(f"  indexed {len(chunks):>4} chunks  {f}")
    print(f"Done: {total} chunks from {len(files)} files. Collection now holds {store.count()} chunks.")


def print_answer(ans) -> None:
    print(f"\n{ans.text}\n")
    for i, s in enumerate(ans.sources, start=1):
        mark = "*" if i in ans.cited else " "
        print(f"  {mark}[{i}] {s.citation}  (distance {s.distance:.2f})")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AI Course Assistant")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", help="Index files or folders")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--reset", action="store_true", help="Clear the collection first")
    pa = sub.add_parser("ask", help="Ask one question")
    pa.add_argument("question")
    sub.add_parser("chat", help="Interactive chat")
    args = p.parse_args(argv)

    settings = Settings()
    if args.cmd == "index":
        cmd_index(settings, args.paths, args.reset)
    elif args.cmd == "ask":
        print_answer(build_assistant(settings).ask(args.question))
    else:
        assistant, history = build_assistant(settings), []
        print("Ask a question (blank line to quit).")
        while (q := input("> ").strip()):
            ans = assistant.ask(q, history)
            print_answer(ans)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans.text}]
    return 0


if __name__ == "__main__":
    sys.exit(main())
