#!/usr/bin/env python3
"""
Local RAG over a directory of markdown, using Ollama for both embedding and
generation and ChromaDB for storage. Nothing leaves the machine.

    python rag.py index --corpus corpus/
    python rag.py query "how does chunking handle code blocks?"
    python rag.py query                       # interactive
    python rag.py query --model llama3.1:8b "..."

Model resolution, highest precedence first:
    --model  >  $RAG_CHAT_MODEL  >  DEFAULT_CHAT_MODEL

The embedding model follows the same order minus the flag. Changing it
invalidates the existing index, because vectors from two different embedding
models are not comparable and there is no migration short of re-embedding.
Re-run `index` after any embedding-model change.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap

import chromadb
import requests
import yaml

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
DEFAULT_CHROMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma")

CORPUS_DIR = os.path.abspath(os.environ.get("RAG_CORPUS") or DEFAULT_CORPUS)
CHROMA_DIR = os.path.abspath(os.environ.get("RAG_CHROMA_DIR") or DEFAULT_CHROMA)
OLLAMA_BASE = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
COLLECTION_NAME = "corpus"
TOP_K = 5

DEFAULT_EMBED_MODEL = "nomic-embed-text"

# On the reference machine (AMD Strix Halo, Ollama's Vulkan backend) a larger
# MoE model produced better answers when it produced them at all, and returned
# empty content often enough to be unusable: 2/8 on the eval suite as it stood
# on 2026-08-13, where llama3.1:8b returned 8/8 on identical prompts. The suite
# has since grown to 12 questions, so those two figures are historical and are
# NOT comparable to a current run. Re-measure before treating them as live:
#
#     python eval.py --model <candidate> --out-dir results/
#
# The general point survives the stale numbers. Reliability is a property of the
# model and backend together, not of the model alone, and the better model on
# answer quality is not automatically the better model to depend on.
DEFAULT_CHAT_MODEL = "llama3.1:8b"

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL") or DEFAULT_EMBED_MODEL
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL") or DEFAULT_CHAT_MODEL

# Some models advertise a very large context window. Ollama allocates the KV
# cache up front, so the advertised ceiling dominates cold-load time and resident
# memory. Retrieval here sends roughly 2K tokens, so cap well below the ceiling
# and leave room to raise TOP_K later.
NUM_CTX = 16384
# Keep the model resident between queries. The default idle unload means a full
# reload on the next question, which swamps the actual generation time.
KEEP_ALIVE = "30m"
# Hard ceiling on generated tokens, reasoning included. A reasoning model will
# occasionally loop on a retrieved context and never emit an answer; without a
# cap the request simply never returns. A capped run fails in seconds and says
# why, which is recoverable. Real answers here run a few hundred tokens.
NUM_PREDICT = 2048

SKIP_FILES = {"log.md", "index.md", "README.md"}


def resolve_chat_model(cli_model: str | None = None) -> str:
    return cli_model or CHAT_MODEL


# --- corpus loading and chunking ---

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from the markdown body."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            return meta, parts[2].strip()
    return {}, text.strip()


def load_pages(corpus_dir: str | None = None) -> list[dict]:
    """Load every markdown page under the corpus directory."""
    corpus_dir = corpus_dir or CORPUS_DIR
    pages = []
    for root, dirs, files in os.walk(corpus_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if not fname.endswith(".md") or fname in SKIP_FILES or fname.startswith("._"):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, corpus_dir).replace("\\", "/")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                meta, body = parse_frontmatter(f.read())
            if not body:
                continue
            pages.append({
                "id": rel,
                "title": meta.get("title", fname[:-3].replace("-", " ").title()),
                "type": meta.get("type", "unknown"),
                "tags": meta.get("tags", []) or [],
                "body": body,
                "path": rel,
            })
    return pages


def balance_fences(text: str) -> str:
    """Close a trailing open ``` fence so no chunk ends inside a code block.

    Last line of defence. The splitters below avoid cutting fences in the first
    place; this also catches a source page whose own fences are unbalanced.
    """
    if text.count("```") % 2 == 1:
        return text.rstrip() + "\n```"
    return text


def split_units(text: str) -> list[str]:
    """Split into paragraph units, treating a fenced block as indivisible.

    A blank line ends a unit only when it is outside a ``` fence, so a code
    block stays whole, blank lines and all, and no downstream packing step can
    cut through it. Half a code block retrieves as noise and reads as garbage.
    """
    units, current, in_fence = [], [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        if not line.strip() and not in_fence:
            if current:
                units.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        units.append("\n".join(current))
    return units


def split_oversized_unit(unit: str, max_chars: int) -> list[str]:
    """Split one over-length unit on line boundaries, keeping fences balanced.

    Two cases arrive here: a fenced block that alone exceeds max_chars, and a
    paragraph with no internal blank line that does the same, which in practice
    means a wide markdown table. When a cut lands inside a fence the piece is
    closed with ``` and the next piece reopens with the original fence line, so
    every piece is individually well-formed.

    A single line longer than max_chars is never broken, so a piece can still
    overshoot. That is deliberate. An oversized chunk costs context; a table row
    sliced down the middle costs meaning.
    """
    pieces, buffer, size = [], [], 0
    in_fence, fence_open = False, "```"

    for line in unit.split("\n"):
        if buffer and size + len(line) + 1 > max_chars:
            piece = "\n".join(buffer)
            if in_fence:
                piece += "\n```"
            pieces.append(piece)
            buffer = [fence_open] if in_fence else []
            size = sum(len(x) + 1 for x in buffer)
        if line.lstrip().startswith("```"):
            if not in_fence:
                fence_open = line
            in_fence = not in_fence
        buffer.append(line)
        size += len(line) + 1

    if buffer:
        piece = "\n".join(buffer)
        if in_fence:
            piece += "\n```"
        pieces.append(piece)
    return [p for p in pieces if p.strip()]


def make_chunk(page: dict, chunk_id: str, chunk_index: int, text: str) -> dict:
    """Wrap chunk text with its page header so a retrieved fragment carries its source."""
    tags = page["tags"]
    return {
        "id": chunk_id,
        "text": f"# {page['title']}\nType: {page['type']}\n\n{balance_fences(text.strip())}",
        "metadata": {
            "title": page["title"],
            "type": page["type"],
            "path": page["path"],
            "tags": ", ".join(tags) if tags else "",
            "chunk": chunk_index,
        },
    }


def chunk_page(page: dict, max_chars: int = 1500) -> list[dict]:
    """Split a page into embeddable chunks.

    Short pages stay whole. Longer pages split on markdown headers, and a
    section still over budget is packed from paragraph units where a fenced
    block counts as one indivisible unit.
    """
    body = page["body"]
    if len(body) <= max_chars:
        return [make_chunk(page, page["id"], 0, body)]

    sections = re.split(r"\n(?=##+ )", body)
    chunks: list[dict] = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(make_chunk(page, f"{page['id']}::chunk{i}", i, section))
            continue

        buffer, sub = "", 0
        for unit in split_units(section):
            # A unit that cannot fit on its own gets its own chunk. Flushing only
            # on a non-empty buffer would let an oversized unit arriving first
            # become a single unbounded chunk.
            if len(unit) > max_chars:
                if buffer.strip():
                    chunks.append(make_chunk(page, f"{page['id']}::chunk{i}.{sub}", i, buffer))
                    sub += 1
                    buffer = ""
                for piece in split_oversized_unit(unit, max_chars):
                    chunks.append(make_chunk(page, f"{page['id']}::chunk{i}.{sub}", i, piece))
                    sub += 1
                continue
            if buffer and len(buffer) + len(unit) > max_chars:
                chunks.append(make_chunk(page, f"{page['id']}::chunk{i}.{sub}", i, buffer))
                sub += 1
                buffer = ""
            buffer += unit + "\n\n"
        if buffer.strip():
            chunks.append(make_chunk(page, f"{page['id']}::chunk{i}.{sub}", i, buffer))
    return chunks


# --- embedding and index ---

def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed", json={"model": EMBED_MODEL, "input": text}, timeout=120
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=300
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def build_index(corpus_dir: str | None = None, chroma_dir: str | None = None) -> int:
    corpus_dir = os.path.abspath(corpus_dir or CORPUS_DIR)
    chroma_dir = os.path.abspath(chroma_dir or CHROMA_DIR)

    print(f"corpus:    {corpus_dir}")
    print(f"chroma:    {chroma_dir}")
    print(f"embedding: {EMBED_MODEL}")

    pages = load_pages(corpus_dir)
    if not pages:
        print(f"\nNo markdown pages found under {corpus_dir}.", file=sys.stderr)
        return 1
    print(f"\nLoaded {len(pages)} pages")

    all_chunks: list[dict] = []
    for page in pages:
        all_chunks.extend(chunk_page(page))
    print(f"Split into {len(all_chunks)} chunks")

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # hnsw:space selects the distance function. Cosine ignores magnitude and
    # compares direction, which is what these embeddings are trained for.
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    batch_size = 32
    total_batches = (len(all_chunks) - 1) // batch_size + 1
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i: i + batch_size]
        print(f"  embedding batch {i // batch_size + 1}/{total_batches}...")
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=get_embeddings_batch([c["text"] for c in batch]),
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    print(f"\nIndex built: {collection.count()} chunks in {chroma_dir}")
    return 0


# --- query ---

SYSTEM_PROMPT = textwrap.dedent("""\
    You are answering questions from a provided set of reference documents.
    Use ONLY the supplied context. If the context does not contain enough
    information to answer, say so plainly rather than guessing. Cite document
    titles when you reference specific information. Be direct and concise.""")


def query(
    question: str,
    model: str | None = None,
    quiet: bool = False,
    chroma_dir: str | None = None,
) -> dict:
    """Retrieve relevant chunks and generate an answer.

    Returns a dict (model, question, answer, done_reason, pages) so a harness can
    measure a run without scraping stdout. `quiet=True` suppresses streaming output.
    """
    model = resolve_chat_model(model)
    client = chromadb.PersistentClient(path=os.path.abspath(chroma_dir or CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[get_embedding(question)],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    documents, metadatas = results["documents"][0], results["metadatas"][0]
    distances = results["distances"][0]

    # Name the model before generating anything, so a saved transcript is never
    # ambiguous about what was under test.
    if not quiet:
        print(f"\nModel: {model} (embed: {EMBED_MODEL})")
        print("\n--- Retrieved ---")

    context_parts, pages, seen = [], [], set()
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1 - dist
        if meta["title"] not in seen:
            if not quiet:
                print(f"  [{similarity:.3f}] {meta['title']} ({meta['path']})")
            seen.add(meta["title"])
            pages.append({
                "title": meta["title"],
                "path": meta["path"],
                "similarity": round(similarity, 4),
            })
        context_parts.append(doc)

    user_prompt = (
        "## Context\n\n" + "\n\n---\n\n".join(context_parts) + f"\n\n## Question\n\n{question}"
    )

    if not quiet:
        print("\n--- Answer ---\n")

    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            # Do NOT add "think": False here. On some reasoning models that does
            # not disable reasoning; it only stops Ollama from splitting the
            # reasoning into a separate "thinking" field. The chain of thought
            # then lands in "content" with no stop discipline and generation runs
            # away until num_predict cuts it off, producing no answer at all.
            "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        },
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()

    thinking_chunks, answer_parts, done_reason = 0, [], None
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        message = data.get("message", {})

        # Reasoning streams into "thinking" and is not part of the answer. Show
        # progress instead of the reasoning text.
        if message.get("thinking"):
            if thinking_chunks == 0 and not quiet:
                print("thinking ", end="", flush=True)
            thinking_chunks += 1
            if thinking_chunks % 20 == 0 and not quiet:
                print(".", end="", flush=True)

        content = message.get("content", "")
        if content:
            if thinking_chunks and not quiet:
                print("\n")
            thinking_chunks = 0
            answer_parts.append(content)
            if not quiet:
                print(content, end="", flush=True)

        if data.get("done"):
            done_reason = data.get("done_reason")
            break

    if not quiet:
        print("\n")

    answer = "".join(answer_parts)
    if not answer and not quiet:
        print(
            f"[no answer: the model spent its entire {NUM_PREDICT}-token budget without "
            f"producing one (done_reason={done_reason}). Retry, or narrow the question.]\n",
            file=sys.stderr,
        )

    return {
        "model": model,
        "question": question,
        "answer": answer,
        "done_reason": done_reason,
        "pages": pages,
    }


def interactive(model: str | None = None, chroma_dir: str | None = None) -> int:
    model = resolve_chat_model(model)
    print(f"Local RAG, interactive. Model {model}. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        query(question, model=model, chroma_dir=chroma_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local RAG over a markdown corpus.")
    sub = parser.add_subparsers(dest="command")

    idx = sub.add_parser("index", help="Build or rebuild the vector index")
    idx.add_argument("--corpus", default=None, help=f"Markdown corpus directory (default: {CORPUS_DIR})")
    idx.add_argument("--chroma-dir", default=None, help=f"ChromaDB directory (default: {CHROMA_DIR})")

    q = sub.add_parser("query", help="Ask a question")
    q.add_argument("question", nargs="?", help="Question. Omit for interactive mode.")
    q.add_argument("--model", default=None,
                   help=f"Ollama chat model (default: $RAG_CHAT_MODEL, else {DEFAULT_CHAT_MODEL})")
    q.add_argument("--chroma-dir", default=None, help=f"ChromaDB directory (default: {CHROMA_DIR})")

    args = parser.parse_args()

    if args.command == "index":
        return build_index(args.corpus, args.chroma_dir)
    if args.command == "query":
        if args.question:
            query(args.question, model=args.model, chroma_dir=args.chroma_dir)
            return 0
        return interactive(model=args.model, chroma_dir=args.chroma_dir)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
