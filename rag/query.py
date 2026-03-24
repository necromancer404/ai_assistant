import argparse
from typing import List, Optional

import chromadb
from ollama import Client as OllamaClient

from .config import get_settings
import re


def get_ollama_client():
    settings = get_settings()
    if settings.ollama_base_url:
        return OllamaClient(host=settings.ollama_base_url)
    return OllamaClient()


def _prioritize_contexts(question: str, contexts: List[dict], top_k: int) -> List[dict]:
    # Heuristic boost for lines likely answering role/name queries
    q = question.lower()
    # Try to capture probable name tokens from the question
    name_tokens = [t for t in re.findall(r"[a-zA-Z][a-zA-Z'.-]+", q) if len(t) > 2]
    dept_tokens = []
    # crude tokenization for departments commonly referenced
    if "computer science" in q or "cse" in q:
        dept_tokens.extend(["computer science", "cse"])
    if "scope" in q:  # school of computing
        dept_tokens.append("scope")
    # role tokens
    role_tokens = ["hod", "head", "head of", "chair", "chairperson"]

    def score(c: dict) -> int:
        text = c["text"].lower()
        s = 0
        # Prefer JSON-derived chunks then flattened .txt
        ft = (c.get("metadata") or {}).get("file_type", "")
        fname = (c.get("metadata") or {}).get("filename", "")
        if ft == "json" or fname.endswith(".flatten.txt"):
            s += 5
        # Keyword presence boosts
        for t in role_tokens:
            if t in text:
                s += 3
        for t in dept_tokens:
            if t in text:
                s += 2
        # Prefer exact name token matches
        for t in name_tokens:
            if t.lower() in text:
                s += 2
        # Shorter distance is better
        try:
            dist = float(c.get("distance", 0.0))
        except Exception:
            dist = 0.0
        s += max(0, int(10 - 10 * dist))  # rough inverse distance boost
        return s

    ranked = sorted(contexts, key=score, reverse=True)
    return ranked[:top_k]


def retrieve(question: str, top_k: int) -> List[dict]:
    settings = get_settings()
    chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
    collection = chroma_client.get_collection("docs")

    # Expand common aliases so CSE/Computer Science resolve to SCOPE and vice versa
    def _expand_aliases(q: str) -> str:
        ql = q.lower()
        aliases: list[str] = []
        if "cse" in ql or "computer science" in ql:
            aliases += [
                "SCOPE",
                "School of Computer Science and Engineering",
                "Computer Science and Engineering",
                "Scope Department",
            ]
        if "scope" in ql or "school of computer science and engineering" in ql:
            aliases += [
                "CSE",
                "Computer Science",
                "Computer Science and Engineering",
            ]
        if not aliases:
            return q
        return q + " " + " ".join(aliases)

    # Query using Chroma's built-in embedding via our client
    oc = get_ollama_client()
    expanded_question = _expand_aliases(question)
    qvec = oc.embeddings(model=settings.embedding_model, prompt=expanded_question)["embedding"]

    # Query all types; we'll re-rank heuristically afterward
    fetch_k = max(top_k * 5, 10)  # fetch more, then re-rank heuristically
    res = collection.query(
        query_embeddings=[qvec],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    results: List[dict] = []
    for idx in range(len(res["documents"][0])):
        results.append(
            {
                "text": res["documents"][0][idx],
                "metadata": res["metadatas"][0][idx],
                "distance": res["distances"][0][idx],
            }
        )
    return _prioritize_contexts(question, results, top_k)


def build_prompt(question: str, contexts: List[dict]) -> str:
    ctx = "\n\n".join(
        f"[Source: {c['metadata'].get('source','unknown')} chunk {c['metadata'].get('chunk_index','?')}]\\n{c['text']}"
        for c in contexts
    )
    instructions = (
        "Answer in a concise paragraph, using only facts from the context. "
        "Use exact spellings from the source. Do not invent or merge fields. "
        "If a fact is not in the context, say that is out of your scope."
    )
    return f"{instructions}\n\nContext:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"


def generate_answer(prompt: str) -> str:
    settings = get_settings()
    oc = get_ollama_client()
    res = oc.generate(model=settings.generation_model, prompt=prompt, options={"temperature": 0.4})
    return res["response"]


def main(question: str):
    settings = get_settings()
    contexts = retrieve(question, top_k=settings.top_k)
    prompt = build_prompt(question, contexts)
    answer = generate_answer(prompt)

    print("\n=== Answer ===\n")
    print(answer.strip())
    print("\n=== Sources ===\n")
    for c in contexts:
        print(f"- {c['metadata'].get('source')} (chunk {c['metadata'].get('chunk_index')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query local RAG index with Ollama generation.")
    parser.add_argument("--question", "-q", type=str, required=True, help="Your question")
    args = parser.parse_args()
    main(args.question)


