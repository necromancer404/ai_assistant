from typing import List

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama import Client as OllamaClient

from .config import get_settings


def _prioritize_contexts(question: str, contexts: list[dict], top_k: int) -> list[dict]:
    q = question.lower()
    dept_tokens: list[str] = []
    if "computer science" in q or "cse" in q:
        dept_tokens.extend(["computer science", "cse"])
    if "scope" in q:
        dept_tokens.append("scope")
    role_tokens = ["hod", "head", "head of", "chair", "chairperson"]
    # naive name tokens from question
    import re as _re
    name_tokens = [t for t in _re.findall(r"[a-zA-Z][a-zA-Z'.-]+", q) if len(t) > 2]

    def score(c: dict) -> int:
        text = (c.get("text") or "").lower()
        s = 0
        ft = (c.get("metadata") or {}).get("file_type", "")
        fname = (c.get("metadata") or {}).get("filename", "")
        if ft == "json" or str(fname).endswith(".flatten.txt"):
            s += 5
        for t in role_tokens:
            if t in text:
                s += 3
        for t in dept_tokens:
            if t in text:
                s += 2
        for t in name_tokens:
            if t in text:
                s += 2
        try:
            dist = float(c.get("distance", 0.0))
        except Exception:
            dist = 0.0
        s += max(0, int(10 - 10 * dist))
        return s

    ranked = sorted(contexts, key=score, reverse=True)
    return ranked[:top_k]


def get_ollama_client():
    settings = get_settings()
    if settings.ollama_base_url:
        return OllamaClient(host=settings.ollama_base_url)
    return OllamaClient()


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


app = FastAPI(title="Local RAG (Ollama + Chroma)")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    top_k = int(req.top_k or settings.top_k)

    chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
    collection = chroma_client.get_collection("docs")

    oc = get_ollama_client()

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

    expanded_question = _expand_aliases(req.question)
    qvec = oc.embeddings(model=settings.embedding_model, prompt=expanded_question)["embedding"]

    # Query all types; we'll re-rank heuristically afterward
    fetch_k = max(top_k * 5, 10)
    res = collection.query(
        query_embeddings=[qvec],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    contexts: List[dict] = []
    for idx in range(len(res["documents"][0])):
        contexts.append(
            {
                "text": res["documents"][0][idx],
                "metadata": res["metadatas"][0][idx],
                "distance": res["distances"][0][idx],
            }
        )
    contexts = _prioritize_contexts(req.question, contexts, top_k)

    ctx = "\n\n".join(
        f"[Source: {c['metadata'].get('source','unknown')} chunk {c['metadata'].get('chunk_index','?')}]\\n{c['text']}"
        for c in contexts
    )
    instructions = ("Answer in a concise paragraph, using only facts from the context. "
                    "Use exact spellings from the source. Do not invent or merge fields. "
                    "If a fact is not in the context, say that is out of your scope."
                    )
    prompt = f"{instructions}\n\nContext:\n{ctx}\n\nQuestion: {req.question}\n\nAnswer:"

    answer = oc.generate(model=settings.generation_model, prompt=prompt, options={"temperature": 0.4})["response"]
    return QueryResponse(answer=answer.strip(), sources=[c["metadata"] for c in contexts])


