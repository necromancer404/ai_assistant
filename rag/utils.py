import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Tuple

from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".txt", ".pdf", ".json"}


def iter_files(root_dir: str | os.PathLike) -> Iterator[Path]:
    base = Path(root_dir)
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    texts: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            texts.append(txt)
    return "\n".join(texts)


def json_to_text(obj) -> str:
    # Convert arbitrary JSON to a readable text blob
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def load_json_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            data = json.load(f)
        except Exception:
            # Fallback: return raw text if not valid JSON
            f.seek(0)
            return f.read()
    return json_to_text(data)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_text_file(path)
    if suffix == ".pdf":
        return load_pdf_file(path)
    if suffix == ".json":
        return load_json_file(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        return [text]
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def as_documents(chunks: Iterable[Tuple[str, str, int]]) -> list[dict]:
    # Convert (doc_id, chunk_text, chunk_index) to Chroma documents with metadata
    docs: list[dict] = []
    for doc_id, chunk_text, chunk_idx in chunks:
        docs.append(
            {
                "id": f"{doc_id}::chunk-{chunk_idx}",
                "text": chunk_text,
                "metadata": {"source": doc_id, "chunk_index": chunk_idx},
            }
        )
    return docs


