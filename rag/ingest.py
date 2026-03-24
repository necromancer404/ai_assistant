import argparse
import os
from pathlib import Path
from typing import List

import chromadb
import numpy as np
from ollama import Client as OllamaClient
from tqdm import tqdm

from .config import get_settings
from .utils import iter_files, load_document, chunk_text


def get_ollama_client():
    settings = get_settings()
    if settings.ollama_base_url:
        return OllamaClient(host=settings.ollama_base_url)
    return OllamaClient()


def embed_texts(oc: OllamaClient, texts: List[str]) -> List[List[float]]:
    settings = get_settings()
    vectors: List[List[float]] = []
    for t in tqdm(texts, desc="Embedding", unit="chunk"):
        resp = oc.embeddings(model=settings.embedding_model, prompt=t)
        vectors.append(resp["embedding"])
    return vectors


def ensure_collection(client: chromadb.ClientAPI, name: str):
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(name=name)


def main(data_dir: str):
    settings = get_settings()
    oc = get_ollama_client()

    # Prepare Chroma persistent client
    chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
    collection = ensure_collection(chroma_client, name="docs")

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metadatas: list[dict] = []

    paths = list(iter_files(data_dir))
    if not paths:
        print("No supported files found. Supported: .txt, .pdf, .json")
        return

    for path in tqdm(paths, desc="Reading files", unit="file"):
        doc_id = str(Path(path).resolve())
        text = load_document(path)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}::chunk-{i}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metadatas.append({
                "source": doc_id,
                "filename": Path(path).name,
                "file_type": Path(path).suffix.lower().lstrip("."),
                "chunk_index": i
            })

    vectors = embed_texts(oc, all_texts)

    
    batch = int(os.getenv("CHROMA_MAX_BATCH", "128"))
    for i in tqdm(range(0, len(all_ids), batch), desc="Saving to Chroma", unit="batch"):
        j = i + batch
        collection.add(
            ids=all_ids[i:j],
            embeddings=vectors[i:j],
            documents=all_texts[i:j],
            metadatas=all_metadatas[i:j],
        )

    print(f"Ingested {len(all_ids)} chunks from {len(paths)} files into {settings.chroma_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest files into local Chroma via Ollama embeddings.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing .txt .pdf .json files")
    args = parser.parse_args()
    main(args.data_dir)


