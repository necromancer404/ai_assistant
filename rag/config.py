import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str | None
    generation_model: str
    embedding_model: str
    chroma_dir: str
    top_k: int
    chunk_size: int
    chunk_overlap: int


def get_settings() -> Settings:
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", None),
        generation_model=os.getenv("GENERATION_MODEL", "llama3.2:3b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        chroma_dir=os.getenv("CHROMA_DIR", "./chroma"),
        top_k=int(os.getenv("TOP_K", "3")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )


