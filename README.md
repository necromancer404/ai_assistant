## RAG with Ollama (TXT, PDF, JSON) - Windows friendly

This project provides a simple local RAG pipeline powered by Ollama:
- Ingests `.txt`, `.pdf`, and `.json` files from a directory
- Stores embeddings locally using Chroma (SQLite-backed)
- Answers questions via retrieval + generation
- Includes a CLI and an optional FastAPI server

Works offline on a 16 GB RAM laptop with small models.

### Prerequisites
- Install Python 3.10+ (64-bit)
- Install Ollama: see `https://ollama.com/download` (Windows support via official installer)
- Ensure `ollama` runs in a terminal: `ollama --version`

Defaults tuned for 16 GB RAM:
- TOP_K=3
- CHUNK_SIZE=800

### Notes
- PDFs with images/scans need OCR; this project reads text-based PDFs.
- Chroma stores data locally; to reset, delete the `CHROMA_DIR` directory.
- To reduce memory usage: lower `top_k`, reduce chunk size, or switch to a smaller model.

### Project structure
```
rag/
  __init__.py
  config.py
  utils.py
  ingest.py
  query.py
  serve.py
requirements.txt
.env.example
data/ 
```


