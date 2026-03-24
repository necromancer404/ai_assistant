## RAG with Ollama (TXT, PDF, JSON) - Windows friendly

This project provides a simple local RAG pipeline powered by Ollama:
- Ingests `.txt`, `.pdf`, and `.json` files from a directory
- Stores embeddings locally using Chroma (SQLite-backed)
- Answers questions via retrieval + generation
- Includes a CLI and an optional FastAPI server

Works offline on a 16 GB RAM laptop with small models.

### 1) Prerequisites
- Install Python 3.10+ (64-bit)
- Install Ollama: see `https://ollama.com/download` (Windows support via official installer)
- Ensure `ollama` runs in a terminal: `ollama --version`

### 2) Pull recommended models (fit 16 GB RAM)
Use smaller models for reliability on 16 GB RAM:
- LLM (choose one):
  - `llama3.2:3b` (recommended)
  - `mistral:7b`
- Embeddings:
  - `nomic-embed-text`

Commands:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Tips:
- Close heavy apps before running the LLM.
- If RAM is tight, try `qwen2:7b` or smaller quant variants.

### 3) Create and activate venv, then install deps
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 4) Configure (optional)
Copy `.env.example` to `.env` and adjust:
- `OLLAMA_BASE_URL` (if Ollama runs on a non-default address)
- `GENERATION_MODEL` (default: llama3.2:3b), `EMBEDDING_MODEL`
- `CHROMA_DIR` for vector store persistence

### 5) Add your documents
Create a `data` directory and drop files:
- `.txt` — plain text
- `.pdf` — text-based PDFs
- `.json` — either a JSON object/array; text will be extracted from values

### 6) Ingest your data
```bash
python -m rag.ingest --data-dir data
```

Optional: convert JSON to pretty .txt before ingest (may help retrieval readability):
```bash
python -m rag.export_json_txt --data-dir data
```
This creates alongside each `*.json`:
- `*.txt` — pretty-printed JSON
- `*.flatten.txt` — flattened key paths with scalar values (better for search)

Export per faculty page into structured TXT files:
```bash
python -m rag.export_faculty_pages --json data\faculty_data_complete.json --out-dir data\faculty_pages
```
Each output file is named from the `faculty_page_url` and contains lines like:
```
"Jane Doe" has "Associate Professor" in "Computer Science" with "EMP123" and "2345" with "jane@univ.edu" and has "PhD" with "AI; NLP" and "Deep Learning" with "scholar123"
```

This will:
- Recursively read supported files
- Chunk them with overlap
- Compute embeddings with `nomic-embed-text` via Ollama
- Save vectors in Chroma (persisted under `./chroma/`)

### 7) Query (CLI)
```bash
python -m rag.query --question "What does the contract say about termination?"
```

### 8) API server (optional) + simple web UI
```bash
uvicorn rag.serve:app --host 0.0.0.0 --port 8000
```

Then POST:
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"Explain the policy on refunds\"}"
```

Open the simple UI:
- Open `web/index.html` in your browser
- Ensure the API URL is `http://localhost:8000/query`

Defaults tuned for 16 GB RAM:
- TOP_K=3
- CHUNK_SIZE=800

### Notes
- PDFs with images/scans need OCR; this project reads text-based PDFs. For OCR, integrate `pytesseract` + `pdfplumber` or `unstructured`.
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
data/ (create yourself)
```


