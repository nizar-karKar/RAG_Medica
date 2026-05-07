# RAG_Medica

**MedRAG Intelligence** — a Retrieval-Augmented Generation (RAG) application for analyzing medical PDF documents through a chat interface that supports both **text** and **voice** queries.

Users upload medical documents, the system indexes them into a local vector database, and an LLM answers questions grounded in the uploaded content.

---

## Features

- Upload medical PDFs and query them in natural language.
- Hybrid retrieval combining **BM25 keyword search** and **semantic (embedding) search** for accurate, terminology-aware results.
- **Voice queries** — record from the browser, transcribe, and answer.
- Per-document filtering: scope a question to a specific uploaded file.
- Local embeddings via Ollama — your document vectors stay on your machine.
- Clean modular backend ready for Docker deployment.

## Tech Stack

| Layer        | Technology                                          |
|--------------|-----------------------------------------------------|
| Frontend     | Vite, Vanilla JS, Axios                             |
| Backend      | FastAPI, Uvicorn, LangChain                         |
| Vector Store | ChromaDB                                            |
| Embeddings   | Ollama (`nomic-embed-text`)                         |
| LLM          | OpenAI `gpt-4o-mini`                                |
| Keyword      | BM25 (`rank_bm25`)                                  |
| Speech-to-Text | ElevenLabs                                        |

## Application Components

The backend is organised into four clear layers:

- **Ingestion** (`backend/ingestion/`) — loads PDFs, splits them into chunks, tags them with metadata, and stores them in ChromaDB.
- **Retrieval** (`backend/retrieval/`) — wraps the vector store and provides semantic, keyword (BM25), and hybrid retrievers.
- **Generation** (`backend/generation/`) — builds the prompt from retrieved context, calls the LLM, and handles voice transcription.
- **API** (`backend/api/`) — FastAPI routes (`/rag/upload`, `/rag/query`, `/rag/voice-query`, `/health`) and Pydantic schemas.

The frontend (`frontend/`) is a single-page Vite app providing the upload zone, chat area, and voice recording controls.

## Project Structure

```
RAG_Medica/
├── backend/
│   ├── api/            # FastAPI routes & schemas
│   ├── ingestion/      # PDF loading, chunking, storage
│   ├── retrieval/      # Vector store, semantic & BM25 retrievers, hybrid fusion
│   ├── generation/     # LLM response & voice transcription
│   ├── evaluation/     # Evaluation scripts
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/           # Vite + Vanilla JS chat UI
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com/) running locally with `nomic-embed-text` pulled
- An OpenAI API key (and an ElevenLabs key for voice queries)

```bash
ollama pull nomic-embed-text
```

### Backend

```bash
cd backend
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Create a `.env` file at the repo root:

```env
OPENAI_API_KEY=sk-...
ELEVENT_LAB_API=...
```

API docs: <http://127.0.0.1:8000/docs>

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API

| Method | Endpoint           | Description                                 |
|--------|--------------------|---------------------------------------------|
| GET    | `/health/`         | Health check                                |
| POST   | `/rag/upload`      | Upload a PDF and ingest it                  |
| POST   | `/rag/query`       | Ask a text question                         |
| POST   | `/rag/voice-query` | Send recorded audio → transcribe → answer   |

## Docker

```bash
cd backend
docker build -t rag-medica .
docker run --rm -p 8000:8000 --env-file ../.env rag-medica
```

## License

For research and educational use.
