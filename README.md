# QdrantRAG

A Retrieval-Augmented Generation (RAG) system built on [Qdrant](https://qdrant.tech/) and [OpenAI](https://openai.com/).
It ingests documents (PDF, TXT, JSON), creates vector embeddings, and answers natural-language questions by combining semantic search with keyword matching.

---

## How It Works

```
 ┌──────────┐      ┌──────────────┐      ┌───────────┐
 │ Documents │─────▶│  Embeddings  │─────▶│  Qdrant   │
 │ PDF/TXT/… │      │  (OpenAI)    │      │  Vector DB│
 └──────────┘      └──────────────┘      └─────┬─────┘
                                               │
 ┌──────────┐      ┌──────────────┐            │
 │  Query   │─────▶│ Hybrid Search│◀───────────┘
 └──────────┘      │ vector+keyword│
                   └──────┬───────┘
                          │
                   ┌──────▼───────┐
                   │   Response   │
                   │  Generator   │
                   │   (GPT-4)    │
                   └──────────────┘
```

1. **Ingest** — documents are chunked, embedded via OpenAI, and stored in Qdrant.
2. **Search** — a query is embedded and matched against stored vectors; a keyword score is blended in for precision.
3. **Respond** — the top-ranked chunks are sent to GPT-4, which synthesises a sourced answer with a confidence score.

---

## Project Structure

```
├── core/                        # Library code
│   ├── config.py                # Pydantic settings (reads .env)
│   ├── models/                  # Data models (Document, SearchResult)
│   ├── database/                # Qdrant client wrapper & document store
│   ├── services/                # Embedding, search engine, response generation
│   └── parsers/                 # File-format parsers (PDF, …)
│
├── scripts/                     # Runnable entry-points
│   ├── setup_database.py        # Create / reset the Qdrant collection
│   ├── ingest_documents.py      # Ingest a single file or directory
│   ├── ingest_all_documents.py  # Batch-ingest everything in data/documents/
│   └── interactive_search.py    # Interactive CLI for search & Q&A
│
├── data/
│   └── documents/               # Drop your source files here
│
├── tests/                       # Pytest test suite
├── notebooks/                   # Jupyter analysis notebooks
├── docker-compose.yml           # Qdrant + Redis containers
└── requirements.txt
```

---

## Getting Started

### Prerequisites

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Runtime |
| Docker & Docker Compose | Run Qdrant (and optional Redis) |
| OpenAI API key | Embeddings + response generation |

### 1. Clone and install

```bash
git clone <your-repo-url>
cd rag

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file in the project root (or copy `.env.example` if present):

```dotenv
OPENAI_API_KEY=sk-…

# Optional — defaults are fine for local development
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MODEL=gpt-4
```

See [core/config.py](core/config.py) for every available setting and its default.

### 3. Start infrastructure

```bash
docker-compose up -d      # starts Qdrant on :6333 and Redis on :6379
```

Qdrant dashboard: <http://localhost:6333/dashboard>

### 4. Initialise the database

```bash
python scripts/setup_database.py
```

### 5. Ingest documents

Place files in `data/documents/` (PDF, TXT, MD, JSON), then:

```bash
# Ingest everything under data/documents/
python scripts/ingest_all_documents.py

# Or ingest a single file
python scripts/ingest_documents.py --file path/to/document.pdf
```

### 6. Search

```bash
python scripts/interactive_search.py
```

Inside the interactive CLI:

| Command | Description |
|---------|-------------|
| `search <query>` | Hybrid search over documents |
| `ask <question>` | Get an AI-generated, sourced answer |
| `config` | Adjust search weights and filters |
| `stats` | Show collection statistics |
| `help` | Full command reference |
| `quit` | Exit |

---

## Key Concepts

### Hybrid Search

Every query runs through two scoring passes:

- **Vector score** — cosine similarity between the query embedding and stored embeddings.
- **Keyword score** — term-frequency overlap between query tokens and document text.

The final ranking is a weighted blend:

```
score = w_vector × vector_score + w_keyword × keyword_score
```

Weights auto-adjust based on query type (technical queries lean on keywords; conversational queries lean on vectors), or you can set them manually.

### Chunking & Embedding

Documents are split into token-bounded chunks (default 512 tokens, 50-token overlap) and embedded with OpenAI's `text-embedding-3-small` (1 536 dimensions). An in-memory cache avoids re-embedding duplicate text.

### Response Generation

Top-ranked chunks are assembled into a context prompt and sent to GPT-4 via [Instructor](https://github.com/jxnl/instructor) for structured output. Each response includes:

- The synthesised **answer**
- A **confidence score** (0–1)
- **Reasoning steps** explaining how the answer was derived
- **Source references** back to the original documents

---

## Configuration Reference

All settings live in [core/config.py](core/config.py) and can be overridden with environment variables or a `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4` | Model for response generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | `qdrant_rag_collection` | Collection name |
| `DEFAULT_VECTOR_WEIGHT` | `0.7` | Vector weight in hybrid score |
| `DEFAULT_KEYWORD_WEIGHT` | `0.3` | Keyword weight in hybrid score |
| `MIN_SEARCH_SCORE` | `0.6` | Minimum score threshold |
| `CHUNK_SIZE_TOKENS` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between chunks |
| `MAX_RESPONSE_TOKENS` | `1000` | Max tokens in generated response |
| `RESPONSE_TEMPERATURE` | `0.1` | LLM temperature |

---

## Running Tests

```bash
pytest tests/
```

---

## Docker Services

`docker-compose.yml` defines:

| Service | Port | Purpose |
|---------|------|---------|
| **qdrant** | 6333 / 6334 | Vector database (HTTP + gRPC) |
| **redis** | 6379 | Optional query cache |
| **jupyter** | 8888 | Optional notebook server |

Start only what you need:

```bash
docker-compose up -d qdrant          # just the database
docker-compose up -d                 # everything
```

---

## Contributing

1. Fork the repo and create a feature branch.
2. Install dev dependencies: `pip install pytest black flake8`
3. Run `pytest tests/` and `black .` before opening a PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT — see [LICENSE](LICENSE).
