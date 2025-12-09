# Production-Ready RAG System with CI/CD

A complete, production-grade Retrieval-Augmented Generation (RAG) system with full CI/CD pipeline integration, optimized for performance, accuracy, and automated maintenance.

## Features

### Core Capabilities
- **Hybrid Search**: Combines dense vector search (embeddings) with sparse keyword search (BM25)
- **Advanced Retrieval**: Query expansion, reranking, and contextual compression
- **Smart Ingestion**: Automatic deduplication, versioning, and metadata management
- **Production-Ready API**: Async FastAPI with streaming support
- **Comprehensive Evaluation**: Built-in metrics for retrieval and generation quality

### Core Components Built:

# 1.Configuration Management (src/core/config.py)

- Environment-specific settings
- Feature flags
- Performance tuning parameters


# 2.Embeddings (src/core/embeddings.py)

- Sentence-transformers integration
- Caching with LRU
- Batch processing


# 3.Vector Store (src/core/vector_store.py)

- Qdrant with quantization (4x memory reduction)
- HNSW indexing for fast ANN search
- Metadata filtering
- ChromaDB support for development


# 4.LLM Integration (src/core/llm.py)

- LiteLLM wrapper supporting multiple providers
- Retry logic with exponential backoff
- Streaming support


# 5.Document Ingestion (src/pipeline/ingestion.py)

- Multi-format support (PDF, TXT, MD, HTML, DOCX)
- Sliding window chunking with overlap
- Automatic metadata enrichment
- Content hashing for deduplication


# 6.Retrieval Pipeline (src/pipeline/retrieval.py)

- Hybrid search: Dense (70%) + Sparse BM25 (30%)
- Query expansion and rewriting
- Cross-encoder reranking
- Contextual compression


# 7.Generation Pipeline (src/pipeline/generation.py)

- RAG-specific prompts
- Citation extraction
- Confidence scoring
- Streaming responses


# 8.FastAPI Application (src/api/main.py)

- production endpoints
- Async request handling
- Request ID tracking
- Health checks & metrics


# 9.Data Management (src/data/management.py)

- Content-based deduplication
- Document versioning
- Automated cleanup
- Index optimization


# 10.Evaluation Framework (src/data/evaluation.py)

- Retrieval metrics (Precision@K, Recall, MRR, NDCG)
- Generation metrics (faithfulness, relevance)



### Optimization Features
- **Vector Quantization**: 4x memory reduction with scalar quantization
- **HNSW Indexing**: Fast approximate nearest neighbor search
- **Caching**: Redis-based caching for embeddings and queries
- **Batch Processing**: Efficient bulk document ingestion

### CI/CD Automation
- **Automated Deduplication**: Content-based duplicate detection
- **Version Management**: Automatic document versioning
- **Index Rebuilding**: Incremental updates and optimization
- **Scheduled Maintenance**: Daily cleanup and weekly optimization
- **Blue-Green Deployments**: Zero-downtime updates



## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 8GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd rag-system
```

2. **Install dependencies**
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

3. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Start services**
```bash
# Start all services (Qdrant, Redis, PostgreSQL, etc.)
docker-compose -f deployment/docker-compose.yml up -d
```

5. **Initialize the system**
```bash
# Create vector collections
poetry run python scripts/setup.py
```

6. **Run the application**
```bash
poetry run uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage Examples

### 1. Ingest Documents

```python
import requests

# Single document
response = requests.post("http://localhost:8000/ingest", json={
    "title": "Machine Learning Guide",
    "content": "Machine learning is a subset of artificial intelligence...",
    "source": "ml-guide.pdf",
    "category": "education",
    "tags": ["ml", "ai", "tutorial"]
})

print(response.json())
# {"document_id": "...", "chunks_created": 5, "status": "success"}
```

### 2. Query the System

```python
# Query with sources
response = requests.post("http://localhost:8000/query", json={
    "query": "What is machine learning?",
    "top_k": 5,
    "include_sources": True,
    "enable_reranking": True
})

result = response.json()
print(result["answer"])
print(f"Confidence: {result['confidence_score']}")
print(f"Sources: {len(result['sources'])}")
```

### 3. Batch Ingestion

```python
documents = [
    {"title": "Doc 1", "content": "...", "source": "doc1.pdf"},
    {"title": "Doc 2", "content": "...", "source": "doc2.pdf"},
]

response = requests.post("http://localhost:8000/ingest/batch", json={
    "documents": documents
})

print(response.json())
# {"job_id": "...", "status": "processing", "total_documents": 2}
```

## Configuration

### Environment Variables

```bash
# Environment
ENVIRONMENT=production  # development, staging, production
DEBUG=false

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
ENABLE_QUANTIZATION=true

# LLM Provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
OPENAI_API_KEY=your_key_here

# Retrieval Settings
TOP_K=5
ENABLE_HYBRID_SEARCH=true
DENSE_WEIGHT=0.7
SPARSE_WEIGHT=0.3
ENABLE_RERANKING=true

# Document Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=128
ENABLE_DEDUPLICATION=true
```

### Performance Tuning

**For High Throughput:**
```yaml
# Increase workers
API_WORKERS=8
EMBEDDING_BATCH_SIZE=64

# Enable quantization
ENABLE_QUANTIZATION=true
ON_DISK_PAYLOAD=true
```



## Testing

### Run All Tests
```bash
poetry run pytest tests/ -v --cov=src
```

### Test Categories
```bash
# Unit tests
poetry run pytest tests/test_ingestion.py -v

# Integration tests
poetry run pytest tests/test_integration.py -v

# API tests
poetry run pytest tests/test_api.py -v
```

### Evaluation
```bash
# Run on golden dataset
poetry run python scripts/evaluate.py

# Custom dataset
poetry run python scripts/evaluate.py --dataset data/my_queries.jsonl
```

## Monitoring

### Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```

Returns:
- Query throughput and latency (p50, p95, p99)
- Document and chunk counts
- Cache hit rates
- Error rates

### Health Check
```bash
curl http://localhost:8000/health
```

### Grafana Dashboards
Access at `http://localhost:3000` (default: admin/admin)

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t rag-system:latest -f deployment/Dockerfile .

# Run container
docker run -p 8000:8000 --env-file .env rag-system:latest
```



## Maintenance

### Scheduled Tasks

**Daily (2 AM UTC):**
- Cleanup superseded documents
- Compact vector collections
- Clear cache

**Weekly (Sunday 3 AM):**
- Full backup
- Index optimization
- Analytics report

### Manual Maintenance
```bash
# Cleanup old documents
poetry run python scripts/maintenance.py --cleanup

# Rebuild index
poetry run python scripts/maintenance.py --rebuild-index

# Create backup
poetry run python scripts/maintenance.py --backup
```

## Project Structure

```
rag-system/
├── src/
│   ├── api/           # FastAPI application
│   ├── core/          # Core components (embeddings, vector store, LLM)
│   ├── pipeline/      # Ingestion, retrieval, generation pipelines
│   ├── data/          # Data management and evaluation
│   └── utils/         # Utilities
├── tests/             # Test suite
├── scripts/           # Setup and maintenance scripts
├── deployment/        # Docker and compose
├── data/              # Data files 

```


### Quick start
```bash
# 1. Install
make install

# 2. Start services
make docker-up

# 3. Initialize
make setup

# 4. Run dev server
make dev

# 5. Test
make test

```
## Troubleshooting

### Common Issues

**1. Qdrant connection failed**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant
```

**2. Out of memory**
```bash
# Enable quantization
export ENABLE_QUANTIZATION=true

# Use disk storage
export ON_DISK_PAYLOAD=true
```

**3. Slow queries**
```bash
# Check index status
curl http://localhost:6333/collections/documents

# Rebuild index
poetry run python scripts/maintenance.py --rebuild-index
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `poetry run pytest`
5. Submit a pull request





