import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
    # App Settings
    project_name: str = "Production RAG System"
    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: List[str] = ["*"]
    
    # Paths
    root_dir: str = os.getcwd()
    chroma_persist_dir: str = os.path.join(root_dir, "data/chroma")
    golden_dataset_path: str = os.path.join(root_dir, "data/golden_dataset.jsonl")
    backup_storage_path: str = os.path.join(root_dir, "backups")

    # Vector Database
    vector_db_backend: str = "qdrant"  # "qdrant" or "chromadb"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_use_https: bool = False
    qdrant_collection: str = "rag_production"
    on_disk_payload: bool = True
    
    # Vector Configuration
    vector_size: int = 384
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    enable_quantization: bool = True
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # "cuda" if GPU available
    
    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 128
    enable_deduplication: bool = True
    document_retention_days: int = 365
    superseded_retention_hours: int = 24

    # Retrieval
    enable_hybrid_search: bool = True
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    similarity_threshold: float = 0.6
    enable_reranking: bool = True
    rerank_top_n: int = 10
    enable_query_expansion: bool = True
    multi_query_count: int = 3
    enable_contextual_compression: bool = False

    # Generation (LLM)
    llm_provider: str = "openai" # openai, anthropic, google
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1000
    llm_max_retries: int = 3
    llm_timeout: int = 30
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    enable_streaming: bool = True
    enable_citations: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# Metadata Schema definition used in ingestion.py
METADATA_SCHEMA = {
    "required": ["source"],
    "properties": {
        "source": {"type": "string"},
        "title": {"type": "string"},
        "author": {"type": "string"},
        "category": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "language": {"type": "string"},
        "version": {"type": "integer"}
    }
}