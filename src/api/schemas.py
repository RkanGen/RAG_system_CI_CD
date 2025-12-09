from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# --- Ingestion Schemas ---
class IngestRequest(BaseModel):
    content: str
    source: str
    title: Optional[str] = "Untitled"
    author: Optional[str] = None
    category: Optional[str] = "General"
    tags: List[str] = []
    language: str = "en"
    metadata: Optional[Dict[str, Any]] = {}

class IngestResponse(BaseModel):
    document_id: str
    status: str
    chunks_created: int
    version: int
    message: str

class BatchIngestRequest(BaseModel):
    documents: List[IngestRequest]

class BatchIngestResponse(BaseModel):
    job_id: str
    total_documents: int
    status: str
    message: str

class UpdateDocumentRequest(BaseModel):
    content: Optional[str] = None
    title: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class UpdateResponse(BaseModel):
    document_id: str
    status: str
    version: int
    chunks_updated: int
    message: str

class DeleteResponse(BaseModel):
    document_id: str
    status: str
    message: str

# --- Internal Data Models ---
class DocumentMetadata(BaseModel):
    source: str
    title: Optional[str]
    chunk_index: int
    chunk_type: str
    token_count: int
    timestamp: Optional[str]
    document_id: str
    version: int

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    chunk_type: str = "text"
    token_count: int
    metadata: Dict[str, Any]

# --- Retrieval & Generation Schemas ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    enable_reranking: bool = True
    include_sources: bool = True
    stream: bool = False

class SourceDocument(BaseModel):
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_index: int
    source: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query_id: str
    latency_ms: float
    confidence_score: float
    metadata: Dict[str, Any]

class RetrievalResult(BaseModel):
    chunks: List[DocumentChunk]
    scores: List[float]
    retrieval_time_ms: float
    reranked: bool

class GenerationResult(BaseModel):
    answer: str
    citations: List[str]
    generation_time_ms: float
    token_count: int
    confidence_score: float

# --- System Schemas ---
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, Any]

class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_documents: int
    total_chunks: int
    cache_hit_rate: float
    error_rate: float
    uptime_seconds: float

class ErrorResponse(BaseModel):
    detail: str

# --- Evaluation Schemas ---
class EvaluationQuery(BaseModel):
    query_id: str
    query: str
    expected_doc_ids: List[str]
    expected_answer: Optional[str] = None

class EvaluationResult(BaseModel):
    timestamp: datetime
    metrics: Dict[str, float]
    query_results: List[Dict[str, Any]]
    total_queries: int
    passed_queries: int
    failed_queries: int