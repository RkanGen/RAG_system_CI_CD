import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from src.core.config import settings
from src.core.vector_store import get_vector_store
from src.core.embeddings import get_embedding_model
from src.pipeline.ingestion import get_ingestion_pipeline
from src.pipeline.retrieval import get_retrieval_pipeline
from src.pipeline.generation import get_generation_pipeline
from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    BatchIngestRequest,
    BatchIngestResponse,
    UpdateDocumentRequest,
    UpdateResponse,
    DeleteResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
    SourceDocument,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app."""
    # Startup
    logger.info("Starting RAG system...")
    
    # Initialize components
    vector_store = get_vector_store()
    embedding_model = get_embedding_model()
    embedding_model.warmup()
    
    logger.info("RAG system started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG system...")


# Create FastAPI app
app = FastAPI(
    title="Production RAG System",
    description="Complete RAG system with CI/CD integration",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Metrics storage (in production, use Redis or Prometheus)
_metrics = {
    "total_queries": 0,
    "total_documents": 0,
    "latencies": [],
    "errors": 0,
}


# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    fastapi_request: Request,
) -> QueryResponse:
    """Query the RAG system."""
    start_time = time.time()
    query_id = str(uuid4())
    
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Retrieve relevant documents
        retrieval_pipeline = get_retrieval_pipeline()
        retrieval_result = await retrieval_pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            enable_reranking=request.enable_reranking,
        )
        
        # Generate answer
        generation_pipeline = get_generation_pipeline()
        
        if request.stream and settings.enable_streaming:
            # Return streaming response
            async def stream_generator() -> AsyncIterator[str]:
                async for chunk in generation_pipeline.generate(
                    query=request.query,
                    retrieval_result=retrieval_result,
                    stream=True,
                ):
                    yield chunk
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/plain",
            )
        
        generation_result = await generation_pipeline.generate(
            query=request.query,
            retrieval_result=retrieval_result,
            stream=False,
            include_citations=request.include_sources,
        )
        
        # Build source documents
        sources = []
        if request.include_sources:
            for chunk, score in zip(retrieval_result.chunks, retrieval_result.scores):
                source = SourceDocument(
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=score,
                    metadata=chunk.metadata,
                    chunk_index=chunk.chunk_index,
                    source=chunk.metadata.get("source", ""),
                )
                sources.append(source)
        
        latency = (time.time() - start_time) * 1000
        
        # Update metrics
        _metrics["total_queries"] += 1
        _metrics["latencies"].append(latency)
        
        response = QueryResponse(
            answer=generation_result.answer,
            sources=sources,
            query_id=query_id,
            latency_ms=latency,
            confidence_score=generation_result.confidence_score,
            metadata={
                "retrieval_time_ms": retrieval_result.retrieval_time_ms,
                "generation_time_ms": generation_result.generation_time_ms,
                "total_chunks_retrieved": len(retrieval_result.chunks),
                "reranked": retrieval_result.reranked,
            },
        )
        
        logger.info(f"Query completed in {latency:.2f}ms")
        return response
    
    except Exception as e:
        _metrics["errors"] += 1
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Ingest endpoint
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Ingest a single document."""
    try:
        logger.info(f"Ingesting document: {request.title}")
        
        ingestion_pipeline = get_ingestion_pipeline()
        result = await ingestion_pipeline.ingest_document(request)
        
        _metrics["total_documents"] += 1
        
        logger.info(f"Document ingested successfully: {result['document_id']}")
        
        return IngestResponse(**result)
    
    except Exception as e:
        _metrics["errors"] += 1
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch ingest endpoint
@app.post("/ingest/batch", response_model=BatchIngestResponse)
async def batch_ingest(
    request: BatchIngestRequest,
    background_tasks: BackgroundTasks,
) -> BatchIngestResponse:
    """Ingest multiple documents."""
    job_id = str(uuid4())
    
    try:
        logger.info(f"Starting batch ingestion: {len(request.documents)} documents")
        
        # Process in background
        async def process_batch():
            ingestion_pipeline = get_ingestion_pipeline()
            results = await ingestion_pipeline.batch_ingest(request.documents)
            logger.info(f"Batch ingestion completed: {job_id}")
            return results
        
        background_tasks.add_task(process_batch)
        
        return BatchIngestResponse(
            job_id=job_id,
            total_documents=len(request.documents),
            status="processing",
            message=f"Batch ingestion started with {len(request.documents)} documents",
        )
    
    except Exception as e:
        logger.error(f"Error starting batch ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Update document endpoint
@app.put("/documents/{document_id}", response_model=UpdateResponse)
async def update_document(
    document_id: str,
    request: UpdateDocumentRequest,
) -> UpdateResponse:
    """Update an existing document."""
    try:
        logger.info(f"Updating document: {document_id}")
        
        ingestion_pipeline = get_ingestion_pipeline()
        result = await ingestion_pipeline.update_document(
            document_id=document_id,
            content=request.content,
            metadata_updates=request.dict(exclude_none=True),
        )
        
        return UpdateResponse(
            document_id=document_id,
            status="updated",
            version=result.get("version", 1),
            chunks_updated=result.get("chunks_created", 0),
            message="Document updated successfully",
        )
    
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Delete document endpoint
@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    hard_delete: bool = False,
) -> DeleteResponse:
    """Delete a document."""
    try:
        logger.info(f"Deleting document: {document_id}")
        
        ingestion_pipeline = get_ingestion_pipeline()
        result = await ingestion_pipeline.delete_document(
            document_id=document_id,
            soft_delete=not hard_delete,
        )
        
        return DeleteResponse(**result)
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    from datetime import datetime
    
    services = {}
    
    # Check vector store
    try:
        vector_store = get_vector_store()
        info = vector_store.get_collection_info()
        services["vector_store"] = {
            "status": "healthy",
            "backend": info.get("backend"),
            "documents": info.get("points_count", info.get("count", 0)),
        }
    except Exception as e:
        services["vector_store"] = {
            "status": "unhealthy",
            "error": str(e),
        }
    
    # Check embedding model
    try:
        embedding_model = get_embedding_model()
        services["embedding_model"] = {
            "status": "healthy",
            "model": embedding_model.model_name,
            "dimension": embedding_model.dimension,
        }
    except Exception as e:
        services["embedding_model"] = {
            "status": "unhealthy",
            "error": str(e),
        }
    
    # Overall status
    overall_status = "healthy" if all(
        s.get("status") == "healthy" for s in services.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        timestamp=datetime.utcnow(),
        services=services,
    )


# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get system metrics."""
    latencies = _metrics["latencies"]
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
    
    total_requests = _metrics["total_queries"]
    error_rate = _metrics["errors"] / total_requests if total_requests > 0 else 0
    
    return MetricsResponse(
        total_queries=_metrics["total_queries"],
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        p99_latency_ms=round(p99_latency, 2),
        total_documents=_metrics["total_documents"],
        total_chunks=0,  # Would come from vector store
        cache_hit_rate=0.0,  # Would come from cache
        error_rate=round(error_rate, 3),
        uptime_seconds=0.0,  # Would track from startup
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Production RAG System",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers if not settings.debug else 1,
    )