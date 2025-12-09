import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchParams,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarType,
    QuantizationSearchParams,
)
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings
from src.core.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStore:
    """Unified interface for vector database operations."""
    
    def __init__(self, backend: str = settings.vector_db_backend):
        self.backend = backend
        self.collection_name = settings.qdrant_collection
        self.embedding_model = get_embedding_model()
        
        if backend == "qdrant":
            self._init_qdrant()
        elif backend == "chromadb":
            self._init_chromadb()
        else:
            raise ValueError(f"Unsupported vector database: {backend}")
        
        logger.info(f"Initialized vector store: {backend}")
    
    def _init_qdrant(self) -> None:
        """Initialize Qdrant client and collection."""
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=settings.qdrant_use_https,
        )
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self._create_qdrant_collection()
        
        logger.info(f"Connected to Qdrant collection: {self.collection_name}")
    
    def _create_qdrant_collection(self) -> None:
        """Create Qdrant collection with optimized settings."""
        quantization_config = None
        if settings.enable_quantization:
            quantization_config = ScalarQuantization(
                scalar=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.vector_size,
                distance=Distance.COSINE,
                on_disk=False,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=10000,
            ),
            hnsw_config={
                "m": settings.hnsw_m,
                "ef_construct": settings.hnsw_ef_construct,
            },
            quantization_config=quantization_config,
            on_disk_payload=settings.on_disk_payload,
        )
        
        # Create payload indexes for metadata filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema="keyword",
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source",
            field_schema="keyword",
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="category",
            field_schema="keyword",
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="timestamp",
            field_schema="datetime",
        )
        
        logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.Client(
            ChromaSettings(
                persist_directory=settings.chroma_persist_dir,
                anonymized_telemetry=False,
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
    
    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add documents to the vector store."""
        if len(ids) != len(texts) != len(embeddings) != len(metadatas):
            raise ValueError("All input lists must have the same length")
        
        if self.backend == "qdrant":
            points = [
                PointStruct(
                    id=id_,
                    vector=embedding,
                    payload={**metadata, "text": text},
                )
                for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        
        elif self.backend == "chromadb":
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        
        logger.info(f"Added {len(ids)} documents to vector store")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Search for similar documents."""
        start_time = time.time()
        
        if self.backend == "qdrant":
            # Build filter
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_qdrant_filter(filters)
            
            # Search parameters
            search_params = SearchParams(
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                ) if settings.enable_quantization else None,
            )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                search_params=search_params,
                score_threshold=score_threshold,
            )
            
            documents = []
            scores = []
            for result in results:
                doc = result.payload.copy()
                doc["id"] = result.id
                documents.append(doc)
                scores.append(result.score)
        
        elif self.backend == "chromadb":
            where = filters if filters else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )
            
            documents = []
            scores = []
            if results["ids"] and results["ids"][0]:
                for i, id_ in enumerate(results["ids"][0]):
                    doc = {
                        "id": id_,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        **(results["metadatas"][0][i] if results["metadatas"] else {}),
                    }
                    documents.append(doc)
                    scores.append(1 - results["distances"][0][i])  # Convert distance to similarity
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Search completed in {elapsed:.2f}ms, found {len(documents)} results")
        
        return documents, scores
    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dict."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                                gt=value.get("gt"),
                                lt=value.get("lt"),
                            ),
                        )
                    )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        if self.backend == "qdrant":
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids,
            )
        elif self.backend == "chromadb":
            self.collection.delete(ids=ids)
        
        logger.info(f"Deleted {len(ids)} documents from vector store")
    
    def delete_by_filter(self, filters: Dict[str, Any]) -> None:
        """Delete documents matching filters."""
        if self.backend == "qdrant":
            qdrant_filter = self._build_qdrant_filter(filters)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_filter,
            )
            logger.info(f"Deleted documents matching filters from vector store")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if self.backend == "qdrant":
            info = self.client.get_collection(self.collection_name)
            return {
                "backend": "qdrant",
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
            }
        elif self.backend == "chromadb":
            count = self.collection.count()
            return {
                "backend": "chromadb",
                "name": self.collection_name,
                "count": count,
            }
    
    def optimize_collection(self) -> None:
        """Optimize the collection (indexing, compaction)."""
        if self.backend == "qdrant":
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )
            logger.info("Triggered Qdrant collection optimization")


# Global vector store instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store