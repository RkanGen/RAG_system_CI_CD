import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from rank_bm25 import BM25Okapi

from src.core.config import settings
from src.core.embeddings import get_embedding_model
from src.core.vector_store import get_vector_store
from src.api.schemas import RetrievalResult, DocumentChunk

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Handles query preprocessing and expansion."""
    
    def __init__(self):
        self.embedding_model = get_embedding_model()
    
    def preprocess_query(self, query: str) -> str:
        """Clean and normalize query."""
        query = query.strip()
        # Remove excessive whitespace
        query = " ".join(query.split())
        return query
    
    def expand_query(self, query: str) -> List[str]:
        """Generate query variations for better retrieval."""
        if not settings.enable_query_expansion:
            return [query]
        
        # Simple expansion - in production, use more sophisticated methods
        queries = [query]
        
        # Add question variations
        if not query.endswith("?"):
            queries.append(f"{query}?")
        
        # Add "what is" prefix if not present
        if not query.lower().startswith(("what", "how", "why", "when", "where", "who")):
            queries.append(f"What is {query}")
        
        return queries[:settings.multi_query_count]
    
    def rewrite_query(self, query: str) -> str:
        """Rewrite query for better retrieval (placeholder for LLM-based rewriting)."""
        # In production, use LLM to rewrite query
        return query


class BM25Retriever:
    """BM25-based sparse retrieval."""
    
    def __init__(self):
        self.corpus: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.k1 = settings.bm25_k1
        self.b = settings.bm25_b
    
    def index_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Index documents for BM25 search."""
        self.corpus = documents
        self.metadatas = metadatas
        
        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"Indexed {len(documents)} documents for BM25 search")
    
    def search(self, query: str, top_k: int = 10) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Search using BM25."""
        if self.bm25 is None:
            return [], []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        result_scores = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = {
                    "text": self.corpus[idx],
                    "metadata": self.metadatas[idx],
                    "bm25_score": float(scores[idx]),
                }
                results.append(doc)
                result_scores.append(float(scores[idx]))
        
        return results, result_scores


class HybridRetriever:
    """Combines dense and sparse retrieval."""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.embedding_model = get_embedding_model()
        self.bm25_retriever = BM25Retriever()
        self.dense_weight = settings.dense_weight
        self.sparse_weight = settings.sparse_weight
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[float], float]:
        """Retrieve documents using hybrid search."""
        start_time = time.time()
        
        if not settings.enable_hybrid_search:
            # Dense retrieval only
            return self._dense_retrieve(query, top_k, filters)
        
        # Dense retrieval
        dense_docs, dense_scores = self._dense_retrieve(query, top_k * 2, filters)
        
        # Sparse retrieval (BM25)
        sparse_docs, sparse_scores = self._sparse_retrieve(query, top_k * 2)
        
        # Combine using Reciprocal Rank Fusion (RRF)
        combined_docs, combined_scores = self._reciprocal_rank_fusion(
            dense_docs, dense_scores,
            sparse_docs, sparse_scores,
            top_k
        )
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return combined_docs, combined_scores, retrieval_time
    
    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Dense vector retrieval."""
        query_embedding = self.embedding_model.embed_query(query)
        
        docs, scores = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            score_threshold=settings.similarity_threshold,
        )
        
        return docs, scores
    
    def _sparse_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Sparse BM25 retrieval."""
        if self.bm25_retriever.bm25 is None:
            # BM25 not initialized, return empty
            return [], []
        
        return self.bm25_retriever.search(query, top_k)
    
    def _reciprocal_rank_fusion(
        self,
        dense_docs: List[Dict[str, Any]],
        dense_scores: List[float],
        sparse_docs: List[Dict[str, Any]],
        sparse_scores: List[float],
        top_k: int,
        k: int = 60,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Combine results using Reciprocal Rank Fusion."""
        # Build document map
        doc_scores = defaultdict(float)
        doc_map = {}
        
        # Add dense results
        for rank, (doc, score) in enumerate(zip(dense_docs, dense_scores), 1):
            doc_id = doc.get("id", doc.get("document_id", str(rank)))
            rrf_score = self.dense_weight / (k + rank)
            doc_scores[doc_id] += rrf_score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Add sparse results
        for rank, (doc, score) in enumerate(zip(sparse_docs, sparse_scores), 1):
            doc_id = doc.get("id", doc.get("document_id", str(rank)))
            rrf_score = self.sparse_weight / (k + rank)
            doc_scores[doc_id] += rrf_score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        combined_docs = [doc_map[doc_id] for doc_id, _ in sorted_docs]
        combined_scores = [score for _, score in sorted_docs]
        
        return combined_docs, combined_scores


class Reranker:
    """Cross-encoder reranking for improved relevance."""
    
    def __init__(self):
        # In production, load a cross-encoder model
        # from sentence_transformers import CrossEncoder
        # self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pass
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        scores: List[float],
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Rerank documents using cross-encoder."""
        if not settings.enable_reranking or not documents:
            return documents[:top_k], scores[:top_k]
        
        # Placeholder: In production, use actual cross-encoder
        # For now, just return top documents by existing scores
        sorted_pairs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in sorted_pairs[:top_k]]
        reranked_scores = [score for _, score in sorted_pairs[:top_k]]
        
        logger.debug(f"Reranked {len(documents)} documents to top {top_k}")
        
        return reranked_docs, reranked_scores


class ContextualCompressor:
    """Compress retrieved context to most relevant parts."""
    
    def __init__(self):
        pass
    
    def compress(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compress documents to most relevant sentences."""
        if not settings.enable_contextual_compression:
            return documents
        
        # Placeholder: In production, use extractive summarization
        # For now, just truncate very long documents
        compressed_docs = []
        for doc in documents:
            text = doc.get("text", "")
            if len(text) > 1000:
                # Keep first 1000 chars
                doc["text"] = text[:1000] + "..."
            compressed_docs.append(doc)
        
        return compressed_docs


class RetrievalPipeline:
    """Complete retrieval pipeline."""
    
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
        self.compressor = ContextualCompressor()
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: bool = True,
    ) -> RetrievalResult:
        """Execute full retrieval pipeline."""
        start_time = time.time()
        
        # Preprocess query
        processed_query = self.query_processor.preprocess_query(query)
        
        # Retrieve documents
        rerank_top_k = settings.rerank_top_n if enable_reranking else top_k
        docs, scores, search_time = self.hybrid_retriever.retrieve(
            processed_query,
            top_k=rerank_top_k,
            filters=filters,
        )
        
        # Rerank if enabled
        reranked = False
        if enable_reranking and len(docs) > top_k:
            docs, scores = self.reranker.rerank(processed_query, docs, scores, top_k)
            reranked = True
        
        # Compress context
        docs = self.compressor.compress(processed_query, docs)
        
        # Convert to DocumentChunk objects
        chunks = []
        for doc, score in zip(docs, scores):
            chunk = DocumentChunk(
                chunk_id=doc.get("id", ""),
                document_id=doc.get("document_id", ""),
                content=doc.get("text", ""),
                chunk_index=doc.get("chunk_index", 0),
                chunk_type=doc.get("chunk_type", "text"),
                token_count=doc.get("token_count", 0),
                metadata=doc,
            )
            chunks.append(chunk)
        
        total_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            retrieval_time_ms=total_time,
            reranked=reranked,
        )


# Global pipeline instance
_retrieval_pipeline: Optional[RetrievalPipeline] = None


def get_retrieval_pipeline() -> RetrievalPipeline:
    """Get or create the global retrieval pipeline instance."""
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline()
    return _retrieval_pipeline