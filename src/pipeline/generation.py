import logging
import time
import re
from typing import List, Dict, Any, Optional, AsyncIterator

from src.core.config import settings
from src.core.llm import get_llm_client
from src.api.schemas import GenerationResult, DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class PromptTemplate:
    """RAG prompt templates."""
    
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

Guidelines:
- Answer based ONLY on the information in the context provided
- If the context doesn't contain enough information to answer the question, say so
- Be concise but complete in your answers
- If you quote from the context, use quotation marks
- Maintain a professional and helpful tone"""
    
    RAG_TEMPLATE = """Context information is below:
---
{context}
---

Based on the context above, please answer the following question. If the context doesn't contain relevant information, acknowledge this limitation.

Question: {query}

Answer:"""
    
    RAG_WITH_CITATIONS_TEMPLATE = """Context information with source IDs is below:
---
{context_with_sources}
---

Based on the context above, please answer the following question. When using information from the context, cite the source by including [Source: source_id] after the relevant information.

Question: {query}

Answer:"""


class CitationExtractor:
    """Extract and format citations from generated text."""
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citation references from text."""
        # Match patterns like [Source: doc_123] or [1] or [doc_123]
        pattern = r'\[(?:Source:\s*)?([^\]]+)\]'
        matches = re.findall(pattern, text)
        return list(set(matches))
    
    def format_sources(
        self,
        chunks: List[DocumentChunk],
        citations: List[str],
    ) -> List[Dict[str, Any]]:
        """Format source information for cited chunks."""
        sources = []
        
        for chunk in chunks:
            # Check if this chunk was cited
            chunk_id = chunk.chunk_id or chunk.document_id
            if any(citation in chunk_id for citation in citations):
                sources.append({
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.metadata.get("source", ""),
                    "title": chunk.metadata.get("title", ""),
                    "content_preview": chunk.content[:200] + "...",
                })
        
        return sources


class ResponseGenerator:
    """Generate responses using retrieved context."""
    
    def __init__(self):
        self.llm_client = get_llm_client()
        self.citation_extractor = CitationExtractor()
    
    def build_context(
        self,
        chunks: List[DocumentChunk],
        include_citations: bool = False,
    ) -> str:
        """Build context string from retrieved chunks."""
        if include_citations:
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                source_id = f"Source_{i}"
                context_parts.append(f"[{source_id}]\n{chunk.content}\n")
            return "\n".join(context_parts)
        else:
            return "\n\n".join([chunk.content for chunk in chunks])
    
    def build_prompt(
        self,
        query: str,
        chunks: List[DocumentChunk],
        include_citations: bool = False,
    ) -> str:
        """Build the complete prompt for generation."""
        if include_citations:
            context = self.build_context(chunks, include_citations=True)
            return PromptTemplate.RAG_WITH_CITATIONS_TEMPLATE.format(
                context_with_sources=context,
                query=query,
            )
        else:
            context = self.build_context(chunks, include_citations=False)
            return PromptTemplate.RAG_TEMPLATE.format(
                context=context,
                query=query,
            )
    
    async def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        include_citations: bool = True,
    ) -> GenerationResult:
        """Generate answer from query and retrieved context."""
        start_time = time.time()
        
        if not retrieval_result.chunks:
            # No context available
            answer = "I don't have enough information to answer this question based on the available documents."
            return GenerationResult(
                answer=answer,
                citations=[],
                generation_time_ms=0,
                token_count=len(answer.split()),
                confidence_score=0.0,
            )
        
        # Build prompt
        prompt = self.build_prompt(
            query,
            retrieval_result.chunks,
            include_citations=include_citations and settings.enable_citations,
        )
        
        # Generate response
        try:
            answer = await self.llm_client.agenerate(
                prompt=prompt,
                system_prompt=PromptTemplate.SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = "I encountered an error while generating the answer. Please try again."
        
        # Extract citations
        citations = []
        if include_citations and settings.enable_citations:
            citations = self.citation_extractor.extract_citations(answer)
        
        # Calculate confidence score based on retrieval scores
        confidence_score = self._calculate_confidence(retrieval_result.scores)
        
        generation_time = (time.time() - start_time) * 1000
        token_count = self.llm_client.count_tokens(answer)
        
        return GenerationResult(
            answer=answer,
            citations=citations,
            generation_time_ms=generation_time,
            token_count=token_count,
            confidence_score=confidence_score,
        )
    
    async def generate_stream(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> AsyncIterator[str]:
        """Generate answer with streaming."""
        if not retrieval_result.chunks:
            yield "I don't have enough information to answer this question."
            return
        
        prompt = self.build_prompt(query, retrieval_result.chunks, include_citations=False)
        
        try:
            async for chunk in self.llm_client.agenerate_stream(
                prompt=prompt,
                system_prompt=PromptTemplate.SYSTEM_PROMPT,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield "\n\n[Error occurred during generation]"
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence score based on retrieval scores."""
        if not scores:
            return 0.0
        
        # Use average of top scores
        avg_score = sum(scores[:3]) / min(len(scores), 3)
        
        # Normalize to 0-1 range
        confidence = min(avg_score, 1.0)
        
        return round(confidence, 3)


class GenerationPipeline:
    """Complete generation pipeline."""
    
    def __init__(self):
        self.generator = ResponseGenerator()
    
    async def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        stream: bool = False,
        include_citations: bool = True,
    ) -> GenerationResult | AsyncIterator[str]:
        """Execute generation pipeline."""
        if stream and settings.enable_streaming:
            return self.generator.generate_stream(query, retrieval_result)
        else:
            return await self.generator.generate(
                query,
                retrieval_result,
                include_citations=include_citations,
            )


# Global pipeline instance
_generation_pipeline: Optional[GenerationPipeline] = None


def get_generation_pipeline() -> GenerationPipeline:
    """Get or create the global generation pipeline instance."""
    global _generation_pipeline
    if _generation_pipeline is None:
        _generation_pipeline = GenerationPipeline()
    return _generation_pipeline