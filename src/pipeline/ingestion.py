import logging
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from uuid import uuid4, UUID
from pathlib import Path
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from docx import Document as DocxDocument

from src.core.config import settings, METADATA_SCHEMA
from src.core.embeddings import get_embedding_model
from src.core.vector_store import get_vector_store
from src.api.schemas import IngestRequest, DocumentMetadata, DocumentChunk

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading, chunking, and preprocessing."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )
    
    def load_document(self, file_path: str, file_type: Optional[str] = None) -> str:
        """Load document content from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_type is None:
            file_type = path.suffix.lower()
        
        try:
            if file_type == ".pdf":
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                content = "\n\n".join([page.page_content for page in pages])
            
            elif file_type == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            elif file_type == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
                content = "\n\n".join([doc.page_content for doc in docs])
            
            elif file_type == ".html":
                loader = UnstructuredHTMLLoader(file_path)
                docs = loader.load()
                content = "\n\n".join([doc.page_content for doc in docs])
            
            elif file_type == ".docx":
                doc = DocxDocument(file_path)
                content = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return self.preprocess_text(content)
        
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize unicode
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        chunks = self.text_splitter.split_text(text)
        
        chunked_docs = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["token_count"] = len(chunk_text.split())
            chunk_metadata["chunk_type"] = self._detect_chunk_type(chunk_text)
            
            chunked_docs.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
            })
        
        return chunked_docs
    
    def _detect_chunk_type(self, text: str) -> str:
        """Detect the type of chunk (text, code, list, etc.)."""
        # Simple heuristics
        if re.match(r'^```|^    |\t', text, re.MULTILINE):
            return "code"
        elif re.match(r'^[\*\-\+]\s|\d+\.\s', text, re.MULTILINE):
            return "list"
        elif re.match(r'^#{1,6}\s', text):
            return "header"
        elif "|" in text and "---" in text:
            return "table"
        else:
            return "text"
    
    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata against schema."""
        required_fields = METADATA_SCHEMA["required"]
        
        for field in required_fields:
            if field not in metadata:
                logger.error(f"Missing required metadata field: {field}")
                return False
        
        return True
    
    def enrich_metadata(self, metadata: Dict[str, Any], content: str) -> Dict[str, Any]:
        """Add auto-generated metadata fields."""
        if "document_id" not in metadata:
            metadata["document_id"] = str(uuid4())
        
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.utcnow().isoformat()
        
        if "content_hash" not in metadata:
            metadata["content_hash"] = self.compute_content_hash(content)
        
        if "version" not in metadata:
            metadata["version"] = 1
        
        if "language" not in metadata:
            metadata["language"] = "en"
        
        if "tags" not in metadata:
            metadata["tags"] = []
        
        metadata["superseded"] = False
        
        return metadata


class IngestionPipeline:
    """Complete document ingestion pipeline."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
    
    async def ingest_document(
        self,
        request: IngestRequest,
    ) -> Dict[str, Any]:
        """Ingest a single document."""
        try:
            # Create metadata
            metadata = {
                "source": request.source,
                "title": request.title,
                "author": request.author,
                "category": request.category,
                "tags": request.tags,
                "language": request.language,
                **(request.metadata or {}),
            }
            
            # Enrich metadata
            metadata = self.processor.enrich_metadata(metadata, request.content)
            document_id = metadata["document_id"]
            
            # Check for duplicates
            if settings.enable_deduplication:
                existing_doc = await self._check_duplicate(metadata["content_hash"], request.source)
                if existing_doc:
                    return existing_doc
            
            # Preprocess content
            processed_content = self.processor.preprocess_text(request.content)
            
            # Chunk document
            chunks = self.processor.chunk_text(processed_content, metadata)
            
            if not chunks:
                raise ValueError("No valid chunks created from document")
            
            # Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_model.embed_texts(chunk_texts, show_progress=False)
            
            # Prepare for vector store
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Store in vector database
            self.vector_store.add_documents(
                ids=chunk_ids,
                texts=chunk_texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            
            logger.info(f"Successfully ingested document {document_id} with {len(chunks)} chunks")
            
            return {
                "document_id": document_id,
                "status": "success",
                "chunks_created": len(chunks),
                "version": metadata["version"],
                "message": f"Document ingested successfully with {len(chunks)} chunks",
            }
        
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise
    
    async def _check_duplicate(
        self,
        content_hash: str,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if document already exists."""
        # Search by metadata filter
        try:
            docs, _ = self.vector_store.search(
                query_embedding=[0.0] * settings.vector_size,  # Dummy embedding
                top_k=1,
                filters={"content_hash": content_hash},
            )
            
            if docs:
                logger.info(f"Duplicate document found with hash {content_hash}")
                return {
                    "document_id": docs[0]["document_id"],
                    "status": "duplicate",
                    "chunks_created": 0,
                    "version": docs[0].get("version", 1),
                    "message": "Document already exists (duplicate content hash)",
                }
        except Exception as e:
            logger.warning(f"Error checking for duplicates: {e}")
        
        return None
    
    async def batch_ingest(
        self,
        requests: List[IngestRequest],
    ) -> List[Dict[str, Any]]:
        """Ingest multiple documents."""
        results = []
        
        for request in requests:
            try:
                result = await self.ingest_document(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch ingestion: {e}")
                results.append({
                    "status": "error",
                    "message": str(e),
                })
        
        return results
    
    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update an existing document."""
        try:
            # Delete old chunks
            self.vector_store.delete_by_filter({"document_id": document_id})
            
            # Re-ingest with updated content/metadata
            # This is simplified - in production, you'd fetch the old document first
            if content:
                request = IngestRequest(
                    title=metadata_updates.get("title", "Updated Document"),
                    content=content,
                    source=metadata_updates.get("source", f"doc://{document_id}"),
                    **(metadata_updates or {}),
                )
                return await self.ingest_document(request)
            
            return {
                "document_id": document_id,
                "status": "success",
                "message": "Document updated successfully",
            }
        
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            raise
    
    async def delete_document(self, document_id: str, soft_delete: bool = True) -> Dict[str, Any]:
        """Delete a document."""
        try:
            if soft_delete:
                # Mark as superseded instead of deleting
                # In production, you'd update the metadata
                logger.info(f"Soft deleting document {document_id}")
            else:
                # Hard delete
                self.vector_store.delete_by_filter({"document_id": document_id})
                logger.info(f"Hard deleted document {document_id}")
            
            return {
                "document_id": document_id,
                "status": "deleted",
                "message": f"Document {'soft' if soft_delete else 'hard'} deleted successfully",
            }
        
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise


# Global pipeline instance
_ingestion_pipeline: Optional[IngestionPipeline] = None


def get_ingestion_pipeline() -> IngestionPipeline:
    """Get or create the global ingestion pipeline instance."""
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline()
    return _ingestion_pipeline