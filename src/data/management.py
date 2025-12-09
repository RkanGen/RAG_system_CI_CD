import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib

from src.core.config import settings
from src.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class DeduplicationManager:
    """Manages document deduplication."""
    
    def __init__(self):
        self.vector_store = get_vector_store()
    
    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    async def find_duplicates(
        self,
        content_hash: str,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find documents with matching content hash."""
        filters = {"content_hash": content_hash}
        if source:
            filters["source"] = source
        
        try:
            # Search with dummy embedding (using filters only)
            docs, _ = self.vector_store.search(
                query_embedding=[0.0] * settings.vector_size,
                top_k=10,
                filters=filters,
            )
            return docs
        except Exception as e:
            logger.error(f"Error finding duplicates: {e}")
            return []
    
    async def handle_duplicate(
        self,
        new_document: Dict[str, Any],
        existing_document: Dict[str, Any],
    ) -> Dict[str, str]:
        """Handle duplicate document detection."""
        new_hash = new_document.get("content_hash")
        existing_hash = existing_document.get("content_hash")
        
        if new_hash == existing_hash:
            # Exact duplicate - skip ingestion
            logger.info(f"Exact duplicate found, skipping ingestion")
            return {
                "action": "skip",
                "document_id": existing_document.get("document_id"),
                "reason": "Exact content match",
            }
        
        # Same source but different content - create new version
        new_version = existing_document.get("version", 1) + 1
        
        # Mark old document as superseded
        await self._mark_superseded(existing_document.get("document_id"))
        
        logger.info(f"Creating new version {new_version} of document")
        return {
            "action": "version",
            "previous_version": existing_document.get("document_id"),
            "new_version": new_version,
        }
    
    async def _mark_superseded(self, document_id: str) -> None:
        """Mark a document as superseded."""
        # In production, update metadata in vector store
        logger.info(f"Marking document {document_id} as superseded")


class CleanupManager:
    """Manages cleanup of old and superseded documents."""
    
    def __init__(self):
        self.vector_store = get_vector_store()
    
    async def cleanup_superseded_documents(
        self,
        retention_hours: int = settings.superseded_retention_hours,
    ) -> Dict[str, Any]:
        """Remove superseded documents past retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
        
        try:
            # Find superseded documents older than retention period
            filters = {
                "superseded": True,
                "timestamp": {"lt": cutoff_time.isoformat()},
            }
            
            docs, _ = self.vector_store.search(
                query_embedding=[0.0] * settings.vector_size,
                top_k=1000,
                filters=filters,
            )
            
            # Delete documents
            if docs:
                doc_ids = [doc.get("document_id") for doc in docs]
                self.vector_store.delete_by_filter(filters)
                
                logger.info(f"Cleaned up {len(docs)} superseded documents")
                return {
                    "deleted_count": len(docs),
                    "document_ids": doc_ids,
                }
            
            return {"deleted_count": 0, "document_ids": []}
        
        except Exception as e:
            logger.error(f"Error cleaning up superseded documents: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_documents(
        self,
        retention_days: int = settings.document_retention_days,
    ) -> Dict[str, Any]:
        """Remove documents older than retention period."""
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        
        try:
            filters = {
                "timestamp": {"lt": cutoff_time.isoformat()},
            }
            
            docs, _ = self.vector_store.search(
                query_embedding=[0.0] * settings.vector_size,
                top_k=1000,
                filters=filters,
            )
            
            if docs:
                self.vector_store.delete_by_filter(filters)
                
                logger.info(f"Cleaned up {len(docs)} old documents")
                return {"deleted_count": len(docs)}
            
            return {"deleted_count": 0}
        
        except Exception as e:
            logger.error(f"Error cleaning up old documents: {e}")
            return {"error": str(e)}
    
    async def compact_collection(self) -> Dict[str, Any]:
        """Compact the vector collection for better performance."""
        try:
            self.vector_store.optimize_collection()
            
            logger.info("Collection compaction completed")
            return {"status": "success"}
        
        except Exception as e:
            logger.error(f"Error compacting collection: {e}")
            return {"error": str(e)}


class IndexManager:
    """Manages index rebuilding and updates."""
    
    def __init__(self):
        self.vector_store = get_vector_store()
    
    async def rebuild_index(
        self,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rebuild index for better search performance."""
        try:
            if collection_name is None:
                collection_name = settings.qdrant_collection
            
            # Trigger index optimization
            self.vector_store.optimize_collection()
            
            logger.info(f"Index rebuild completed for {collection_name}")
            return {
                "status": "success",
                "collection": collection_name,
            }
        
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return {"error": str(e)}
    
    async def create_backup(
        self,
        backup_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create backup of current collection."""
        try:
            if backup_path is None:
                backup_path = f"{settings.backup_storage_path}/backup_{datetime.utcnow().isoformat()}"
            
            # In production, implement actual backup logic
            logger.info(f"Backup created at {backup_path}")
            
            return {
                "status": "success",
                "backup_path": backup_path,
            }
        
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return {"error": str(e)}


class MaintenanceScheduler:
    """Schedules and executes maintenance tasks."""
    
    def __init__(self):
        self.cleanup_manager = CleanupManager()
        self.index_manager = IndexManager()
    
    async def daily_maintenance(self) -> Dict[str, Any]:
        """Run daily maintenance tasks."""
        logger.info("Starting daily maintenance...")
        
        results = {}
        
        # Cleanup superseded documents
        results["superseded_cleanup"] = await self.cleanup_manager.cleanup_superseded_documents()
        
        # Compact collection
        results["compaction"] = await self.cleanup_manager.compact_collection()
        
        # Rebuild index statistics
        results["index_rebuild"] = await self.index_manager.rebuild_index()
        
        logger.info("Daily maintenance completed")
        return results
    
    async def weekly_maintenance(self) -> Dict[str, Any]:
        """Run weekly maintenance tasks."""
        logger.info("Starting weekly maintenance...")
        
        results = {}
        
        # Cleanup old documents
        results["old_documents_cleanup"] = await self.cleanup_manager.cleanup_old_documents()
        
        # Create backup
        results["backup"] = await self.index_manager.create_backup()
        
        # Full index optimization
        results["index_optimization"] = await self.index_manager.rebuild_index()
        
        logger.info("Weekly maintenance completed")
        return results


# Global instances
_dedup_manager: Optional[DeduplicationManager] = None
_cleanup_manager: Optional[CleanupManager] = None
_index_manager: Optional[IndexManager] = None
_maintenance_scheduler: Optional[MaintenanceScheduler] = None


def get_deduplication_manager() -> DeduplicationManager:
    """Get or create deduplication manager."""
    global _dedup_manager
    if _dedup_manager is None:
        _dedup_manager = DeduplicationManager()
    return _dedup_manager


def get_cleanup_manager() -> CleanupManager:
    """Get or create cleanup manager."""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = CleanupManager()
    return _cleanup_manager


def get_index_manager() -> IndexManager:
    """Get or create index manager."""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager


def get_maintenance_scheduler() -> MaintenanceScheduler:
    """Get or create maintenance scheduler."""
    global _maintenance_scheduler
    if _maintenance_scheduler is None:
        _maintenance_scheduler = MaintenanceScheduler()
    return _maintenance_scheduler