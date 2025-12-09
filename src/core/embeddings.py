import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wrapper for SentenceTransformers to handle text embedding."""
    
    def __init__(self):
        self.model_name = settings.embedding_model
        self.device = settings.embedding_device
        self.model = None
        self.dimension = settings.vector_size

    def warmup(self):
        """Load the model into memory."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Verify dimension
            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim != self.dimension:
                logger.warning(f"Configured vector size {self.dimension} does not match model dimension {actual_dim}")

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """Embed a list of texts."""
        if self.model is None:
            self.warmup()
        
        # SentenceTransformers returns numpy array, convert to list
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=show_progress, 
            convert_to_numpy=True,
            normalize_embeddings=True 
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        if self.model is None:
            self.warmup()
            
        embedding = self.model.encode(
            query, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()

# Global Singleton
_embedding_model: Optional[EmbeddingModel] = None

def get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model