# scripts/setup.py
import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.core.vector_store import get_vector_store
from src.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_db():
    """Initialize Vector DB Collections."""
    logger.info(f"Initializing connection to {settings.vector_db_backend}...")
    try:
        store = get_vector_store()
        # The __init__ of VectorStore handles creation, but we can force a check here
        info = store.get_collection_info()
        logger.info(f"Success! Collection '{info.get('name')}' is ready.")
        logger.info(f"Status: {info}")
    except Exception as e:
        logger.error(f"Failed to initialize DB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(init_db())