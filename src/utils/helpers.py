# src/utils/helpers.py
import logging
import time
import functools
import hashlib
import json
from typing import Any, Callable
from datetime import datetime

class JSONLogFormatter(logging.Formatter):
    """Format logs as JSON for production observability."""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id
        return json.dumps(log_obj)

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configures a structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    logger.addHandler(handler)
    return logger

def time_execution(logger: logging.Logger = None):
    """Decorator to measure function execution time."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = (time.time() - start) * 1000
                msg = f"{func.__name__} executed in {duration:.2f}ms"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
        return wrapper
    return decorator

def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash for content deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()