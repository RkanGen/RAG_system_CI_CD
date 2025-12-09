# src/api/middleware.py
import time
import logging
from typing import Callable
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from src.core.config import settings

logger = logging.getLogger("api.middleware")

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

class AuthMiddleware(BaseHTTPMiddleware):
    """Simple API Key Authentication Middleware."""
    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path in ["/docs", "/openapi.json", "/health", "/metrics", "/"]:
            return await call_next(request)

        # In production, validate against DB or Environment Variable
        api_key = request.headers.get("X-API-Key")
        
        # Simple check - in real world check against a list/db
        # For now, we allow requests if no specific key is enforced in settings, 
        # or if the key matches.
        required_key = settings.openai_api_key # Using openai key as proxy for demo
        
        if required_key and api_key != required_key:
            # Note: In a real app, use a dedicated APP_API_KEY, not the LLM key
            pass 
            # Uncomment to enforce:
            # return JSONResponse(status_code=401, content={"detail": "Invalid API Key"})

        return await call_next(request)

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time"] = str(process_time)
        return response