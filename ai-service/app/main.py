from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.exceptions import AppException
from app.core.exception_handlers import (
    app_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)

from app.api.routes.query import router as query_router
from app.api.routes.store import router as store_router
from app.api.routes.search import router as search_router
from app.api.routes.health import router as health_router

from app.services.embedding_service import EmbeddingService
from app.services.rerank_service import RerankService
from app.services.generation_service import GenerationService
from app.services.retrieval_service import RetrievalService
from app.storage.faiss_store import FaissStore
from app.pipelines.rag_pipeline import RagPipeline

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(title=settings.app_name)

# Middleware
app.add_middleware(RequestLoggingMiddleware)

# Exception handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Services
embedding_service = EmbeddingService(settings.embedding_model)
rerank_service = RerankService(settings.rerank_model)
generation_service = GenerationService(settings.ollama_url, settings.ollama_model)
faiss_store = FaissStore(settings.faiss_dimension)

retrieval_service = RetrievalService(embedding_service, faiss_store)
rag_pipeline = RagPipeline(retrieval_service, rerank_service, generation_service)

# Routes
app.include_router(query_router)
app.include_router(store_router)
app.include_router(search_router)
app.include_router(health_router)