from fastapi import APIRouter, Depends
from datetime import datetime

from app.api.dependencies import (
    get_embedding_service,
    get_faiss_store,
)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def live():
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ready")
def ready(
    embedding_service=Depends(get_embedding_service),
    faiss_store=Depends(get_faiss_store),
):
    checks = {
        "embedding_model": embedding_service is not None,
        "faiss_index": faiss_store is not None,
    }

    ready_state = all(checks.values())

    return {
        "status": "ready" if ready_state else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }