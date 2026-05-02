from fastapi import APIRouter, Depends

from app.models.request import StoreRequest
from app.api.dependencies import get_embedding_service, get_faiss_store

router = APIRouter()


@router.post("/store")
def store(
    req: StoreRequest,
    embedding_service=Depends(get_embedding_service),
    faiss_store=Depends(get_faiss_store),
):
    if not req.texts:
        return {"message": "No texts provided"}

    vectors = embedding_service.embed_documents(req.texts)
    faiss_store.add_documents(req.texts, vectors)

    return {
        "message": "stored successfully",
        "count": len(faiss_store.documents),
    }