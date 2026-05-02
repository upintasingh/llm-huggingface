from fastapi import APIRouter, Depends

from app.models.request import SearchRequest
from app.api.dependencies import get_retrieval_service

router = APIRouter()


@router.post("/search")
def search(
    req: SearchRequest,
    retrieval_service=Depends(get_retrieval_service),
):
    results = retrieval_service.retrieve(req.query, k=3)
    return {"results": results}