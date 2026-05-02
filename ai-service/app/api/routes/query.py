from fastapi import APIRouter
from app.models.request import AskRequest
from app.models.response import AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    from app.main import rag_pipeline
    return rag_pipeline.ask(req.query)