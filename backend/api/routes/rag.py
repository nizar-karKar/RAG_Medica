from fastapi import APIRouter
from api.schemas.request import QueryRequest
from api.schemas.response import QueryResponse
from api.dependencies.rag_pipeline import rag_pipeline

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = rag_pipeline.run(request.query)
    return result