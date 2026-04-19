from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ResearchQuery(BaseModel):
    query: str
    mode: str = "hybrid"

@router.post("/query", summary="Deep Research query",
             description="Ask a question using RAG + web search")
async def research_query(body: ResearchQuery):
    return {
        "query": body.query,
        "mode": body.mode,
        "status": "coming soon"
    }