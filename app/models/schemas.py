from pydantic import BaseModel
from typing import List, Optional

class Source(BaseModel):
    id: int
    type: str           # "document" | "web"
    title: str
    subtitle: str
    url: Optional[str]
    excerpt: Optional[str]

class ResearchResponse(BaseModel):
    status: str
    answer: str
    sources: List[Source]
    session_id: str
    language: str

class UploadResponse(BaseModel):
    status: str
    session_id: str
    doc_id: str
    summary: Optional[str]
    answer: Optional[str]