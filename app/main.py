from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, research

app = FastAPI(
    title="AI Document Intelligence API",
    description="RAG + Deep Research backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(research.router, prefix="/research", tags=["Research"])

@app.get("/health")
def health():
    return {"status": "ok"}