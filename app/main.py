import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import documents, chat, research
from app.services.vector_store import vector_store
from app.services.pii_scrubber import warmup as pii_warmup


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Document AI Assistant")

    # Pre-load spaCy models so first upload has no cold-start delay
    pii_warmup()

    async def ttl_cleanup_loop():
        while True:
            await asyncio.sleep(3600)
            try:
                deleted = vector_store.cleanup_expired(max_age_hours=24)
                if deleted:
                    print(f"🧹 Scheduled cleanup: {deleted} expired chunks removed")
            except Exception as e:
                print(f"Cleanup error: {e}")

    cleanup_task = asyncio.create_task(ttl_cleanup_loop())
    yield
    cleanup_task.cancel()
    print("🛑 Shutting down")


app = FastAPI(
    title="AI Document Intelligence API",
    description="RAG + Deep Research backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(chat.router,      prefix="/chat",      tags=["Chat"])
app.include_router(research.router,  prefix="/research",  tags=["Research"])


@app.get("/health")
def health():
    return {"status": "ok"}