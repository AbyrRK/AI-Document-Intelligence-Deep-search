"""
app/routes/chat.py
"""

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json

from app.services.rag_service import rag_service
from app.services.document_service import ensure_indexed
from app.services.chat_memory import delete_session

router = APIRouter()


@router.post("/ask")
async def ask_question(
    question:   str = Form(...),
    session_id: str = Form(...),
):
    """Non-streaming fallback."""
    try:
        was_indexed = ensure_indexed(session_id)
        if was_indexed:
            print(f"🔄 Lazy index triggered for session {session_id[:8]}")

        result = rag_service.ask(question, session_id=session_id)
        return JSONResponse(content={
            "answer":     result["answer"],
            "sources":    result["sources"],
            "session_id": session_id,
            "indexed_now": was_indexed,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
async def ask_stream(
    question:   str = Form(...),
    session_id: str = Form(...),
):
    """Streaming endpoint with lazy indexing support."""
    def generate():
        try:
            was_indexed = ensure_indexed(session_id)
            if was_indexed:
                yield f"data: {json.dumps({'info': 'Indexing document...'})}\n\n"

            for event in rag_service.ask_stream(question, session_id=session_id):
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'done': True, 'indexed_now': was_indexed})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/session/{session_id}")
async def clear_chat_session(session_id: str):
    delete_session(session_id)
    return JSONResponse(content={"message": "Session deleted"})