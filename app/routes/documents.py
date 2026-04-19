"""
routes/documents.py
===================
POST /documents/upload  — upload & extract document
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.document_service import process_document

router = APIRouter()

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx",
    ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
}


@router.post("/upload")
async def upload_document(
    file:  UploadFile = File(...),
    mode:  str        = Form("brief"),
    index: bool       = Form(False),
):
    if mode not in ("brief", "summary", "full", "table"):
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Use: brief | summary | full | table")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: '{suffix}'")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = process_document(tmp_path, mode=mode, index=index)
        result["file_name"] = file.filename
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        Path(tmp_path).unlink(missing_ok=True)