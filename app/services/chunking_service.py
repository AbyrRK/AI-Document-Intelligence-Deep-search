"""
app/services/chunking_service.py
"""

import re
import hashlib
from typing import List, Dict
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    text: str,
    metadata: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    chunks = splitter.split_text(text)

    enriched_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()

        page_match = re.search(r"\[Page\s+(\d+)\]", chunk)
        page_number = int(page_match.group(1)) if page_match else metadata.get("page_number", idx + 1)

        chunk_meta = {
            "file_name":   metadata.get("file_name", ""),
            "doc_id":      metadata.get("doc_id", ""),      #  same value for all chunks of this doc
            "session_id":  metadata.get("session_id", ""),
            "language":    metadata.get("language", ""),
            "chunk_index": idx,
            "chunk_hash":  chunk_hash,
            "page_number": page_number,
            "created_at":  datetime.utcnow().isoformat(),
        }

        enriched_chunks.append({
            "content":  chunk,
            "metadata": chunk_meta
        })

    return enriched_chunks