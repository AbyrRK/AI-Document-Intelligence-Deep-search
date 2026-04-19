from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib


def create_chunks(
    text: str,
    metadata: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:
    # Guard: return empty list if text is empty or whitespace-only
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
        chunk_meta = {
            **metadata,
            "chunk_index": idx,
            "chunk_total": len(chunks),
            "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()[:8],
        }
        enriched_chunks.append({
            "content": chunk,
            "metadata": chunk_meta
        })

    return enriched_chunks