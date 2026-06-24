# tests/test_integration.py
"""
Full pipeline integration test.
Requires: Qdrant running on localhost:6333
Run with: python -m pytest tests/test_integration.py -v -s
"""

import pytest
from app.services.document_service import ingest_document
from app.services.vector_store import vector_store


SAMPLE_FR = """
L'intelligence artificielle transforme profondément le monde du travail.
Les modèles de langage peuvent désormais traiter des documents complexes,
répondre à des questions et rédiger des rapports complets.

Dans ce projet, nous intégrons un module de traitement documentaire
et un agent de recherche dans un chatbot d'entreprise.
L'agent utilise LangGraph pour décider s'il consulte les documents
locaux via RAG ou effectue une recherche web via Tavily.
"""

SAMPLE_EN = """
Artificial intelligence is transforming the way companies process documents.
Large language models can now extract text, answer questions,
and generate structured reports from unstructured data.

This project integrates a document intelligence module and a deep research
agent into an existing company chatbot using FastAPI and LangChain.
"""

FILE_META = {
    "file_name":    "test_doc.pdf",
    "file_type":    "pdf",
    "file_size_kb": 12.4,
    "mode":         "brief",
}


class TestIntegration:

    def test_ingest_returns_stats(self):
        """ingest_document must return valid stats dict."""
        result = ingest_document(SAMPLE_FR, FILE_META)
        assert result["chunks_indexed"] > 0
        assert result["language"] in {"fr", "en", "ar", "unknown"}
        assert result["word_count"] > 0
        assert 0.0 <= result["reduction_pct"] <= 100.0

    def test_language_detected_french(self):
        """French text must be detected as 'fr'."""
        result = ingest_document(SAMPLE_FR, FILE_META)
        assert result["language"] == "fr"

    def test_language_detected_english(self):
        """English text must be detected as 'en'."""
        meta = {**FILE_META, "file_name": "test_en.pdf"}
        result = ingest_document(SAMPLE_EN, meta)
        assert result["language"] == "en"

    def test_search_returns_results(self):
        """After ingestion, search must return relevant chunks."""
        ingest_document(SAMPLE_FR, FILE_META)
        results = vector_store.search("intelligence artificielle", k=3)
        assert len(results) > 0
        assert len(results[0].page_content) > 0

    def test_metadata_survives_qdrant_roundtrip(self):
        """file_name and language must be preserved through Qdrant."""
        ingest_document(SAMPLE_FR, FILE_META)
        results = vector_store.search("traitement documentaire", k=3)
        meta = results[0].metadata
        assert meta["file_name"] == "test_doc.pdf"
        assert meta["language"] == "fr"
        assert "chunk_index" in meta
        assert "chunk_total" in meta

    def test_collection_count_increases(self):
        """Vector count must increase after ingestion."""
        before = vector_store.collection_count()
        ingest_document(SAMPLE_EN, {**FILE_META, "file_name": "count_test.pdf"})
        after = vector_store.collection_count()
        assert after > before

    def test_empty_text_indexes_nothing(self):
        """Empty text must produce 0 chunks and not crash."""
        result = ingest_document("", FILE_META)
        assert result["chunks_indexed"] == 0

    def test_search_with_real_sample_file(self):
        """Real document test — uses your existing PDF sample."""
        import os
        path = "tests/samples/AI_Deep_Research_Module_Architecture.pdf"
        if not os.path.exists(path):
            pytest.skip("Sample file not found")

        import fitz
        doc = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
        doc.close()

        result = ingest_document(text, {
            "file_name":    "AI_Deep_Research_Module_Architecture.pdf",
            "file_type":    "pdf",
            "file_size_kb": os.path.getsize(path) / 1024,
            "mode":         "full",
        })

        assert result["chunks_indexed"] > 0
        results = vector_store.search("LangGraph agent research", k=3)
        assert len(results) > 0