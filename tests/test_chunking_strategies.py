# tests/test_chunking_strategies.py

import pytest
import os
from app.services.chunking_service import create_chunks
from app.services.document_service import process_document

SAMPLE_PATH = "tests/samples/AI_Deep_Research_Module_Architecture.pdf"


class TestChunkingBasics:
    def test_empty_string_returns_empty_list(self):
        assert create_chunks("", {"file_name": "test.pdf"}) == []
        assert create_chunks("   \n\n  ", {"file_name": "test.pdf"}) == []

    def test_single_short_text_single_chunk(self):
        result = create_chunks("Short text.", {"file_name": "test.pdf"}, chunk_size=1000)
        assert len(result) == 1
        assert result[0]["content"] == "Short text."

    def test_no_empty_chunks_produced(self):
        text = "Word. " * 50
        result = create_chunks(text, {"file_name": "test.pdf"}, chunk_size=20, chunk_overlap=5)
        for i, chunk in enumerate(result):
            assert chunk["content"].strip() != "", f"Chunk {i} is empty"


class TestChunkingQuality:
    def test_chunks_preserve_sentence_boundaries(self):
        text = "First sentence here. Second sentence there. Third sentence everywhere."
        result = create_chunks(text, {"file_name": "test.pdf"}, chunk_size=80, chunk_overlap=10)
        full_text = " ".join(c["content"] for c in result)
        assert "First sentence here" in full_text

    def test_no_micro_chunks(self):
        """Chunks should be meaningful — not just punctuation."""
        text = "This is a longer piece of text that will definitely need to be split into multiple chunks"
        result = create_chunks(text, {"file_name": "test.pdf"}, chunk_size=60, chunk_overlap=15)
        
        for i, chunk in enumerate(result):
            content = chunk["content"].strip()
            assert len(content) >= 3, f"Chunk {i} too short: {repr(content)}"
            assert content not in {".", "!", "?", ",", " "}, f"Chunk {i} is just punctuation: {repr(content)}"

    def test_overlap_is_real_shared_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        result = create_chunks(text, {"file_name": "test.pdf"}, chunk_size=80, chunk_overlap=20)
        if len(result) > 1:
            overlap_found = False
            for i in range(len(result) - 1):
                a, b = result[i]["content"], result[i + 1]["content"]
                for j in range(len(a) - 5):
                    substr = a[j:j+8].strip()
                    if len(substr) > 5 and substr in b:
                        overlap_found = True
                        break
                if overlap_found:
                    break
            assert overlap_found


class TestChunkingMetadata:
    def test_required_metadata_fields(self):
        meta = {"file_name": "doc.pdf", "doc_id": "abc", "session_id": "s1", "language": "en"}
        result = create_chunks("Some text.", meta)
        required = {"file_name", "doc_id", "session_id", "language", 
                   "chunk_index", "chunk_hash", "page_number", "created_at"}
        for chunk in result:
            assert required.issubset(chunk["metadata"].keys())

    def test_doc_id_is_consistent_across_chunks(self):
        text = "Chunk one. Chunk two. Chunk three."
        result = create_chunks(text, {"file_name": "t.pdf", "doc_id": "D1"}, chunk_size=20, chunk_overlap=5)
        assert len(result) > 1
        assert all(c["metadata"]["doc_id"] == "D1" for c in result)

    def test_chunk_index_is_sequential(self):
        result = create_chunks("A. B. C. D.", {"file_name": "t.pdf"}, chunk_size=10, chunk_overlap=2)
        assert [c["metadata"]["chunk_index"] for c in result] == list(range(len(result)))

    def test_chunk_hash_is_unique_and_stable(self):
        result = create_chunks("Hash test.", {"file_name": "t.pdf"}, chunk_size=15, chunk_overlap=3)
        hashes = [c["metadata"]["chunk_hash"] for c in result]
        assert len(hashes) == len(set(hashes))
        result2 = create_chunks("Hash test.", {"file_name": "t.pdf"}, chunk_size=15, chunk_overlap=3)
        assert hashes == [c["metadata"]["chunk_hash"] for c in result2]

    def test_page_number_from_content_marker(self):
        result = create_chunks("Text. [Page 5] More.", {"file_name": "t.pdf"}, chunk_size=100, chunk_overlap=20)
        page_chunks = [c for c in result if "[Page 5]" in c["content"]]
        if page_chunks:
            assert page_chunks[0]["metadata"]["page_number"] == 5

    def test_page_number_fallback_to_metadata(self):
        result = create_chunks("No markers.", {"file_name": "t.pdf", "page_number": 3}, chunk_size=100, chunk_overlap=20)
        assert result[0]["metadata"]["page_number"] == 3

    def test_page_number_defaults_to_index_plus_one(self):
        result = create_chunks("No markers.", {"file_name": "t.pdf"}, chunk_size=100, chunk_overlap=20)
        for i, chunk in enumerate(result):
            assert chunk["metadata"]["page_number"] == i + 1


class TestChunkingDeterminism:
    def test_chunking_is_deterministic(self):
        meta = {"file_name": "t.pdf", "doc_id": "d1"}
        r1 = create_chunks("Determinism test.", meta, chunk_size=20, chunk_overlap=5)
        r2 = create_chunks("Determinism test.", meta, chunk_size=20, chunk_overlap=5)
        assert len(r1) == len(r2)
        for c1, c2 in zip(r1, r2):
            assert c1["content"] == c2["content"]
            assert c1["metadata"]["chunk_hash"] == c2["metadata"]["chunk_hash"]


@pytest.mark.skipif(not os.path.exists(SAMPLE_PATH), reason="Sample not found")
class TestPipelineChunking:
    def test_pipeline_creates_non_empty_chunks(self):
        result = process_document(file_path=SAMPLE_PATH, mode="full", index=True)
        assert result["status"] == "success"
        assert result["extraction"]["chunks_created"] > 0
        assert result["extraction"]["indexed"] is True

    def test_pipeline_chunk_count_is_reasonable(self):
        result = process_document(file_path=SAMPLE_PATH, mode="full", index=True)
        count = result["extraction"]["chunks_created"]
        assert 2 < count < 10000

    def test_pipeline_processing_time_is_reasonable(self):
        result = process_document(file_path=SAMPLE_PATH, mode="full", index=True)
        assert result["processing_time_ms"] < 30000

    def test_pipeline_doc_id_preserved(self):
        result = process_document(file_path=SAMPLE_PATH, mode="full", index=True, doc_id="TEST-123")
        assert result["doc_id"] == "TEST-123"

    def test_pipeline_brief_vs_full(self):
        brief = process_document(file_path=SAMPLE_PATH, mode="brief", index=True)
        full = process_document(file_path=SAMPLE_PATH, mode="full", index=True)
        assert brief["extraction"]["chunks_created"] <= full["extraction"]["chunks_created"]