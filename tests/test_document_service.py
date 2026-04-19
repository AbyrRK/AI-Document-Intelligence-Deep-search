# tests/test_document_service.py

import pytest
from app.services.document_service import extract_text

def test_pdf_extraction():
    result = extract_text(
        "tests/samples/AI_Deep_Research_Module_Architecture.pdf"
    )
    assert result["text"] is not None
    assert len(result["text"]) > 0
    assert result["metadata"]["file_type"] == "pdf"
    assert result["metadata"]["page_count"] > 0
    print(f"\n✅ PDF extracted : {result['metadata']['char_count']} characters")
    print(f"📄 Pages : {result['metadata']['page_count']}")
    print(f"Preview : {result['text'][:200]}")

def test_docx_extraction():
    result = extract_text("tests/samples/farcture.docx")
    assert result["text"] is not None
    assert len(result["text"]) > 0
    assert result["metadata"]["file_type"] == "docx"
    print(f"\n✅ DOCX extracted : {result['metadata']['char_count']} characters")
    print(f"📝 Paragraphs : {result['metadata']['paragraph_count']}")
    print(f"Preview : {result['text'][:200]}")

def test_xlsx_extraction():
    result = extract_text("tests/samples/Gantt_PFE_6mois.xlsx")
    assert result["text"] is not None
    assert len(result["text"]) > 0
    assert result["metadata"]["file_type"] == "xlsx"
    print(f"\n✅ XLSX extracted : {result['metadata']['char_count']} characters")
    print(f"📊 Sheets : {result['metadata']['sheet_count']}")
    print(f"Preview : {result['text'][:200]}")

def test_metadata_present():
    result = extract_text(
        "tests/samples/AI_Deep_Research_Module_Architecture.pdf"
    )
    assert "file_name" in result["metadata"]
    assert "file_size_kb" in result["metadata"]
    assert "char_count" in result["metadata"]
    assert "word_count" in result["metadata"]
    assert "extracted_at" in result["metadata"]
    print(f"\n✅ Metadata : {result['metadata']}")

def test_unsupported_format():
    with pytest.raises(ValueError):
        extract_text("tests/samples/sample.mp4")
    print(f"\n✅ Unsupported format correctly rejected")

def test_empty_file_rejected():
    # Create a temporary empty file for testing
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        temp_path = f.name
    try:
        with pytest.raises(ValueError):
            extract_text(temp_path)
        print(f"\n✅ Empty file correctly rejected")
    finally:
        os.unlink(temp_path)