# tests/test_ocr_service.py

import pytest
from app.services.ocr_service import extract_text_ocr, is_scanned_pdf

def test_ocr_unsupported_format():
    """Unsupported formats must be rejected cleanly."""
    with pytest.raises(ValueError):
        extract_text_ocr("tests/samples/sample.mp4")
    print("\n✅ Unsupported OCR format correctly rejected")

def test_is_scanned_pdf_native():
    """Native PDF should NOT be detected as scanned."""
    result = is_scanned_pdf(
        "tests/samples/AI_Deep_Research_Module_Architecture.pdf"
    )
    assert result == False
    print(f"\n✅ Native PDF correctly detected : is_scanned = {result}")

def test_native_pdf_uses_direct_extraction():
    """Native PDF should use direct extraction not OCR."""
    from app.services.document_service import extract_text
    result = extract_text(
        "tests/samples/AI_Deep_Research_Module_Architecture.pdf"
    )
    assert result["metadata"]["extraction_method"] == "native"
    print(f"\n✅ Native PDF used direct extraction")
    print(f"Method : {result['metadata']['extraction_method']}")
def test_ocr_real_image():
    """Test OCR on a real generated image with text."""
    result = extract_text_ocr("tests/samples/Capture.PNG")
    
    assert result["text"] is not None
    assert len(result["text"]) > 0
    assert result["metadata"]["file_type"] == "image"
    
    print(f"\n✅ OCR extracted : {result['metadata']['char_count']} characters")
    print(f"🌍 Languages used : {result['metadata']['languages']}")
    print(f"📝 Extracted text :\n{result['text']}")
def test_real_pdf_extraction():
    """Test full extraction on real PDF — shows complete result."""
    from app.services.document_service import extract_text

    result = extract_text("tests/samples/BP+CCTP LOT VRD.pdf")

    print(f"\n{'='*60}")
    print(f"📄 File        : {result['metadata']['file_name']}")
    print(f"🔍 Method      : {result['metadata']['extraction_method']}")
    print(f"📊 Characters  : {result['metadata']['char_count']}")
    print(f"📝 Words       : {result['metadata']['word_count']}")
    print(f"📑 Pages       : {result['metadata'].get('page_count', 'N/A')}")
    print(f"{'='*60}")
    print(f"📖 FULL EXTRACTED TEXT :\n")
    print(result['text'][:3000])  # show first 3000 chars
    print(f"\n{'='*60}")

    assert result["text"] is not None
    assert len(result["text"]) > 0

def test_handwriting_ocr():
    """
    Test smart OCR on a real handwritten document.
    Shows which engine was selected and the extracted text.
    """
    from app.services.ocr_service import ocr_smart
    from PIL import Image

    # Load your real handwritten image
    # Replace 'your_file.png' with your actual filename
    image = Image.open("tests/samples/handwritingpage.png")

    print(f"\n{'='*60}")
    print(f"🖊️  Testing handwriting OCR...")

    result = ocr_smart(image, use_gemini_fallback=True)

    print(f"🔍 Engine used  : {result['engine']}")
    print(f"📊 Confidence   : {result['confidence']:.1f}%")
    print(f"📝 Extracted text :\n{result['text']}")
    print(f"{'='*60}")

    assert result["text"] is not None
    assert len(result["text"]) > 0
    assert result["engine"] in ["tesseract", "gemini"]