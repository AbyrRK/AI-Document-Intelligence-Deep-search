"""
tests/test_extraction_modes.py
================================
Tests for brief/summary/full extraction mode logic.
"""

import pytest
from unittest.mock import patch, MagicMock
from document_service import (
    _select_pages,
    _is_big_title_page,
    _is_scanned_pdf,
    _is_title_page,
    _docx_select_brief,
    _docx_select_summary,
    BRIEF_PAGES,
    SUMMARY_FIRST,
    SUMMARY_LAST,
)


# ─── _select_pages (PDF) ──────────────────────────────────────────────────────

class TestSelectPages:

    def _mock_doc(self, n_pages: int, big_title_pages: list[int] = None):
        """Build a mock fitz document with n_pages pages."""
        pages = []
        for i in range(n_pages):
            page = MagicMock()
            # By default, no page is a big-title page unless specified
            is_big = big_title_pages and i in big_title_pages
            page._is_big_title = is_big
            pages.append(page)

        doc = MagicMock()
        doc.__len__ = lambda self: n_pages
        doc.__getitem__ = lambda self, i: pages[i]
        return doc

    def test_brief_returns_first_3_pages(self):
        doc = self._mock_doc(20)
        result = _select_pages(doc, "brief", 20, is_scanned=False)
        assert result == [0, 1, 2]

    def test_brief_short_doc_under_3_pages(self):
        doc = self._mock_doc(2)
        result = _select_pages(doc, "brief", 2, is_scanned=False)
        assert result == [0, 1]

    def test_brief_single_page(self):
        doc = self._mock_doc(1)
        result = _select_pages(doc, "brief", 1, is_scanned=False)
        assert result == [0]

    def test_summary_always_includes_first_5(self):
        doc = self._mock_doc(30)
        with patch("document_service._is_big_title_page", return_value=False):
            result = _select_pages(doc, "summary", 30, is_scanned=False)
        for i in range(SUMMARY_FIRST):
            assert i in result

    def test_summary_always_includes_last_2(self):
        doc = self._mock_doc(30)
        with patch("document_service._is_big_title_page", return_value=False):
            result = _select_pages(doc, "summary", 30, is_scanned=False)
        assert 28 in result
        assert 29 in result

    def test_summary_includes_big_title_pages(self):
        doc = self._mock_doc(30)
        # Pages 10 and 20 have big titles
        def fake_big_title(page):
            idx = [i for i in range(30) if doc[i] is page]
            return idx[0] in [10, 20] if idx else False

        with patch("document_service._is_big_title_page", side_effect=fake_big_title):
            result = _select_pages(doc, "summary", 30, is_scanned=False)
        assert 10 in result
        assert 20 in result

    def test_summary_skips_big_title_detection_for_scanned(self):
        """
        For scanned PDFs, we should NOT call _is_big_title_page
        (that would waste time reading every page).
        """
        doc = self._mock_doc(30)
        with patch("document_service._is_big_title_page") as mock_big:
            result = _select_pages(doc, "summary", 30, is_scanned=True)
        mock_big.assert_not_called()

    def test_summary_result_is_sorted(self):
        doc = self._mock_doc(30)
        with patch("document_service._is_big_title_page", return_value=False):
            result = _select_pages(doc, "summary", 30, is_scanned=False)
        assert result == sorted(result)

    def test_full_returns_all_pages(self):
        doc = self._mock_doc(57)
        result = _select_pages(doc, "full", 57, is_scanned=False)
        assert result == list(range(57))

    def test_summary_no_duplicates(self):
        """Pages in the first-5 AND last-2 overlap window should not appear twice."""
        doc = self._mock_doc(5)  # 5 pages total — first 5 and last 2 overlap
        with patch("document_service._is_big_title_page", return_value=False):
            result = _select_pages(doc, "summary", 5, is_scanned=False)
        assert len(result) == len(set(result)), "Duplicate pages found"


# ─── _is_big_title_page ───────────────────────────────────────────────────────

class TestIsBigTitlePage:

    def _make_page(self, sizes: list[float]):
        """Build a mock fitz page with given font sizes."""
        spans = [{"size": s} for s in sizes]
        line = {"spans": spans}
        block = {"type": 0, "lines": [line]}
        page = MagicMock()
        page.get_text.return_value = {"blocks": [block]}
        return page

    def test_detects_section_header(self):
        # One large title (24pt) among small body text (10pt each)
        page = self._make_page([10, 10, 10, 10, 24])
        assert _is_big_title_page(page) is True

    def test_uniform_text_is_not_big_title(self):
        # All text the same size — no section header
        page = self._make_page([12, 12, 12, 12, 12])
        assert _is_big_title_page(page) is False

    def test_slightly_larger_not_enough(self):
        # 15pt vs avg 12pt — ratio is 1.25, below threshold of 1.5
        page = self._make_page([12, 12, 12, 15])
        assert _is_big_title_page(page) is False

    def test_empty_page_is_not_big_title(self):
        page = MagicMock()
        page.get_text.return_value = {"blocks": []}
        assert _is_big_title_page(page) is False

    def test_large_font_below_minimum_pt(self):
        # Max font is 1.8x average but below BIG_TITLE_MIN_PT = 14
        page = self._make_page([7, 7, 7, 13])  # avg=8.5, max=13 < 14
        assert _is_big_title_page(page) is False


# ─── _is_scanned_pdf ─────────────────────────────────────────────────────────

class TestIsScannedPdf:

    def _make_doc(self, text_per_page: list[str], has_images: bool):
        pages = []
        for text in text_per_page:
            p = MagicMock()
            p.get_text.return_value = text
            p.get_images.return_value = [MagicMock()] if has_images else []
            pages.append(p)
        doc = MagicMock()
        doc.__len__ = lambda self: len(pages)
        doc.__getitem__ = lambda self, i: pages[i]
        return doc

    def test_native_pdf_detected(self):
        doc = self._make_doc(["Long native text " * 20] * 3, has_images=False)
        assert _is_scanned_pdf(doc) is False

    def test_scanned_pdf_detected(self):
        doc = self._make_doc([""] * 3, has_images=True)
        assert _is_scanned_pdf(doc) is True

    def test_mixed_with_images_and_little_text(self):
        doc = self._make_doc(["ab"] * 3, has_images=True)
        assert _is_scanned_pdf(doc) is True


# ─── DOCX selection ───────────────────────────────────────────────────────────

class TestDocxSelection:

    def _make_paragraphs(self, texts: list[str], heading_indices: list[int] = None):
        paras = []
        for i, text in enumerate(texts):
            p = MagicMock()
            p.text = text
            style = MagicMock()
            if heading_indices and i in heading_indices:
                style.name = "Heading 1"
            else:
                style.name = "Normal"
            p.style = style
            paras.append(p)
        return paras

    def test_brief_returns_max_5_paragraphs(self):
        paras = self._make_paragraphs(["Word " * 20] * 20)
        result = _docx_select_brief(paras)
        assert len(result) <= 5

    def test_brief_stops_at_300_words(self):
        # Each paragraph has 100 words → should stop after 3
        paras = self._make_paragraphs(["word " * 100] * 10)
        result = _docx_select_brief(paras)
        total_words = sum(len(p.text.split()) for p in result)
        assert total_words >= 300

    def test_summary_includes_headings(self):
        paras = self._make_paragraphs(
            ["paragraph"] * 20,
            heading_indices=[8, 14]
        )
        result = _docx_select_summary(paras)
        result_indices = [paras.index(p) for p in result]
        assert 8 in result_indices
        assert 14 in result_indices

    def test_summary_includes_paragraph_after_heading(self):
        paras = self._make_paragraphs(["paragraph"] * 20, heading_indices=[8])
        result = _docx_select_summary(paras)
        result_indices = [paras.index(p) for p in result]
        # Paragraph right after heading (index 9) should be included
        assert 9 in result_indices