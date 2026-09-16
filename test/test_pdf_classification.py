"""
Unit tests for PDF page classification (digital text vs scanned image).
"""

import unittest
import pymupdf
from core.document_processing.pdf_classifier import (
    PageKind,
    PageInfo,
    PdfProfile,
    classify_pdf,
    diacritic_ratio,
)


class TestPdfClassification(unittest.TestCase):
    """Test suite for PDF page classification and profile extraction."""

    def test_pdf_profile_properties(self):
        text_page = PageInfo(
            index=0,
            kind=PageKind.TEXT,
            text="Digital text content",
            char_count=20,
            image_count=0,
            image_coverage=0.0,
            diacritic_ratio=0.0,
        )
        scan_page = PageInfo(
            index=1,
            kind=PageKind.SCAN,
            text="",
            char_count=0,
            image_count=1,
            image_coverage=0.95,
            diacritic_ratio=0.0,
        )

        mixed_profile = PdfProfile(pages=[text_page, scan_page])
        self.assertEqual(mixed_profile.page_count, 2)
        self.assertEqual(len(mixed_profile.scan_pages), 1)
        self.assertFalse(mixed_profile.is_fully_text)
        self.assertFalse(mixed_profile.is_fully_scanned)
        self.assertIn("1 text, 1 scan", mixed_profile.summary())

        fully_text = PdfProfile(pages=[text_page])
        self.assertTrue(fully_text.is_fully_text)
        self.assertFalse(fully_text.is_fully_scanned)

        fully_scanned = PdfProfile(pages=[scan_page])
        self.assertFalse(fully_scanned.is_fully_text)
        self.assertTrue(fully_scanned.is_fully_scanned)

    def test_classify_pdf_synthetic(self):
        # Create an in-memory PDF with 2 pages:
        # Page 0: digital text with Vietnamese accents
        # Page 1: empty page (no text, should be classified as scan)
        doc = pymupdf.open()
        
        # Page 0: lots of text
        page0 = doc.new_page()
        vn_text = (
            "Cộng hòa Xã hội Chủ nghĩa Việt Nam. "
            "Đây là văn bản số một chứa đầy đủ nội dung số hóa với các dấu thanh điệu chuẩn tiếng Việt. "
            "Hệ thống sẽ nhận dạng trang này là trang văn bản kỹ thuật số (TEXT) mà không cần OCR. "
        ) * 5
        page0.insert_text((50, 72), vn_text)

        # Page 1: scanned image page (an image without sufficient text)
        page1 = doc.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), 0)
        page1.insert_image(pymupdf.Rect(50, 50, 150, 150), pixmap=pix)

        pdf_bytes = doc.tobytes()
        doc.close()

        profile = classify_pdf(pdf_bytes)

        self.assertEqual(profile.page_count, 2)
        self.assertEqual(profile.pages[0].kind, PageKind.TEXT)
        self.assertEqual(profile.pages[1].kind, PageKind.SCAN)
        self.assertFalse(profile.is_fully_text)
        self.assertEqual(len(profile.scan_pages), 1)
        self.assertEqual(profile.scan_pages[0].index, 1)


if __name__ == "__main__":
    unittest.main()
