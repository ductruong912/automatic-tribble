"""
Unit tests for text chunking and cleaning utilities.
"""

import unittest
from utils.text_processing.text_utils import TextUtils


class TestChunking(unittest.TestCase):
    """Test suite for TextUtils chunking and manipulation routines."""

    def test_clean_text_basic(self):
        raw = "  Hello   world \r\n\r\n\r\n  This is a   test.  "
        cleaned = TextUtils.clean_text(raw)
        self.assertIn("Hello world", cleaned)
        self.assertIn("This is a test.", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_clean_text_empty(self):
        self.assertEqual(TextUtils.clean_text(""), "")
        self.assertEqual(TextUtils.clean_text(None), "")

    def test_extract_sentences(self):
        text = "This is the first sentence. Short! This is the third sentence that is long enough to pass."
        sentences = TextUtils.extract_sentences(text)
        # "Short" has len <= 10 and should be filtered out
        self.assertEqual(len(sentences), 2)
        self.assertIn("This is the first sentence", sentences[0])
        self.assertIn("This is the third sentence", sentences[1])

    def test_truncate_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        self.assertEqual(TextUtils.truncate_text(text, 10), "The qui...")
        self.assertEqual(TextUtils.truncate_text(text, 100), text)
        self.assertEqual(TextUtils.truncate_text("", 10), "")

    def test_clean_chunk_text_removes_table_dividers(self):
        chunk = "Header\n|---|---|---|\n| Column 1 | Column 2 |\n\nContent"
        cleaned = TextUtils.clean_chunk_text(chunk)
        self.assertNotIn("|---|---|---|", cleaned)
        self.assertIn("Header", cleaned)
        self.assertIn("Content", cleaned)

    def test_chunk_text_empty_or_small(self):
        self.assertEqual(TextUtils.chunk_text(""), [])
        self.assertEqual(TextUtils.chunk_text("   "), [])
        short = "Short document."
        self.assertEqual(TextUtils.chunk_text(short, chunk_size=100), [short])

    def test_chunk_text_long_sentences(self):
        sentences = [
            f"Sentence number {i} provides detailed information about RAG systems."
            for i in range(20)
        ]
        doc = " ".join(sentences)
        chunks = TextUtils.chunk_text(doc, chunk_size=200, overlap=50, language="en")
        
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)
            self.assertLessEqual(len(chunk), 300)  # Reasonable bounds around chunk_size

    def test_chunk_text_vietnamese(self):
        vn_doc = (
            "Hệ thống RAG cho phép truy xuất thông tin từ tài liệu nội bộ. "
            "Người dùng có thể đặt câu hỏi bằng tiếng Việt tự nhiên. "
            "Mô hình ngôn ngữ lớn sẽ tổng hợp câu trả lời dựa trên ngữ cảnh đã trích xuất. "
            "Độ tin cậy của câu trả lời được đánh giá tự động."
        )
        chunks = TextUtils.chunk_text(vn_doc, chunk_size=120, overlap=30, language="vi")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any("truy xuất thông tin" in c for c in chunks))


if __name__ == "__main__":
    unittest.main()
