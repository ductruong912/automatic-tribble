"""
Unit tests for Vietnamese text processing, accent stripping, and tokenization.
"""

import unittest
from utils.text_processing.text_utils import TextUtils
from core.document_processing.pdf_classifier import diacritic_ratio
from core.retrieval.search.retriever import ContextRetriever


class TestVietnamesePreprocessing(unittest.TestCase):
    """Test suite for Vietnamese diacritics removal and tokenization."""

    def test_strip_vietnamese_accents(self):
        sample = "Cộng hòa Xã hội Chủ nghĩa Việt Nam - Độc lập Tự do Hạnh phúc"
        expected = "Cong hoa Xa hoi Chu nghia Viet Nam - Doc lap Tu do Hanh phuc"
        self.assertEqual(TextUtils.strip_vietnamese_accents(sample), expected)

    def test_strip_vietnamese_accents_special_characters(self):
        sample = "đường đi, ĐÀ NẴNG, quả bưởi, bánh mì, ốc bươu"
        stripped = TextUtils.strip_vietnamese_accents(sample)
        self.assertNotIn("đ", stripped)
        self.assertNotIn("Đ", stripped)
        self.assertIn("duong di", stripped)
        self.assertIn("DA NANG", stripped)
        self.assertIn("banh mi", stripped)

    def test_diacritic_ratio_vietnamese(self):
        vn_text = "Hợp đồng kinh tế và biên bản bàn giao thiết bị công nghệ thông tin"
        ratio = diacritic_ratio(vn_text)
        # Vietnamese prose typically exceeds 10% diacritics
        self.assertGreater(ratio, 0.10)

    def test_diacritic_ratio_english_or_empty(self):
        en_text = "This is a purely English document without any special accents or marks."
        self.assertEqual(diacritic_ratio(en_text), 0.0)
        self.assertEqual(diacritic_ratio(""), 0.0)
        self.assertEqual(diacritic_ratio("12345 !@#$%^&*()"), 0.0)

    def test_retriever_tokenize_vietnamese(self):
        retriever = ContextRetriever.__new__(ContextRetriever)
        retriever._query_preprocessing_cache = {}

        query = "Thủ tục đăng ký tạm trú tạm vắng gồm những gì?"
        tokens = retriever._tokenize(query)

        self.assertIsInstance(tokens, list)
        self.assertIn("thu", tokens)
        self.assertIn("tuc", tokens)
        self.assertIn("dang", tokens)
        self.assertIn("ky", tokens)
        self.assertNotIn("?", tokens)

        # Verify caching works
        self.assertIn(query, retriever._query_preprocessing_cache)
        cached_tokens = retriever._tokenize(query)
        self.assertEqual(tokens, cached_tokens)


if __name__ == "__main__":
    unittest.main()
