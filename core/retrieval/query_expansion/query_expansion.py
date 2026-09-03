"""
Simple Vietnamese text preprocessing module.
Provides clean, normalized, tokenized, and POS-tagged Vietnamese text processing.
"""

import logging
import re
from typing import List, Tuple, Optional

try:
    import underthesea

    UNDERTHE_SEA_AVAILABLE = True
except ImportError:
    UNDERTHE_SEA_AVAILABLE = False

# Local imports
from config.settings import Config

logger = logging.getLogger(__name__)


class VietnamesePreprocessor:
    """Simple Vietnamese text preprocessing with cleaning, tokenization, and POS tagging."""

    def __init__(self):
        # Load configuration
        self.enabled = Config.RAG.VIETNAMESE_PREPROCESSING_ENABLED()
        self.clean_special_chars = Config.RAG.CLEAN_SPECIAL_CHARS()
        self.extract_content_words = (
            Config.RAG.EXTRACT_CONTENT_WORDS()
        )

        if not self.enabled:
            logger.info("ℹ️ Vietnamese preprocessing is disabled")
            return

        if UNDERTHE_SEA_AVAILABLE:
            logger.info(" underthesea available for Vietnamese preprocessing")
        else:
            logger.warning(
                " underthesea not available - Vietnamese preprocessing will be limited"
            )

    def preprocess(self, text: str) -> dict:
        """
        Complete Vietnamese text preprocessing pipeline.

        Args:
            text: Input Vietnamese text

        Returns:
            Dictionary containing:
            - original: Original text
            - cleaned: Cleaned and normalized text
            - tokens: List of tokens
            - pos_tags: List of (token, pos_tag) tuples
            - content_words: List of content words (nouns, verbs, adjectives)
        """
        if not self.enabled:
            return {
                "original": text,
                "cleaned": text.lower().strip() if text else "",
                "tokens": text.lower().split() if text else [],
                "pos_tags": [],
                "content_words": [],
            }

        if not text or not text.strip():
            return {
                "original": text,
                "cleaned": "",
                "tokens": [],
                "pos_tags": [],
                "content_words": [],
            }

        try:
            # 1. Clean and normalize input
            cleaned = (
                self._clean_text(text)
                if self.clean_special_chars
                else text.lower().strip()
            )

            # 2. Tokenize Vietnamese text
            tokens = self._tokenize(cleaned)

            # 3. Use Vietnamese POS tagger
            pos_tags = self._pos_tag(cleaned)

            # Extract content words
            content_words = (
                self._extract_content_words(pos_tags)
                if self.extract_content_words
                else []
            )

            return {
                "original": text,
                "cleaned": cleaned,
                "tokens": tokens,
                "pos_tags": pos_tags,
                "content_words": content_words,
            }

        except Exception as e:
            logger.warning(f" Vietnamese preprocessing failed: {e}")
            return {
                "original": text,
                "cleaned": text.lower().strip(),
                "tokens": text.lower().split(),
                "pos_tags": [],
                "content_words": [],
            }

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize Vietnamese text.
        - Convert to lowercase
        - Remove special characters (keep Vietnamese diacritics)
        - Normalize whitespace using TextUtils
        """
        from utils.text_processing.text_utils import TextUtils

        if not text:
            return ""
        # Convert to lowercase and clean with TextUtils
        cleaned = TextUtils.clean_text(text.lower())
        # Remove special characters but keep Vietnamese diacritics
        cleaned = re.sub(
            r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
            " ",
            cleaned,
        )
        # Normalize whitespace with TextUtils
        cleaned = TextUtils.normalize_whitespace(cleaned)
        return cleaned.strip()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text using underthesea.
        Falls back to simple word splitting if underthesea unavailable.
        """
        if not text:
            return []

        if UNDERTHE_SEA_AVAILABLE:
            try:
                tokens = underthesea.word_tokenize(text)
                return [str(token) for token in tokens]
            except Exception as e:
                logger.warning(f" underthesea tokenization failed: {e}")

        # Fallback: simple word splitting
        return text.split()

    def _pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """
        POS tag Vietnamese text using underthesea.
        Returns list of (token, pos_tag) tuples.
        """
        if not text:
            return []

        if UNDERTHE_SEA_AVAILABLE:
            try:
                pos_tags = underthesea.pos_tag(text)
                return [(str(token), str(pos)) for token, pos in pos_tags]
            except Exception as e:
                logger.warning(f" underthesea POS tagging failed: {e}")

        # Fallback: no POS tagging
        return []

    def _extract_content_words(self, pos_tags: List[Tuple[str, str]]) -> List[str]:
        """
        Extract content words (nouns, verbs, adjectives) from POS tags.
        """
        content_words = []

        for token, pos in pos_tags:
            # Focus on content words: nouns (N), verbs (V), adjectives (A)
            if pos in ("N", "V", "A", "Np", "Nc", "Nu", "Ny", "Nv") and len(token) > 2:
                content_words.append(token)

        return content_words

    def get_tokens(self, text: str) -> List[str]:
        """Get just the tokens from Vietnamese text."""
        result = self.preprocess(text)
        return result["tokens"]

    def get_content_words(self, text: str) -> List[str]:
        """Get just the content words from Vietnamese text."""
        result = self.preprocess(text)
        return result["content_words"]

    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Get just the POS tags from Vietnamese text."""
        result = self.preprocess(text)
        return result["pos_tags"]


# Example usage
if __name__ == "__main__":
    # Initialize the preprocessor
    preprocessor = VietnamesePreprocessor()

    # Example Vietnamese text
    vietnamese_text = "Làm thế nào để tải tài liệu từ hệ thống?"

    # Full preprocessing
    result = preprocessor.preprocess(vietnamese_text)

    print("=== Vietnamese Text Preprocessing ===")
    print(f"Original: {result['original']}")
    print(f"Cleaned: {result['cleaned']}")
    print(f"Tokens: {result['tokens']}")
    print(f"POS Tags: {result['pos_tags']}")
    print(f"Content Words: {result['content_words']}")

    # Individual methods
    print("\n=== Individual Methods ===")
    print(f"Tokens only: {preprocessor.get_tokens(vietnamese_text)}")
    print(f"Content words only: {preprocessor.get_content_words(vietnamese_text)}")
    print(f"POS tags only: {preprocessor.get_pos_tags(vietnamese_text)}")
