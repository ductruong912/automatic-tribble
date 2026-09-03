"""
Text processing utilities for document content manipulation and analysis.
Provides text cleaning, keyword extraction, and quality assessment functions.
"""

# Standard library imports
import re
from typing import List, Optional


class TextUtils:
    """
    Utility functions for text processing, cleaning, and analysis.
    Provides comprehensive text manipulation capabilities for document processing.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text by removing excessive whitespace and normalizing line endings.
        Improves text quality for processing and storage.
        """
        if not text:
            return ""

        # Normalize line endings across different platforms
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive whitespace and normalize spacing
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Multiple newlines to double
        text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces/tabs to single space

        # Strip leading/trailing whitespace
        return text.strip()

    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """
        Extract sentences from text using simple punctuation-based splitting.
        Filters out very short fragments to improve quality.
        """
        if not text:
            return []

        # Simple sentence splitting (can be improved with NLTK if needed)
        sentences = re.split(r"[.!?]+", text)

        # Clean and filter sentences for quality
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to maximum length with optional suffix.
        Handles edge cases where suffix is longer than max_length.
        """
        if not text or len(text) <= max_length:
            return text

        if len(suffix) >= max_length:
            return text[:max_length]

        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def count_words(text: str) -> int:
        """
        Count words in text using regex pattern matching.
        Provides accurate word count for text analysis.
        """
        if not text:
            return 0

        # Simple word counting using regex
        words = re.findall(r"\b\w+\b", text)
        return len(words)

    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract potential keywords from text using frequency analysis.
        Filters out common stop words and short words for better quality.
        """
        if not text:
            return []

        # Convert to lowercase and extract words
        words = re.findall(r"\b\w{3,}\b", text.lower())

        # Filter out common stop words (basic list)
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "was",
            "were",
            "been",
            "have",
            "has",
            "had",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "must",
            "shall",
            "are",
            "is",
        }

        # Count word frequencies for keyword extraction
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]

    @staticmethod
    def is_meaningful_text(text: str, min_length: int = 20) -> bool:
        """
        Check if text contains meaningful content based on length and character ratio.
        Helps filter out low-quality or empty content.
        """
        if not text or len(text.strip()) < min_length:
            return False

        # Check if text has a reasonable ratio of letters to other characters
        letters = sum(1 for c in text if c.isalpha())
        total_chars = len(text)

        if total_chars == 0:
            return False

        letter_ratio = letters / total_chars
        return letter_ratio > 0.5  # At least 50% letters

    @staticmethod
    def is_probably_gibberish(text: str, threshold: float = 0.3) -> bool:
        """
        Determine if extracted text is likely gibberish and needs OCR processing.
        Analyzes Unicode character categories to detect low-quality text.
        """
        if not text:
            return True

        # Import unicodedata here to avoid dependency issues
        import unicodedata

        # Check for too many control/symbol characters
        bad_chars = sum(
            1 for c in text if unicodedata.category(c).startswith(("C", "S"))
        )
        return bad_chars / len(text) > threshold

    @staticmethod
    def needs_ocr_fallback(text: str) -> bool:
        """
        Determine if OCR fallback is needed for the extracted text.
        Consolidated from OCR utils for unified text quality assessment.
        """
        return (
            not text  # Empty text
            or not text.strip()  # Only whitespace
            or TextUtils.is_probably_gibberish(text)  # Gibberish content
        )

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace in text by replacing multiple whitespace with single space.
        Improves text consistency for processing.
        """
        if not text:
            return ""

        # Replace multiple whitespace characters with single space
        normalized = re.sub(r"\s+", " ", text)
        return normalized.strip()

    @staticmethod
    def remove_urls(text: str) -> str:
        """
        Remove URLs from text using regex pattern matching.
        Useful for cleaning text content for analysis.
        """
        if not text:
            return ""

        # Simple URL pattern for removal
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.sub(url_pattern, "", text)

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """
        Extract URLs from text using regex pattern matching.
        Useful for web content analysis and link extraction.
        """
        if not text:
            return []

        # Simple URL pattern for extraction
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text)

    @staticmethod
    def strip_vietnamese_accents(text: str) -> str:
        """
        Remove Vietnamese diacritics from a string while preserving ASCII characters.
        Ensures special Vietnamese letters 'đ'/'Đ' are mapped to 'd'/'D'.
        """
        if not text:
            return ""

        import unicodedata

        # Normalize to NFD to separate base chars and combining marks
        normalized = unicodedata.normalize("NFD", text)
        # Drop all combining marks
        without_marks = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        # Map special Vietnamese letters
        without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
        # Return as-is (do not recompose) to avoid reintroducing marks
        return without_marks

    @staticmethod
    def clean_chunk_text(text: str) -> str:
        """
        Remove empty lines and table border lines from a chunk.
        """
        lines = text.splitlines()
        filtered = [
            line
            for line in lines
            if line.strip() and not re.fullmatch(r"\|[-| ]+\|", line.strip())
        ]
        return "\n".join(filtered)

    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = 1000, overlap: int = 200, language: str = "vi"
    ) -> list:
        """
        Hybrid chunking: group full sentences into chunks up to chunk_size, with overlap at sentence boundaries.
        Uses NLTK's sent_tokenize for sentence splitting if available, supporting English and Vietnamese.
        Falls back to regex if NLTK is unavailable.
        Args:
            text: Input text
            chunk_size: Max chunk size (characters)
            overlap: Overlap size (characters)
            language: Language code for sentence splitting ('en' or 'vi')
        Returns:
            List of chunked strings
        """
        import re

        if not text or not text.strip():
            return []
        text = text.strip()
        if len(text) <= chunk_size:
            return [text]
        # For markdown content, preserve line structure and only split at major boundaries
        lines = text.split('\n')
        current_section = []
        sections = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new section or is a separator
            is_header = line.startswith('##')
            # Check if the entire row is NaN (all cells contain NaN)
            is_nan_separator = line.strip().startswith('|') and line.strip().endswith('|') and all('NaN' in cell.strip() for cell in line.strip().split('|')[1:-1] if cell.strip())
            
            if is_header and current_section:
                # Save current section and start new one
                sections.append('\n'.join(current_section))
                current_section = [line]
            elif is_nan_separator and current_section:
                # Save current section and start new one, but don't include the separator row
                sections.append('\n'.join(current_section))
                current_section = []  # Don't add the separator row
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section:
            sections.append('\n'.join(current_section))
        
        # If no sections were created, treat the whole text as one section
        if not sections:
            sections = [text]
        
        # Now chunk each section using the original logic
        all_chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                all_chunks.append(section)
            else:
                # Use original sentence-based chunking for large sections
                try:
                    import nltk
                    try:
                        nltk.data.find("tokenizers/punkt")
                    except LookupError:
                        nltk.download("punkt")
                    from nltk.tokenize import sent_tokenize
                    
                    lang_map = {"vi": "vietnamese", "en": "english"}
                    lang = lang_map.get(language, "vietnamese")
                    sentences = sent_tokenize(section, language=lang)
                except Exception:
                    if language == "vi":
                        sentences = re.split(r"(?<=[.!?。！？])\s+", section)
                    else:
                        sentences = re.split(r"(?<=[.!?])\s+", section)
                
                # Chunk the sentences within this section
                chunks = []
                current_chunk = []
                current_len = 0
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    if current_len + len(sentence) + 1 > chunk_size and current_chunk:
                        chunks.append(" ".join(current_chunk).strip())
                        if overlap > 0 and current_chunk:
                            overlap_text = current_chunk[-1]
                            current_chunk = [overlap_text]
                            current_len = len(overlap_text)
                        else:
                            current_chunk = []
                            current_len = 0
                    
                    current_chunk.append(sentence)
                    current_len += len(sentence) + 1
                
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())
                
                all_chunks.extend([c for c in chunks if c.strip()])
        
        return all_chunks
