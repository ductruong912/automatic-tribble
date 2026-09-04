"""
OCR backends.
Pluggable OCR engines selected through the OCR_ENGINE setting.
"""

from .base import OCREngine, OCRLine, OCRPageResult
from .easyocr_engine import EasyOCREngine
from .rapidocr_engine import RapidOCREngine
from .factory import ENGINES, get_ocr_engine, reset_ocr_engine_cache

__all__ = [
    "OCREngine",
    "OCRLine",
    "OCRPageResult",
    "EasyOCREngine",
    "RapidOCREngine",
    "ENGINES",
    "get_ocr_engine",
    "reset_ocr_engine_cache",
]
