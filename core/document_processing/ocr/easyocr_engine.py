"""
EasyOCR backend (CRAFT detector + latin_g2 recogniser).
Best Vietnamese diacritic accuracy, slow on CPU. Models download on first use.
"""

# Standard library imports
import logging
from typing import List, Optional

# Third-party imports
import numpy as np

# Local imports
from config.settings import Config
from .base import OCRBox, OCREngine, OCRPageResult, group_into_lines

logger = logging.getLogger(__name__)


class EasyOCREngine(OCREngine):
    """Vietnamese-capable OCR backed by EasyOCR (PyTorch)."""

    name = "easyocr"

    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = False):
        super().__init__(languages=languages, use_gpu=use_gpu)
        self._reader = None

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_reader(self):
        """Build the reader on first use (downloads models)."""
        if self._reader is None:
            import easyocr

            logger.info(
                f"Loading EasyOCR reader (languages={self.languages}, gpu={self.use_gpu})..."
            )
            self._reader = easyocr.Reader(
                self.languages, gpu=self.use_gpu, verbose=False
            )
            logger.info("EasyOCR reader ready")
        return self._reader

    def recognize(self, image: np.ndarray) -> OCRPageResult:
        reader = self._get_reader()
        # low_text: stroke faintness threshold; the default misses words
        # on scanned Vietnamese.
        raw = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            low_text=Config.OCR.OCR_EASYOCR_LOW_TEXT(),
        )

        boxes = []
        for item in raw:
            box, text = item[0], item[1]
            confidence = item[2] if len(item) >= 3 else None
            ys = [point[1] for point in box]
            xs = [point[0] for point in box]
            boxes.append(
                OCRBox(
                    text=str(text),
                    top=min(ys),
                    bottom=max(ys),
                    left=min(xs),
                    confidence=confidence,
                )
            )

        return OCRPageResult(lines=group_into_lines(boxes), engine=self.name)
