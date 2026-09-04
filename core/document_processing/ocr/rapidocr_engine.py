"""
RapidOCR backend: PP-OCRv6 on ONNXRuntime.
Fast, but drops some Vietnamese tone marks. Models download on first use.
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


class RapidOCREngine(OCREngine):
    """Fast ONNXRuntime OCR backed by RapidOCR."""

    name = "rapidocr"

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        model_size: Optional[str] = None,
    ):
        super().__init__(languages=languages, use_gpu=use_gpu)
        self._engine = None
        #: Overrides the configured tier, for cheap probe recognitions.
        self._model_size = model_size

    def is_available(self) -> bool:
        try:
            from rapidocr import RapidOCR  # noqa: F401

            return True
        except ImportError:
            return False

    #: PP-OCRv6 tiers; 'small' is the accuracy/speed sweet spot.
    MODEL_SIZES = {"tiny": "tiny", "small": "small", "medium": "medium"}

    def _get_engine(self):
        """Build the engine on first use (downloads models)."""
        if self._engine is None:
            from rapidocr import ModelType, OCRVersion, RapidOCR

            lang = self.languages[0] if self.languages else "vi"
            size = (self._model_size or Config.OCR.OCR_RAPIDOCR_MODEL_SIZE()).strip().lower()
            if size not in self.MODEL_SIZES:
                logger.warning(f"Unknown RapidOCR model size '{size}', using 'small'")
                size = "small"
            model_type = ModelType(self.MODEL_SIZES[size])

            logger.info(f"Loading RapidOCR engine (PP-OCRv6 {size}, lang={lang})...")
            self._engine = RapidOCR(
                params={
                    "Det.ocr_version": OCRVersion.PPOCRV6,
                    "Det.model_type": model_type,
                    "Det.lang_type": lang,
                    "Rec.ocr_version": OCRVersion.PPOCRV6,
                    "Rec.model_type": model_type,
                    "Rec.lang_type": lang,
                }
            )
            logger.info("RapidOCR engine ready")
        return self._engine

    def recognize(self, image: np.ndarray) -> OCRPageResult:
        engine = self._get_engine()
        result = engine(image)

        boxes = []
        if result is not None and result.txts:
            scores = result.scores or []
            points = result.boxes if result.boxes is not None else []
            for i, text in enumerate(result.txts):
                confidence = float(scores[i]) if i < len(scores) else None
                if i < len(points):
                    corners = points[i]
                    ys = [float(c[1]) for c in corners]
                    xs = [float(c[0]) for c in corners]
                    top, bottom, left = min(ys), max(ys), min(xs)
                else:
                    # No geometry; keep the engine's own ordering.
                    top = bottom = float(i)
                    left = 0.0
                boxes.append(
                    OCRBox(text=str(text), top=top, bottom=bottom, left=left, confidence=confidence)
                )

        return OCRPageResult(lines=group_into_lines(boxes), engine=self.name)
