"""
Scanned-PDF text extraction: render, straighten, clean up, OCR.
Pages with a usable text layer never reach this module.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pymupdf

# Local imports
from config.settings import Config
from ..ocr import OCREngine, get_ocr_engine
from ..orientation import deskew_quarter_turns, detect_rotation
from ..pdf_classifier import PageInfo, PageKind, PdfProfile, classify_pdf
from .preprocessing import DocumentPreprocessor, PreprocessingConfig

logger = logging.getLogger(__name__)


def build_preprocessing_config(profile: str) -> Optional[PreprocessingConfig]:
    """
    Build a preprocessing configuration for a named profile.

    Binarisation and sharpening lower neural OCR accuracy on Vietnamese, so only
    deskew and denoise are enabled by default; "full" keeps the old behaviour.

    Args:
        profile: "none", "light", "standard" or "full".

    Returns:
        A PreprocessingConfig, or None when no preprocessing should run.
    """
    profile = (profile or "standard").strip().lower()

    if profile == "none":
        return None

    disabled = dict(
        deskew_enabled=False,
        denoise_enabled=False,
        binarize_enabled=False,
        contrast_enabled=False,
        sharpen_enabled=False,
        crop_margins_enabled=False,
        scale_enabled=False,
        remove_artifacts_enabled=False,
    )

    if profile == "light":
        return PreprocessingConfig(**{**disabled, "deskew_enabled": True})

    if profile == "full":
        from .preprocessing import create_ocr_optimized_config

        return create_ocr_optimized_config()

    if profile != "standard":
        logger.warning(f"Unknown preprocessing profile '{profile}', using 'standard'")

    return PreprocessingConfig(
        **{
            **disabled,
            "deskew_enabled": True,
            "denoise_enabled": True,
            "denoise_method": "bilateral",
        }
    )


class ScannedPdfExtractor:
    """
    Turns scanned PDF pages into text.
    One instance reuses its loaded OCR models across documents.
    """

    def __init__(self, engine: Optional[OCREngine] = None, preprocess_profile: Optional[str] = None):
        self._engine = engine
        self._profile_name = preprocess_profile or Config.OCR.OCR_PREPROCESS_PROFILE()
        config = build_preprocessing_config(self._profile_name)
        self._preprocessor = DocumentPreprocessor(config) if config else None
        self._dpi = Config.OCR.OCR_DPI()
        self._auto_rotate = Config.OCR.OCR_AUTO_ROTATE()
        self._probe_engine = None

    @property
    def engine(self) -> OCREngine:
        """The OCR engine, resolved from configuration on first access."""
        if self._engine is None:
            self._engine = get_ocr_engine()
        return self._engine

    def render_page(self, doc: "pymupdf.Document", index: int) -> np.ndarray:
        """Render one page to a grayscale numpy array at the configured DPI."""
        pixmap = doc[index].get_pixmap(dpi=self._dpi, colorspace=pymupdf.csGRAY)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8)
        return image.reshape(pixmap.height, pixmap.width)

    def _get_probe_engine(self) -> OCREngine:
        """
        Engine used for orientation probes.
        RapidOCR's smallest tier keeps the probe near a second on any engine.
        """
        if self._probe_engine is None:
            from ..ocr import RapidOCREngine

            probe = RapidOCREngine(languages=self.engine.languages, model_size="tiny")
            self._probe_engine = probe if probe.is_available() else self.engine
        return self._probe_engine

    def _probe(self, image: np.ndarray) -> Tuple[float, int]:
        """Read a downscaled page and report how convincing the result looks."""
        result = self._get_probe_engine().recognize(image)
        confidence = result.mean_confidence or 0.0
        return confidence, len(result.text)

    def straighten(self, image: np.ndarray) -> np.ndarray:
        """Rotate a quarter-turned page upright; returns it unchanged when disabled."""
        if not self._auto_rotate:
            return image
        try:
            angle = detect_rotation(image, probe=self._probe)
            if angle:
                logger.info(f"Page is rotated {angle}deg; straightening before OCR")
                return deskew_quarter_turns(image, angle)
        except Exception as e:
            logger.warning(f"Orientation detection failed, using the page as-is: {e}")
        return image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Clean up a rendered page; returns the image unchanged when disabled."""
        if self._preprocessor is None:
            return image
        try:
            return self._preprocessor.preprocess_image(image)
        except Exception as e:
            logger.warning(f"Preprocessing failed, using the raw page image: {e}")
            return image

    def extract_page(self, doc: "pymupdf.Document", index: int) -> str:
        """Render, clean and OCR a single page."""
        image = self.preprocess(self.straighten(self.render_page(doc, index)))
        result = self.engine.recognize(image)
        confidence = result.mean_confidence
        logger.info(
            f"OCR page {index + 1}: {len(result.lines)} lines"
            + (f", mean confidence {confidence:.2f}" if confidence is not None else "")
        )
        return result.text

    def extract(self, content: bytes, profile: Optional[PdfProfile] = None) -> Dict[str, object]:
        """
        Extract text from a PDF, OCRing only the pages that need it.

        Args:
            content: Raw PDF bytes.
            profile: Pre-computed classification; computed here when omitted.

        Returns:
            Dict with the joined text, the profile used, and which pages were OCRed.
        """
        profile = profile or classify_pdf(content)
        ocr_pages: List[int] = []
        parts: List[str] = []

        with pymupdf.open(stream=content, filetype="pdf") as doc:
            for page in profile.pages:
                if page.kind is PageKind.TEXT:
                    parts.append(page.text)
                else:
                    parts.append(self.extract_page(doc, page.index))
                    ocr_pages.append(page.index)

        return {
            "text": "\n\n".join(part for part in parts if part.strip()),
            "profile": profile,
            "ocr_pages": ocr_pages,
            "engine": self.engine.name if ocr_pages else None,
            "preprocess_profile": self._profile_name if ocr_pages else None,
        }
