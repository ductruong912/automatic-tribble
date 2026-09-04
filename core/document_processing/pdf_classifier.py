"""
Per-page PDF classification: real text layer versus scanned image.
"""

# Standard library imports
import logging
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import List

# Third-party imports
import pymupdf

# Local imports
from config.settings import Config

logger = logging.getLogger(__name__)


class PageKind(str, Enum):
    """How a page's text should be obtained."""

    TEXT = "text"  # Embedded text layer is usable; read it directly
    SCAN = "scan"  # No usable text layer; needs OCR


#: Image coverage above which a page is a scan, not a page with a picture.
FULL_PAGE_IMAGE_COVERAGE = 0.8


@dataclass
class PageInfo:
    """Classification result for one PDF page."""

    index: int  # 0-based page number
    kind: PageKind
    text: str  # Embedded text, empty for scanned pages
    char_count: int
    image_count: int
    image_coverage: float = 0.0  # Share of the page area covered by images
    diacritic_ratio: float = 0.0  # Share of letters carrying Vietnamese diacritics


def diacritic_ratio(text: str) -> float:
    """
    Share of letters carrying a Vietnamese diacritic (including 'đ').
    Real Vietnamese prose runs above 10%; near zero means a broken text layer.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    accented = sum(
        1 for c in letters if len(unicodedata.normalize("NFD", c)) > 1 or c in "đĐ"
    )
    return accented / len(letters)


def _image_coverage(page) -> float:
    """Share of the page area covered by embedded images, capped at 1.0."""
    page_area = abs(page.rect.width * page.rect.height)
    if not page_area:
        return 0.0

    covered = 0.0
    for info in page.get_images(full=True):
        for rect in page.get_image_rects(info[0]):
            covered += abs(rect.width * rect.height)
    return min(covered / page_area, 1.0)


@dataclass
class PdfProfile:
    """Classification result for a whole PDF."""

    pages: List[PageInfo]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def scan_pages(self) -> List[PageInfo]:
        return [p for p in self.pages if p.kind is PageKind.SCAN]

    @property
    def is_fully_text(self) -> bool:
        """Every page has a usable text layer."""
        return bool(self.pages) and not self.scan_pages

    @property
    def is_fully_scanned(self) -> bool:
        """No page has a usable text layer."""
        return bool(self.pages) and len(self.scan_pages) == len(self.pages)

    def summary(self) -> str:
        scanned = len(self.scan_pages)
        return f"{self.page_count} pages ({self.page_count - scanned} text, {scanned} scan)"


def classify_pdf(
    content: bytes, min_chars: int = None, min_diacritic_ratio: float = None
) -> PdfProfile:
    """
    Classify each page of a PDF as text or scan.

    A page is TEXT when its text layer yields at least ``min_chars``
    non-whitespace characters. Pages below that go to OCR only if they carry an
    image, so sparse but digital pages are not OCRed for nothing.

    Args:
        content: Raw PDF bytes.
        min_chars: Characters required to trust a page's text layer;
            defaults to the PDF_TEXT_MIN_CHARS_PER_PAGE setting.

    Returns:
        PdfProfile holding one PageInfo per page.
    """
    if min_chars is None:
        min_chars = Config.OCR.PDF_TEXT_MIN_CHARS_PER_PAGE()
    if min_diacritic_ratio is None:
        min_diacritic_ratio = Config.OCR.PDF_TEXT_MIN_DIACRITIC_RATIO()

    # The diacritic test only means anything when we expect Vietnamese text.
    check_diacritics = "vi" in Config.OCR.OCR_LANGUAGES()

    pages: List[PageInfo] = []
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text") or ""
            stripped = "".join(text.split())
            image_count = len(page.get_images(full=True))
            coverage = _image_coverage(page) if image_count else 0.0
            accents = diacritic_ratio(text)

            has_enough_text = len(stripped) >= min_chars
            # Full-page image + diacritic-free text = stale OCR layer; re-read it.
            stale_ocr_layer = (
                check_diacritics
                and coverage >= FULL_PAGE_IMAGE_COVERAGE
                and accents < min_diacritic_ratio
            )

            if stale_ocr_layer or (not has_enough_text and image_count > 0):
                kind = PageKind.SCAN
            else:
                kind = PageKind.TEXT

            if stale_ocr_layer:
                logger.info(
                    f"Page {index + 1}: text layer has {accents:.1%} accented letters "
                    f"over a full-page image - treating as a scan"
                )

            pages.append(
                PageInfo(
                    index=index,
                    kind=kind,
                    text=text if kind is PageKind.TEXT else "",
                    char_count=len(stripped),
                    image_count=image_count,
                    image_coverage=coverage,
                    diacritic_ratio=accents,
                )
            )

    profile = PdfProfile(pages=pages)
    logger.info(f"PDF classified: {profile.summary()}")
    return profile
