"""
Base contract for OCR engines.
"""

# Standard library imports
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

# Third-party imports
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRLine:
    """One recognised line of text."""

    text: str
    confidence: Optional[float] = None


@dataclass
class OCRBox:
    """A recognised text region and its position on the page."""

    text: str
    top: float
    bottom: float
    left: float
    confidence: Optional[float] = None


@dataclass
class OCRPageResult:
    """Result of running OCR over one page image."""

    lines: List[OCRLine] = field(default_factory=list)
    engine: str = ""

    @property
    def text(self) -> str:
        """Recognised text, one line per entry."""
        return "\n".join(line.text for line in self.lines if line.text.strip())

    @property
    def mean_confidence(self) -> Optional[float]:
        """Average line confidence, or None if the engine reports none."""
        scores = [l.confidence for l in self.lines if l.confidence is not None]
        return sum(scores) / len(scores) if scores else None


def group_into_lines(boxes: List[OCRBox]) -> List[OCRLine]:
    """
    Put detected regions back into reading order.

    Detectors emit regions in their own order, not reading order. Groups regions
    with overlapping vertical centres into lines and sorts each line left to right.
    """
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda b: (b.top, b.left))
    lines: List[List[OCRBox]] = []
    current: List[OCRBox] = []
    top = bottom = 0.0

    for box in ordered:
        centre = (box.top + box.bottom) / 2
        if current and not (top <= centre <= bottom):
            lines.append(current)
            current = []

        if not current:
            top, bottom = box.top, box.bottom
        else:
            top, bottom = min(top, box.top), max(bottom, box.bottom)
        current.append(box)

    if current:
        lines.append(current)

    result = []
    for line in lines:
        parts = sorted(line, key=lambda b: b.left)
        scores = [p.confidence for p in parts if p.confidence is not None]
        result.append(
            OCRLine(
                text=" ".join(p.text for p in parts),
                confidence=sum(scores) / len(scores) if scores else None,
            )
        )
    return result


class OCREngine(ABC):
    """Base class for OCR backends. Models load lazily on first recognise()."""

    #: Identifier used in configuration and metadata.
    name: str = "base"

    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = False):
        self.languages = languages or ["vi"]
        self.use_gpu = use_gpu

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRPageResult:
        """Recognise text in a page image (grayscale or BGR)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this engine's dependencies are importable."""
