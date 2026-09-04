"""
Page orientation detection for scanned pages.
"""

# Standard library imports
import logging
from typing import Callable, Optional, Tuple

# Third-party imports
import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: How much to rotate an image to undo a page rotation of the given angle.
_UNDO_ROTATION_ANGLE = {0: 0, 90: 270, 180: 180, 270: 90}

#: Longest edge the probe image is scaled down to.
PROBE_MAX_EDGE = 1000


def rotate(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an image by 0, 90, 180 or 270 degrees clockwise."""
    if angle == 0:
        return image
    op = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[angle]
    return cv2.rotate(image, op)


def text_axis(image: np.ndarray) -> Tuple[float, float]:
    """
    Measure whether text lines run horizontally or vertically.

    Ink projected across the lines alternates between line bodies and gaps, so
    its coefficient of variation is high; projected along them it stays flat.

    Returns:
        (row_variation, column_variation). Rows winning means horizontal text.
    """
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    binary = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def variation(values: np.ndarray) -> float:
        mean = values.mean()
        return float(values.std() / mean) if mean else 0.0

    return variation(binary.sum(axis=1).astype(np.float64)), variation(
        binary.sum(axis=0).astype(np.float64)
    )


def _downscale(image: np.ndarray, max_edge: int = PROBE_MAX_EDGE) -> np.ndarray:
    """Shrink an image to keep probe recognitions cheap."""
    longest = max(image.shape[:2])
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def detect_rotation(
    image: np.ndarray, probe: Optional[Callable[[np.ndarray], Tuple[float, int]]] = None
) -> int:
    """
    Detect how far a page image has been rotated clockwise.

    The projection profile gives the axis for free; telling apart the two
    orientations sharing that axis needs the probe, run on a downscaled copy.

    Args:
        image: Page image, grayscale.
        probe: Returns (mean_confidence, character_count) for an image.
            Without it only the axis is corrected.

    Returns:
        0, 90, 180 or 270.
    """
    rows, cols = text_axis(image)
    horizontal = rows >= cols
    candidates = (0, 180) if horizontal else (90, 270)

    if probe is None:
        return candidates[0]

    small = _downscale(image)
    scores = {}
    for angle in candidates:
        confidence, characters = probe(rotate(small, _UNDO_ROTATION_ANGLE[angle]))
        scores[angle] = confidence * characters

    best = max(scores, key=scores.get)
    logger.debug(
        f"Orientation: axis={'horizontal' if horizontal else 'vertical'} "
        f"(rows={rows:.2f}, cols={cols:.2f}), scores={scores}, chose {best}deg"
    )
    return best


def deskew_quarter_turns(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an image so a page rotated by ``angle`` comes out upright."""
    return rotate(image, _UNDO_ROTATION_ANGLE[angle])
