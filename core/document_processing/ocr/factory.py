"""
OCR engine selection: resolves the configured engine and caches one per process.
"""

# Standard library imports
import logging
import threading
from typing import Dict, Optional, Type

# Local imports
from config.settings import Config
from .base import OCREngine
from .easyocr_engine import EasyOCREngine
from .rapidocr_engine import RapidOCREngine

logger = logging.getLogger(__name__)

#: Engines selectable through OCR_ENGINE.
ENGINES: Dict[str, Type[OCREngine]] = {
    EasyOCREngine.name: EasyOCREngine,
    RapidOCREngine.name: RapidOCREngine,
}

_engine_cache: Dict[str, OCREngine] = {}
_cache_lock = threading.Lock()


def _resolve_gpu() -> bool:
    """True only when a CUDA-enabled torch build can see a device."""
    setting = Config.OCR.OCR_GPU()
    if setting == "false":
        return False

    try:
        import torch

        available = torch.cuda.is_available()
    except ImportError:
        available = False

    if setting == "true" and not available:
        logger.warning("OCR_GPU=true but no CUDA device is available; falling back to CPU")
    return available


def get_ocr_engine(name: Optional[str] = None) -> OCREngine:
    """
    Get the configured OCR engine, building it once per process.

    Args:
        name: Engine name; defaults to the OCR_ENGINE setting.

    Raises:
        ValueError: Unknown engine.
        RuntimeError: Engine known but its package is not installed.
    """
    name = (name or Config.OCR.OCR_ENGINE()).strip().lower()

    if name not in ENGINES:
        raise ValueError(
            f"Unknown OCR engine '{name}'. Available: {', '.join(sorted(ENGINES))}"
        )

    with _cache_lock:
        if name not in _engine_cache:
            engine = ENGINES[name](
                languages=Config.OCR.OCR_LANGUAGES(), use_gpu=_resolve_gpu()
            )
            if not engine.is_available():
                raise RuntimeError(
                    f"OCR engine '{name}' is selected but its package is not installed"
                )
            _engine_cache[name] = engine
            logger.info(f"OCR engine '{name}' selected (gpu={engine.use_gpu})")

    return _engine_cache[name]


def reset_ocr_engine_cache() -> None:
    """Drop cached engines so the next call rebuilds them."""
    with _cache_lock:
        _engine_cache.clear()
