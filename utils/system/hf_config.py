"""
Hugging Face Hub Windows compatibility utilities.
Fixes [WinError 1314] symlink errors on Windows by falling back to file copies
when Developer Mode is off (compatible with huggingface_hub==0.33.1).
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

_hf_hub_setup_done = False


def check_symlink_support() -> bool:
    """
    Test whether the current process can create symbolic links on this OS/filesystem.
    Returns True if symlink creation succeeds, False otherwise.
    """
    if sys.platform != "win32":
        return True

    temp_dir = tempfile.gettempdir()
    src_file = os.path.join(temp_dir, f"_hf_symlink_test_src_{os.getpid()}.tmp")
    dst_file = os.path.join(temp_dir, f"_hf_symlink_test_dst_{os.getpid()}.tmp")

    try:
        # Create dummy src file
        with open(src_file, "w", encoding="utf-8") as f:
            f.write("test")

        # Try creating symlink
        os.symlink(src_file, dst_file)
        return True
    except (OSError, NotImplementedError, PermissionError):
        return False
    finally:
        # Clean up temporary test files
        for p in (dst_file, src_file):
            try:
                if os.path.islink(p) or os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def setup_huggingface_hub(force_disable_symlinks: Optional[bool] = None) -> bool:
    """
    Configure Hugging Face Hub environment and apply compatibility shim for Windows.

    Args:
        force_disable_symlinks: If explicitly provided, forces symlinks on (False) or off (True).
                                If None, checks env var HF_HUB_DISABLE_SYMLINKS or probes system capability.

    Returns:
        bool: True if symlinks are disabled, False if symlinks are enabled and supported.
    """
    global _hf_hub_setup_done

    # 1. Determine whether symlinks should be disabled
    env_disable = os.getenv("HF_HUB_DISABLE_SYMLINKS")
    if force_disable_symlinks is not None:
        disable_symlinks = force_disable_symlinks
    elif env_disable is not None:
        disable_symlinks = env_disable.strip().lower() in ("1", "true", "yes", "on")
    else:
        # On Windows, probe whether symlinks actually work
        if sys.platform == "win32":
            can_symlink = check_symlink_support()
            disable_symlinks = not can_symlink
            if disable_symlinks:
                logger.info(
                    "[HF Hub] Windows detected without Developer Mode/Admin symlink privileges. "
                    "Enabling file copy fallback for HuggingFace Hub."
                )
        else:
            disable_symlinks = False

    # 2. Set standard HF environment variables
    if disable_symlinks:
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    else:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "0")

    # 3. Patch huggingface_hub if installed
    try:
        import huggingface_hub.constants as hf_constants
        import huggingface_hub.file_download as hf_file_download

        # Sync constants if already imported
        if os.getenv("HF_HUB_DISABLE_SYMLINKS_WARNING") in ("1", "true", "yes", "on"):
            hf_constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True

        if disable_symlinks and not _hf_hub_setup_done:
            orig_are_symlinks_supported = getattr(hf_file_download, "are_symlinks_supported", None)
            orig_create_symlink = getattr(hf_file_download, "_create_symlink", None)

            # Shim are_symlinks_supported to always report False when symlinks are disabled
            def _patched_are_symlinks_supported(cache_dir=None) -> bool:
                return False

            hf_file_download.are_symlinks_supported = _patched_are_symlinks_supported

            # Wrap _create_symlink to catch [WinError 1314] / OSError and fall back to copying
            if orig_create_symlink is not None:
                def _safe_create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
                    try:
                        orig_create_symlink(src, dst, new_blob=new_blob)
                    except OSError as e:
                        # Catch WinError 1314 or any symlink OSError
                        import shutil
                        abs_src = os.path.abspath(os.path.expanduser(src))
                        abs_dst = os.path.abspath(os.path.expanduser(dst))
                        try:
                            os.remove(abs_dst)
                        except OSError:
                            pass

                        if new_blob:
                            copy_fn = getattr(hf_file_download, "_copy_no_matter_what", shutil.copyfile)
                            shutil.move(abs_src, abs_dst, copy_function=copy_fn)
                        else:
                            shutil.copyfile(abs_src, abs_dst)
                        logger.debug(f"[HF Hub Shim] Fallback copy completed for {abs_dst} after OSError: {e}")

                hf_file_download._create_symlink = _safe_create_symlink

            _hf_hub_setup_done = True
            logger.info("[HF Hub] Hugging Face Hub symlink compatibility shim activated successfully.")

    except ImportError:
        # huggingface_hub is not installed yet
        pass
    except Exception as e:
        logger.warning(f"[HF Hub] Could not apply HuggingFace Hub compatibility patch: {e}")

    return disable_symlinks
