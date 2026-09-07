"""
File manager for handling document chunk storage and file operations.
Provides safe file management with chunk organization and cleanup capabilities.
"""

# Standard library imports
import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import aiofiles

# Local imports
from config.settings import Config
from utils.file_operations.file_manager import FileManager as FileUtils
from utils.text_processing.text_utils import TextUtils

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


class FileManager:
    """
    Manages file storage and organization for document processing.
    Handles chunk storage, file cleanup, and safe file operations.
    """

    def __init__(self):
        """
        Initialize file manager without upload directory dependency.
        Files are processed directly in memory for better performance.
        """
        # No upload directory needed - we process files directly

    async def save_chunks_to_files(
        self, documents: List[str], filename: str
    ) -> Dict[str, Any]:
        """
        Save document chunks to files, replacing any previous version of the same file.

        Chunks are staged in a temporary directory and swapped in, so re-uploading a file
        never leaves chunks of the old version behind. The vector store rebuilds from this
        directory, so a leftover chunk would keep being answered as if it were current.

        Args:
            documents: List of document chunks to save
            filename: Original filename, used to identify the document

        Returns:
            Dict with the chunks directory, chunk count, content hash and replacement info
        """
        chunks_root = Path(Config.File.CHUNKS_DIR())
        target_dir = chunks_root / self._source_dir_name(filename)

        # Stage outside the chunks root so a rebuild never indexes a half-written directory
        staging_root = chunks_root.parent / ".chunk_staging"
        staging_dir = staging_root / uuid.uuid4().hex
        FileUtils.ensure_directory_exists(str(staging_dir))

        try:
            cleaned_chunks = [
                cleaned
                for cleaned in (TextUtils.clean_chunk_text(chunk) for chunk in documents)
                if cleaned.strip()
            ]

            for idx, chunk in enumerate(cleaned_chunks, start=1):
                chunk_path = staging_dir / f"chunk_{idx:03d}.txt"
                async with aiofiles.open(chunk_path, "w", encoding="utf-8") as f:
                    await f.write(chunk)

            content_hash = hashlib.sha256(
                "\n".join(cleaned_chunks).encode("utf-8")
            ).hexdigest()

            manifest = {
                "original_filename": filename,
                "chunk_count": len(cleaned_chunks),
                "content_sha256": content_hash,
                "created_at": datetime.now().isoformat(),
            }
            manifest_path = staging_dir / MANIFEST_FILENAME
            async with aiofiles.open(manifest_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(manifest, ensure_ascii=False, indent=2))

            previous = self._read_manifest(target_dir)
            legacy_removed = self._remove_legacy_dir(chunks_root, filename, target_dir)
            replaced = target_dir.exists() or legacy_removed

            self._swap_in(staging_dir, target_dir, staging_root)
        except Exception:
            FileUtils.safe_delete_directory(str(staging_dir))
            raise

        self._log_replacement(
            filename, target_dir, previous, content_hash, len(cleaned_chunks)
        )

        return {
            "path": str(target_dir),
            "source_name": target_dir.name,
            "chunk_count": len(cleaned_chunks),
            "content_sha256": content_hash,
            "replaced_existing": replaced,
            "previous_chunk_count": previous.get("chunk_count") if previous else None,
            "content_changed": (
                previous is None or previous.get("content_sha256") != content_hash
            ),
        }

    @staticmethod
    def _source_dir_name(filename: str) -> str:
        """
        Build the directory name identifying this document.

        Keeps the extension so report.pdf and report.docx stay separate, and appends a
        short hash whenever sanitizing changed the name, so two different names cannot
        collapse into the same directory.
        """
        name = Path(filename).name
        safe = FileUtils.sanitize_filename(name)
        if safe != name:
            digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
            safe = f"{safe}_{digest}"
        return safe

    @staticmethod
    def _read_manifest(source_dir: Path) -> Optional[Dict[str, Any]]:
        """Read a source directory's manifest, or None if it has none."""
        manifest_path = source_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read manifest at {manifest_path}: {e}")
            return None

    @classmethod
    def _remove_legacy_dir(
        cls, chunks_root: Path, filename: str, target_dir: Path
    ) -> bool:
        """
        Remove a directory written by the old stem-based naming scheme.

        Older uploads stored chunks under the filename stem with no extension and no
        manifest. Without this the old directory would survive the swap and its chunks
        would stay in the index alongside the new ones.
        """
        legacy_name = FileUtils.sanitize_filename(Path(filename).stem)
        legacy_dir = chunks_root / legacy_name

        if legacy_dir == target_dir or not legacy_dir.is_dir():
            return False
        if cls._read_manifest(legacy_dir) is not None:
            return False  # Written by the current scheme, belongs to another document

        logger.info(f"🧹 Removing legacy chunk directory for {filename}: {legacy_dir}")
        return FileUtils.safe_delete_directory(str(legacy_dir))

    @staticmethod
    def _swap_in(staging_dir: Path, target_dir: Path, staging_root: Path) -> None:
        """
        Move the staged chunks into place, discarding any previous version.

        The old directory is renamed aside first, so the target is only missing for the
        duration of a rename rather than for the whole deletion.
        """
        FileUtils.ensure_directory_exists(str(target_dir.parent))
        discarded_dir = None

        if target_dir.exists():
            discarded_dir = staging_root / f"discarded_{uuid.uuid4().hex}"
            target_dir.replace(discarded_dir)

        try:
            staging_dir.replace(target_dir)
        except Exception:
            if discarded_dir is not None:
                discarded_dir.replace(target_dir)  # Put the previous version back
            raise

        if discarded_dir is not None:
            FileUtils.safe_delete_directory(str(discarded_dir))

    @staticmethod
    def _log_replacement(
        filename: str,
        target_dir: Path,
        previous: Optional[Dict[str, Any]],
        content_hash: str,
        chunk_count: int,
    ) -> None:
        """Log what the upload replaced, flagging same-name-different-content uploads."""
        if previous and previous.get("content_sha256") != content_hash:
            logger.warning(
                f"⚠️ {filename} replaced a different document stored under the same name "
                f"(was '{previous.get('original_filename')}', "
                f"{previous.get('chunk_count')} chunks -> {chunk_count} chunks)"
            )
        logger.info(f"💾 Saved {chunk_count} chunks to {target_dir}")

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file safely using consolidated FileUtils.
        Provides error handling and logging for file deletion operations.
        """
        return FileUtils.safe_delete_file(file_path)

    def delete_directory(self, dir_path: str) -> bool:
        """
        Delete a directory and all its contents safely using consolidated FileUtils.
        Provides error handling and logging for directory deletion operations.
        """
        return FileUtils.safe_delete_directory(dir_path)
