# Standard library imports
from pathlib import Path
from typing import List, Dict, Any

# Local imports
from config.settings import Config


class MetadataStore:
    """Handles document metadata storage and management."""

    def __init__(self):
        self.chunks_dir = Path(Config.File.CHUNKS_DIR())

    def create_metadata_from_chunks(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Create metadata for each document from chunk files."""
        document_metadata = []

        if not self.chunks_dir.exists():
            # Fallback metadata if no chunks directory
            return [{"source": "unknown", "source_type": "unknown"} for _ in documents]

        # Ensure deterministic ordering of sources to align with DocumentStore
        for chunk_subdir in sorted(self.chunks_dir.iterdir(), key=lambda p: p.name.lower()):
            if chunk_subdir.is_dir():
                source_name = chunk_subdir.name
                # Determine source type based on directory name
                if source_name.startswith(("http", "www")):
                    source_type = "url"
                else:
                    source_type = "file"

                # List chunk files for this source in sorted order to match documents order
                chunk_files = sorted(chunk_subdir.glob("chunk_*.txt"), key=lambda p: p.name)
                for chunk_file in chunk_files:
                    document_metadata.append(
                        {
                            "source_id": source_name,
                            "source_type": source_type,
                            "source_name": source_name,
                            "chunk_file": chunk_file.name,
                            "chunk_path": str(chunk_file),
                        }
                    )

        return document_metadata

    def get_metadata_by_source(self, source_name: str) -> Dict[str, Any]:
        """Get metadata for a specific source."""
        source_dir = self.chunks_dir / source_name

        if not source_dir.exists():
            return {}

        # Determine source type
        if source_name.startswith(("http", "www")):
            source_type = "url"
        else:
            source_type = "file"

        # Count chunks
        chunk_files = list(source_dir.glob("chunk_*.txt"))

        return {
            "source_id": source_name,
            "source_name": source_name,
            "source_type": source_type,
            "chunk_count": len(chunk_files),
            "directory": str(source_dir),
        }

    def get_all_sources_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all sources."""
        metadata_list = []

        if not self.chunks_dir.exists():
            return metadata_list

        for chunk_subdir in self.chunks_dir.iterdir():
            if chunk_subdir.is_dir():
                metadata = self.get_metadata_by_source(chunk_subdir.name)
                if metadata:
                    metadata_list.append(metadata)

        return metadata_list

    def update_source_metadata(
        self, source_name: str, metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for a specific source."""
        try:
            # For now, we store metadata implicitly through directory structure
            # In the future, this could write to a metadata file
            source_dir = self.chunks_dir / source_name

            if source_dir.exists():
                # Could write metadata.json file here if needed
                print(f"📝 Metadata updated for source: {source_name}")
                return True

            return False

        except Exception as e:
            print(f" Error updating metadata for {source_name}: {e}")
            return False

    def get_source_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about stored sources."""
        stats = {
            "total_sources": 0,
            "total_chunks": 0,
            "sources_by_type": {"file": 0, "url": 0},
            "sources": [],
        }

        if not self.chunks_dir.exists():
            return stats

        for chunk_subdir in self.chunks_dir.iterdir():
            if chunk_subdir.is_dir():
                source_name = chunk_subdir.name
                chunk_count = len(list(chunk_subdir.glob("chunk_*.txt")))

                # Determine source type
                if source_name.startswith(("http", "www")):
                    source_type = "url"
                    stats["sources_by_type"]["url"] += 1
                else:
                    source_type = "file"
                    stats["sources_by_type"]["file"] += 1

                stats["total_sources"] += 1
                stats["total_chunks"] += chunk_count
                stats["sources"].append(
                    {"name": source_name, "type": source_type, "chunks": chunk_count}
                )

        return stats
