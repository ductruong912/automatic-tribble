"""
Log utilities for viewing and managing log files.
"""

import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from config.settings import Config


class LogManager:
    """Utility class for managing log files."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or Config.Logging.LOG_DIR()

    def get_log_files(self) -> List[Dict[str, Any]]:
        """Get all log files with metadata."""
        log_files = []
        pattern = os.path.join(self.log_dir, "chatbot_*.log")

        for filepath in glob.glob(pattern):
            stat = os.stat(filepath)
            log_files.append(
                {
                    "filename": os.path.basename(filepath),
                    "filepath": filepath,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                }
            )

        # Sort by creation time (newest first)
        log_files.sort(key=lambda x: x["created"], reverse=True)
        return log_files

    def get_latest_log_file(self) -> Optional[str]:
        """Get the path to the most recent log file."""
        log_files = self.get_log_files()
        if not log_files:
            return None
        return log_files[0]["filepath"]

    def read_log_file(self, filepath: str, lines: int = 50) -> List[str]:
        """Read the last N lines from a log file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if lines > 0 else all_lines
        except Exception as e:
            return [f"Error reading log file: {e}"]

    def search_logs(
        self, search_term: str, filepath: Optional[str] = None
    ) -> List[str]:
        """Search for a term in log files."""
        if not filepath:
            filepath = self.get_latest_log_file()

        if not filepath:
            return ["No log files found"]

        matching_lines = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if search_term.lower() in line.lower():
                        matching_lines.append(line.strip())
        except Exception as e:
            return [f"Error searching log file: {e}"]

        return matching_lines

    def get_rag_debug_info(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Extract RAG-specific debug information from logs."""
        if not filepath:
            filepath = self.get_latest_log_file()

        if not filepath:
            return {"error": "No log files found"}

        rag_info = {
            "queries_processed": 0,
            "documents_retrieved": 0,
            "context_lengths": [],
            "semantic_scores": [],
            "keyword_scores": [],
            "errors": [],
        }

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line_lower = line.lower()

                    # Count queries
                    if "processing query:" in line_lower:
                        rag_info["queries_processed"] += 1

                    # Count retrieved documents
                    if "retrieved context length:" in line_lower:
                        rag_info["documents_retrieved"] += 1
                        # Extract context length
                        try:
                            length_str = line.split("retrieved context length:")[
                                1
                            ].split()[0]
                            rag_info["context_lengths"].append(int(length_str))
                        except:
                            pass

                    # Extract scores
                    if "semantic score:" in line_lower:
                        try:
                            score_str = line.split("semantic score:")[1].split()[0]
                            rag_info["semantic_scores"].append(float(score_str))
                        except:
                            pass

                    if "keyword score:" in line_lower:
                        try:
                            score_str = line.split("keyword score:")[1].split()[0]
                            rag_info["keyword_scores"].append(float(score_str))
                        except:
                            pass

                    # Collect errors
                    if "error" in line_lower or "" in line:
                        rag_info["errors"].append(line.strip())

        except Exception as e:
            rag_info["error"] = f"Error reading log file: {e}"

        return rag_info

    def cleanup_old_logs(self, keep_days: int = 7) -> int:
        """Remove log files older than specified days."""
        import time
        from utils.file_operations.file_manager import FileManager

        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        removed_count = 0
        for log_file in self.get_log_files():
            if log_file["created"].timestamp() < cutoff_time:
                try:
                    if FileManager.safe_delete_file(log_file["filepath"]):
                        removed_count += 1
                    else:
                        print(
                            f"Error removing {log_file['filename']}: Could not delete file"
                        )
                except Exception as e:
                    print(f"Error removing {log_file['filename']}: {e}")
        return removed_count


def print_log_summary():
    """Print a summary of available logs."""
    log_manager = LogManager()
    log_files = log_manager.get_log_files()

    print("📝 Log Files Summary:")
    print("=" * 50)

    if not log_files:
        print("No log files found.")
        return

    for i, log_file in enumerate(log_files[:5]):  # Show last 5 files
        size_mb = log_file["size"] / (1024 * 1024)
        print(f"{i+1}. {log_file['filename']}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Created: {log_file['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Modified: {log_file['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    if len(log_files) > 5:
        print(f"... and {len(log_files) - 5} more files")


def print_rag_debug_summary():
    """Print RAG debug information from logs."""
    log_manager = LogManager()
    rag_info = log_manager.get_rag_debug_info()

    print("RAG Debug Summary:")
    print("=" * 50)

    if "error" in rag_info:
        print(f" {rag_info['error']}")
        return

    print(f" Queries Processed: {rag_info['queries_processed']}")
    print(f"📄 Documents Retrieved: {rag_info['documents_retrieved']}")

    if rag_info["context_lengths"]:
        avg_length = sum(rag_info["context_lengths"]) / len(rag_info["context_lengths"])
        print(f"📏 Average Context Length: {avg_length:.0f} characters")

    if rag_info["semantic_scores"]:
        avg_semantic = sum(rag_info["semantic_scores"]) / len(
            rag_info["semantic_scores"]
        )
        print(f" Average Semantic Score: {avg_semantic:.3f}")

    if rag_info["keyword_scores"]:
        avg_keyword = sum(rag_info["keyword_scores"]) / len(rag_info["keyword_scores"])
        print(f"🔤 Average Keyword Score: {avg_keyword:.3f}")

    if rag_info["errors"]:
        print(f"\n Errors Found: {len(rag_info['errors'])}")
        for error in rag_info["errors"][:3]:  # Show first 3 errors
            print(f"   - {error}")


if __name__ == "__main__":
    print_log_summary()
    print()
    print_rag_debug_summary()
