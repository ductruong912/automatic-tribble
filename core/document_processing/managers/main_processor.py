"""
Main document processor for orchestrating specialized file processors.
Provides unified interface for processing various file formats with metadata generation.
Now uses Docling for enhanced document conversion with heading-aware chunking.
"""

# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Local imports
from config.settings import Config
from ..processors.docling_processor import DoclingProcessor, AsyncDoclingProcessor
from .file_manager import FileManager


class MainDocumentProcessor:
    """
    Main document processor that orchestrates all specialized processors.
    Now uses Docling for enhanced document conversion with heading-aware chunking.
    Provides unified interface for processing various file formats with comprehensive metadata.
    """

    SUPPORTED_EXTENSIONS = set(Config.File.ALLOWED_EXTENSIONS())

    def __init__(
        self,
        file_manager=None,
        enable_ocr: bool = None,  # Deprecated - OCR is now automatic for PDFs
        llm_client=None,
        llm_model: str = None,
        preprocessing_config: str = "ocr_optimized",
    ):
        """
        Initialize main processor with file manager and Docling processor.
        - PDFs: Automatically use OCR (like the test file)
        - Other formats: Use normal Docling extraction
        
        Args:
            file_manager: File manager for handling file operations
            enable_ocr: Deprecated - OCR is now automatic for PDFs
            llm_client: OpenAI client for enhanced processing
            llm_model: LLM model to use for enhanced processing
            preprocessing_config: Preprocessing configuration name ("ocr_optimized", "fast", "high_quality", "default")
        """
        self.file_manager = file_manager if file_manager is not None else FileManager()
        
        # Initialize Docling processor with automatic OCR for PDFs and preprocessing
        self.docling_processor = DoclingProcessor(
            enable_ocr=False,  # OCR is now automatic based on file type
            llm_client=llm_client,
            llm_model=llm_model,
            preprocessing_config=preprocessing_config
        )
        
        # Create async wrapper for compatibility
        self.async_processor = AsyncDoclingProcessor(
            enable_ocr=False,  # OCR is now automatic based on file type
            llm_client=llm_client,
            llm_model=llm_model,
            preprocessing_config=preprocessing_config
        )
        
        logger = __import__('logging').getLogger(__name__)
        logger.info(f"MainDocumentProcessor initialized with automatic OCR for PDFs, preprocessing config: {preprocessing_config}")

    async def process_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process file content using appropriate processor based on file type.
        Uses custom SpreadsheetProcessor for Excel files and Docling for other formats.

        Args:
            file_content: Raw file content in bytes
            filename: Original filename for processing and metadata

        Returns:
            Dict containing processed documents and comprehensive metadata
        """
        file_ext = Path(filename).suffix.lower()

        # Validate file extension against supported types
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_ext}")

        try:
            # Use custom SpreadsheetProcessor for Excel files to handle sheets separately
            if file_ext in ['.xlsx', '.xls', '.csv']:
                from ..processors.processors import SpreadsheetProcessor
                spreadsheet_processor = SpreadsheetProcessor()
                
                # Process using our custom extractor that handles sheets separately
                documents = await spreadsheet_processor.process(file_content, filename)
                
                if not documents:
                    raise ValueError(f"No content could be extracted from {filename}")

                # Save chunks to files for persistence and debugging
                chunks_dir = await self.file_manager.save_chunks_to_files(
                    documents, filename
                )

                # Return comprehensive metadata and processed documents
                return {
                    "documents": documents,
                    "metadata": {
                        "chunks_directory": chunks_dir,
                        "processing_timestamp": datetime.now().isoformat(),
                        "processor_version": "spreadsheet_processor",
                        "chunk_size": Config.File.CHUNK_SIZE(),
                        "chunk_overlap": Config.File.CHUNK_OVERLAP(),
                        "file_type": file_ext,
                        "total_chunks": len(documents),
                    }
                }
            else:
                # Use Docling for other supported formats
                # Check if Docling supports this format
                if not self.docling_processor.is_format_supported(filename):
                    raise ValueError(f"Docling does not support {file_ext} files")

                # Process the file using Docling with heading-aware chunking
                result = await self.docling_processor.process_document(
                    file_content, 
                    filename
                )

                # Validate that content was successfully extracted
                if not result.get("documents"):
                    raise ValueError(f"No content could be extracted from {filename}")

                # Save chunks to files for persistence and debugging
                chunks_dir = await self.file_manager.save_chunks_to_files(
                    result["documents"], filename
                )

                # Return comprehensive metadata and processed documents
                return {
                    "documents": result["documents"],
                    "metadata": {
                        **result["metadata"],
                        "chunks_directory": chunks_dir,
                        "processing_timestamp": datetime.now().isoformat(),
                        "processor_version": result["metadata"].get("processor_version", "docling"),
                        "chunk_size": Config.File.CHUNK_SIZE(),
                        "chunk_overlap": Config.File.CHUNK_OVERLAP(),
                    }
                }

        except Exception as e:
            logger = __import__('logging').getLogger(__name__)
            logger.error(f"Error processing file {filename}: {e}")
            raise

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.SUPPORTED_EXTENSIONS)

    def is_format_supported(self, filename: str) -> bool:
        """Check if a file format is supported."""
        extension = Path(filename).suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS and self.docling_processor.is_format_supported(filename)

    def is_supported_file(self, filename: str) -> bool:
        """Check if a file is supported (alias for is_format_supported for compatibility)."""
        return self.is_format_supported(filename)

    def get_ocr_status(self) -> bool:
        """Get current OCR fallback status."""
        return getattr(self.docling_processor, 'enable_ocr', False)
