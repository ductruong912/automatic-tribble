"""
Docling-based document processor for conversion and heading-aware chunking.
Uses Docling's DocumentConverter to convert a local file path, exports markdown,
then chunks by markdown headings (sections). Keeps a similar interface to
MarkItDownProcessor for drop-in replacement.
"""

# Standard library imports
import logging
import re
import tempfile
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
try:
    from docling.document_converter import DocumentConverter
except Exception as _e:  # defer hard failure to runtime path
    DocumentConverter = None  # type: ignore

# Local imports
from models.metadata import MetadataBuilder, ProcessingMethod, SourceType, ProcessingStatus
from config.settings import Config

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """
    Document processor using Docling for multi-format conversion.
    Produces heading-based chunks from the exported markdown.
    """

    def __init__(self, enable_ocr: bool = False, llm_client=None, llm_model: str = None, 
                 preprocessing_config: str = "ocr_optimized"):
        """
        Initialize Docling processor with automatic OCR for PDFs and normal extraction for other formats.
        - PDFs: Always use OCR (like the test file)
        - Other formats: Use normal Docling extraction
        
        Args:
            enable_ocr: Deprecated - OCR is now automatic for PDFs
            llm_client: OpenAI client for enhanced processing
            llm_model: LLM model to use for enhanced processing
            preprocessing_config: Preprocessing configuration name ("ocr_optimized", "fast", "high_quality", "default")
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        
        # Converter reuse for memory efficiency
        self._normal_converter = None
        self._converter_lock = asyncio.Lock()
        
        # Built lazily so a process with no scanned PDFs loads no OCR model.
        self._scanned_extractor = None

    def _clear_memory_caches(self):
        """Release transient Python and CUDA allocations after document processing."""
        try:
            import gc

            gc.collect()

            # Clear only PyTorch's documented CUDA allocator cache. Do not
            # modify process-wide GC thresholds or inspect arbitrary modules.
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.debug(f"Cache clearing warning: {e}")

    def _get_scanned_extractor(self):
        """Get the scanned-PDF extractor, building it (and its OCR engine) on first use."""
        if self._scanned_extractor is None:
            from .scanned_pdf import ScannedPdfExtractor

            self._scanned_extractor = ScannedPdfExtractor()
        return self._scanned_extractor

    async def _get_normal_converter(self):
        """Get or create normal converter (thread-safe). Building it loads models, so keep it off the loop."""
        if self._normal_converter is None:
            async with self._converter_lock:
                if self._normal_converter is None:
                    logger.info("Initializing normal converter...")
                    self._normal_converter = await asyncio.to_thread(DocumentConverter)
                    logger.info(" Normal converter initialized and cached")
        return self._normal_converter

    def _extract_text(self, content: bytes, suffix: str, pdf_profile, converter) -> Dict[str, Any]:
        """
        Convert or OCR one document.

        Blocking from start to finish - always call it through a worker thread so a
        multi-minute OCR run cannot freeze the whole server.
        """
        temp_path = None
        try:
            # Write to a temp file for Docling to consume
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)

            if pdf_profile is not None and not pdf_profile.is_fully_text:
                # Mixed or scanned: read text pages directly, OCR the rest.
                logger.info(
                    f"PDF needs OCR on {len(pdf_profile.scan_pages)} of {pdf_profile.page_count} pages"
                )
                result = self._get_scanned_extractor().extract(content, profile=pdf_profile)
                return {
                    "text": result["text"],
                    "ocr_pages": result["ocr_pages"],
                    "engine": result["engine"],
                    "preprocess_profile": result["preprocess_profile"],
                }

            if pdf_profile is not None:
                # Docling reconstructs tables and headings better than raw text.
                logger.info(
                    f"PDF has a text layer on every page ({pdf_profile.summary()}); skipping OCR"
                )
            else:
                logger.info(f"Processing {suffix} file with normal extraction")

            doc = converter.convert(str(temp_path)).document
            return {
                "text": doc.export_to_markdown(),
                "ocr_pages": [],
                "engine": None,
                "preprocess_profile": None,
            }
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
            self._clear_memory_caches()

    async def process_document(
        self,
        content: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Process document content using Docling with simple OCR configuration.
        Conversion and OCR run in a worker thread so the API keeps answering during an upload.
        """
        start_time = time.time()
        if DocumentConverter is None:
            raise RuntimeError("Docling DocumentConverter is not available. Install 'docling'.")

        suffix = Path(filename).suffix
        is_pdf = suffix.lower() == '.pdf'

        try:
            # Classification parses every page, so it goes to a thread as well
            pdf_profile = None
            if is_pdf:
                from ..pdf_classifier import classify_pdf

                pdf_profile = await asyncio.to_thread(classify_pdf, content)

            # Only needed when Docling does the conversion, not on the OCR path
            converter = None
            if pdf_profile is None or pdf_profile.is_fully_text:
                converter = await self._get_normal_converter()

            extracted = await asyncio.to_thread(
                self._extract_text, content, suffix, pdf_profile, converter
            )
            text_md: str = extracted["text"]
            ocr_pages: List[int] = extracted["ocr_pages"]
            ocr_engine_name: Optional[str] = extracted["engine"]
            preprocess_profile: Optional[str] = extracted["preprocess_profile"]
            ocr_used: bool = bool(ocr_pages)

            # Heading-based chunking
            chunks = self._chunk_markdown_by_headings(text_md)

            # Build metadata
            metadata = (
                MetadataBuilder()
                .set_source_info(
                    source_id=filename,
                    source_name=filename,
                    source_type=SourceType.FILE,
                )
                .set_file_info(file_extension=suffix.lower(), file_size_bytes=len(content))
                .set_processing_info(
                    method=ProcessingMethod.DOCLING,
                    status=ProcessingStatus.SUCCESS,
                    processing_time=time.time() - start_time,
                )
                .set_content_stats(total_chunks=len(chunks), total_characters=len(text_md))
                .set_ocr_info(ocr_enabled=is_pdf, ocr_used=ocr_used)
                .add_custom_metadata("ocr_engine", ocr_engine_name)
                .add_custom_metadata("ocr_pages", ocr_pages)
                .add_custom_metadata("ocr_preprocess_profile", preprocess_profile)
                .add_custom_metadata("pdf_page_count", pdf_profile.page_count if pdf_profile else None)
                .add_custom_metadata(
                    "pdf_text_pages",
                    pdf_profile.page_count - len(pdf_profile.scan_pages) if pdf_profile else None,
                )
                .set_content_features(
                    has_tables="|" in text_md,
                    has_images="![" in text_md,
                    has_links="[" in text_md and "]" in text_md,
                )
                .set_quality_metrics(conversion_success=True)
                .add_custom_metadata("markdown_length", len(text_md))
                .add_custom_metadata("converter_reuse", True)
                .build()
            )

            return {
                "documents": chunks,
                "metadata": metadata.model_dump() if hasattr(metadata, 'model_dump') else metadata.dict() if hasattr(metadata, 'dict') else dict(metadata),
            }
        except Exception as e:
            logger.error(f"Docling processing failed for {filename}: {e}")
            raise
        # Temp file removal and memory cleanup happen inside _extract_text

    def _chunk_markdown_by_headings(self, markdown_text: str) -> List[str]:
        """
        Split markdown by headings, then apply the configured size limit.
        - Recognizes headings starting with '#' (ATX-style) at any level.
        - Each heading and its content becomes one chunk.
        - Large sections are split at sentence boundaries with configured overlap.
        - If no headings are found, the document is treated as one section.
        """
        if not markdown_text:
            return []

        lines = markdown_text.splitlines()
        
        # First, try to chunk by markdown headings
        sections: List[List[str]] = []
        current: List[str] = []
        heading_pattern = re.compile(r"^#{1,6}\s+")

        def push_current():
            if current:
                sections.append(current.copy())
                current.clear()

        for line in lines:
            if heading_pattern.match(line):
                push_current()
                current.append(line)
            else:
                current.append(line)
        push_current()

        # Convert sections to strings and split oversized sections without cutting
        # through the configured chunk boundary.
        from utils.text_processing.text_utils import TextUtils

        chunk_size = Config.File.CHUNK_SIZE()
        chunk_overlap = Config.File.CHUNK_OVERLAP()
        chunks: List[str] = []
        for block in sections:
            # Trim
            while block and not block[0].strip():
                block.pop(0)
            while block and not block[-1].strip():
                block.pop()
            chunk = "\n".join(block).strip()
            if chunk:
                chunks.extend(
                    TextUtils.chunk_text(
                        chunk,
                        chunk_size=chunk_size,
                        overlap=chunk_overlap,
                    )
                )

        # If no headings were found, return the document as one configured section.
        if not chunks:
            chunks = TextUtils.chunk_text(
                markdown_text.strip(),
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

        logger.info(
            "Created %d chunks from headings with size=%d and overlap=%d",
            len(chunks),
            chunk_size,
            chunk_overlap,
        )
        return chunks

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """DEPRECATED: No longer used - chunking is now heading-only."""
        """
        Split text by sentences while preserving text structure.
        """
        if not text or len(text) < 200:
            return [text] if text.strip() else []
        
        # Sentence endings
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period if sentence doesn't end with punctuation
            if sentence and not re.search(r'[.!?]$', sentence):
                sentence += "."
            
            if current_chunk and len(current_chunk) + len(sentence) > 1000:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def get_supported_formats(self) -> List[str]:
        """Expose file types supported by Docling."""
        return [
            '.pdf', '.docx', '.pptx', '.html', '.htm',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
            '.csv', '.xlsx', '.md', '.asciidoc', '.vtt',
            '.mp3', '.wav', '.m4a', '.json'
        ]

    def is_format_supported(self, filename: str) -> bool:
        extension = Path(filename).suffix.lower()
        return extension in self.get_supported_formats()
