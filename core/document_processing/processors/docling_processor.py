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
import os
from typing import List, Dict, Any, Optional
import io
import numpy as np

# Third-party imports
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, 
        TesseractCliOcrOptions
    )
    from docling.document_converter import PdfFormatOption
    # OCR preprocessing imports
    import cv2
    from PIL import Image
    import fitz  # PyMuPDF
except Exception as _e:  # defer hard failure to runtime path
    DocumentConverter = None  # type: ignore
    InputFormat = None  # type: ignore
    PdfPipelineOptions = None  # type: ignore
    TesseractOcrOptions = None  # type: ignore
    TesseractCliOcrOptions = None  # type: ignore
    PdfFormatOption = None  # type: ignore
    cv2 = None  # type: ignore
    Image = None  # type: ignore
    fitz = None  # type: ignore

# Local preprocessing imports
try:
    from .preprocessing import DocumentPreprocessor, create_ocr_optimized_config
    from config.settings import Config
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Advanced preprocessing not available: {e}")
    DocumentPreprocessor = None  # type: ignore
    create_ocr_optimized_config = None  # type: ignore
    PreprocessingConfigManager = None  # type: ignore
    PREPROCESSING_AVAILABLE = False

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
        self._pdf_converter = None
        self._normal_converter = None
        self._converter_lock = asyncio.Lock()
        
        # OCR preprocessing configuration
        self._preprocessing_enabled = cv2 is not None and Image is not None
        
        # Initialize advanced preprocessing
        self._advanced_preprocessing_enabled = PREPROCESSING_AVAILABLE and self._preprocessing_enabled
        self._preprocessor = None
        
        if self._advanced_preprocessing_enabled:
            try:
                if PreprocessingConfigManager:
                    settings = Config.OCR.get_config_by_name(preprocessing_config)
                    config = settings.to_preprocessing_config()
                    self._preprocessor = DocumentPreprocessor(config)
                    logger.info(f"Advanced preprocessing enabled with config: {preprocessing_config}")
                else:
                    self._preprocessor = DocumentPreprocessor()
                    logger.info("Advanced preprocessing enabled with default config")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced preprocessing: {e}")
                self._advanced_preprocessing_enabled = False
        
        try:
            if DocumentConverter is None:
                logger.warning("Docling not available at import time; will raise on first use")
                return
            
            # Set TESSDATA_PREFIX to project tessdata folder
            import os
            project_root = Path(__file__).parent.parent.parent
            tessdata_dir = project_root / "tessdata"
            os.environ['TESSDATA_PREFIX'] = str(tessdata_dir.absolute())
            
            # Also copy traineddata files to the default Tesseract tessdata directory
            tesseract_tessdata_dir = Path(Config.OCR.TESSERACT_CMD()).parent / "tessdata"
            
            # Copy vie.traineddata
            vie_traineddata_src = tessdata_dir / "vie.traineddata"
            vie_traineddata_dst = tesseract_tessdata_dir / "vie.traineddata"
            
            if vie_traineddata_src.exists() and not vie_traineddata_dst.exists():
                try:
                    import shutil
                    shutil.copy2(vie_traineddata_src, vie_traineddata_dst)
                    logger.info(f"Copied vie.traineddata to {vie_traineddata_dst}")
                except Exception as e:
                    logger.warning(f"Failed to copy vie.traineddata: {e}")
            
            # Copy osd.traineddata
            osd_traineddata_src = tessdata_dir / "osd.traineddata"
            osd_traineddata_dst = tesseract_tessdata_dir / "osd.traineddata"
            
            if osd_traineddata_src.exists() and not osd_traineddata_dst.exists():
                try:
                    import shutil
                    shutil.copy2(osd_traineddata_src, osd_traineddata_dst)
                    logger.info(f"Copied osd.traineddata to {osd_traineddata_dst}")
                except Exception as e:
                    logger.warning(f"Failed to copy osd.traineddata: {e}")
            
            # Reset TESSDATA_PREFIX to default to avoid conflicts
            if 'TESSDATA_PREFIX' in os.environ:
                del os.environ['TESSDATA_PREFIX']
            
            # Check if Tesseract is available for PDF OCR
            tesseract_cmd = Config.OCR.TESSERACT_CMD()
            self.tesseract_available = Config.OCR.TESSERACT_CMD_EXISTS()
            
            if self.tesseract_available:
                logger.info(f" Docling processor initialized with OCR support for PDFs")
                logger.info(f"   Tesseract path: {tesseract_cmd}")
                logger.info(f"   Tessdata directory: {project_root}")
            else:
                logger.warning(f"Tesseract not found at {tesseract_cmd}, PDFs will use basic extraction")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Docling configuration: {e}")
            self.tesseract_available = False

    def _clear_memory_caches(self):
        """Clear all memory caches to prevent memory accumulation."""
        try:
            import gc
            import torch
            
            # Clear Python garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            # Clear PyTorch GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection with aggressive settings
            if hasattr(gc, 'set_threshold'):
                gc.set_threshold(100, 10, 10)  # More aggressive garbage collection
            
            # Clear any potential caches in Docling (safer approach)
            try:
                # Only clear caches for known safe objects to avoid torch warnings
                import sys
                for module_name, module in sys.modules.items():
                    if 'docling' in module_name.lower() and hasattr(module, '__dict__'):
                        for attr_name, attr_value in module.__dict__.items():
                            try:
                                if hasattr(attr_value, 'clear_cache') and callable(getattr(attr_value, 'clear_cache')):
                                    attr_value.clear_cache()
                            except (AttributeError, TypeError, Exception):
                                # Skip any problematic objects
                                pass
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Cache clearing warning: {e}")

    async def _get_pdf_converter(self):
        """Get or create PDF converter with OCR (thread-safe)."""
        if self._pdf_converter is None:
            async with self._converter_lock:
                if self._pdf_converter is None:
                    logger.info("Initializing PDF converter with OCR...")
                    # Create PDF converter with OCR
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.do_ocr = True
                    pipeline_options.do_table_structure = True
                    pipeline_options.table_structure_options.do_cell_matching = True

                    # Use the exact same OCR options as the test file
                    ocr_options = TesseractCliOcrOptions(
                        force_full_page_ocr=True, 
                        lang=["vie"],
                        tesseract_cmd=Config.OCR.TESSERACT_CMD()
                    )
                    pipeline_options.ocr_options = ocr_options

                    self._pdf_converter = DocumentConverter(
                        format_options={
                            InputFormat.PDF: PdfFormatOption(
                                pipeline_options=pipeline_options,
                            )
                        }
                    )
                    logger.info(" PDF converter initialized and cached")
        return self._pdf_converter

    async def _get_normal_converter(self):
        """Get or create normal converter (thread-safe)."""
        if self._normal_converter is None:
            async with self._converter_lock:
                if self._normal_converter is None:
                    logger.info("Initializing normal converter...")
                    self._normal_converter = DocumentConverter()
                    logger.info(" Normal converter initialized and cached")
        return self._normal_converter

    def _preprocess_page_for_ocr(self, page_image: np.ndarray) -> np.ndarray:
        """
        Apply memory-efficient OCR preprocessing to a single page.
        Returns preprocessed image optimized for OCR accuracy.
        """
        if not self._preprocessing_enabled:
            return page_image
            
        try:
            # Use advanced preprocessing if available
            if self._advanced_preprocessing_enabled and self._preprocessor:
                logger.debug("Using advanced preprocessing pipeline")
                return self._preprocessor.preprocess_pdf_page(page_image)
            
            # Fallback to basic preprocessing
            logger.debug("Using basic preprocessing pipeline")
            return self._basic_preprocess_page(page_image)
            
        except Exception as e:
            logger.warning(f"OCR preprocessing failed, using original image: {e}")
            return page_image
    
    def _basic_preprocess_page(self, page_image: np.ndarray) -> np.ndarray:
        """
        Basic preprocessing fallback when advanced preprocessing is not available.
        """
        try:
            # 1. Scale to optimal resolution (300 DPI equivalent)
            height, width = page_image.shape[:2]
            target_dpi = 300
            scale_factor = min(target_dpi / 72, 2.0)  # Cap at 2x to prevent memory issues
            
            if scale_factor > 1.1:  # Only scale if significant improvement
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                page_image = cv2.resize(page_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 2. Convert to grayscale if needed
            if len(page_image.shape) == 3:
                gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = page_image.copy()
            
            # 3. Binarize using Otsu's method for automatic threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 4. Correct skew using Hough transform (memory efficient)
            edges = cv2.Canny(binary, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Calculate average angle
                angles = []
                for line in lines[:min(20, len(lines))]:  # Limit to first 20 lines for efficiency
                    rho, theta = line[0]
                    angle = theta - np.pi/2
                    angles.append(angle)
                
                if angles:
                    avg_angle = np.median(angles)
                    # Only correct if angle is significant (> 0.5 degrees)
                    if abs(avg_angle) > 0.0087:  # 0.5 degrees in radians
                        # Rotate image to correct skew
                        h, w = binary.shape
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(avg_angle), 1.0)
                        binary = cv2.warpAffine(binary, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
            
            # 5. Remove noise using morphological operations
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 6. Enhance contrast using CLAHE (memory efficient)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(cleaned)
            
            logger.debug(" Basic OCR preprocessing applied successfully")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Basic OCR preprocessing failed, using original image: {e}")
            return page_image

    def _extract_page_as_image(self, pdf_doc, page_num: int) -> np.ndarray:
        """
        Extract a single page as a numpy array for preprocessing.
        Memory efficient - processes one page at a time.
        """
        try:
            page = pdf_doc[page_num]
            # Render page as image with optimal DPI
            mat = fitz.Matrix(2.0, 2.0)  # 2x scaling for better OCR
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            
            # Clean up immediately
            pix = None
            page = None
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to extract page {page_num} as image: {e}")
            raise

    async def _process_pdf_page_by_page(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process PDFs with concurrent page processing to maximize performance while maintaining memory safety.
        Uses a semaphore to limit concurrent page processing and prevent memory exhaustion.
        """
        start_time = time.time()
        try:
            import fitz  # PyMuPDF
            import tempfile
            from pathlib import Path

            # Open PDF and get page count
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            total_pages = len(pdf_doc)

            logger.info(f"Processing PDF with concurrent pages ({total_pages} pages) with optimized pipeline...")

            # Get reusable converter
            converter = await self._get_pdf_converter()
            logger.info(" Using cached PDF converter with OCR")

            # Create semaphore to limit concurrent page processing (configurable)
            max_concurrent_pages = Config.OCR.OCR_CONCURRENT_PAGES()
            page_semaphore = asyncio.Semaphore(max_concurrent_pages)
            logger.info(f"Using {max_concurrent_pages} concurrent pages per PDF")
            
            async def process_single_page(page_num: int) -> tuple[int, str]:
                """Process a single page with preprocessing and return (page_num, text)"""
                async with page_semaphore:
                    logger.info(f"Processing page {page_num + 1} of {total_pages}")

                    try:
                        # Extract page as image for preprocessing
                        if self._preprocessing_enabled:
                            page_image = self._extract_page_as_image(pdf_doc, page_num)
                            preprocessed_image = self._preprocess_page_for_ocr(page_image)
                            
                            # Convert preprocessed image back to PDF
                            temp_dir = Path(tempfile.gettempdir()) / "docling_ocr"
                            temp_dir.mkdir(exist_ok=True)
                            temp_img_path = temp_dir / f"page_{page_num}_{int(time.time())}_{id(asyncio.current_task())}.png"
                            
                            # Save preprocessed image
                            cv2.imwrite(str(temp_img_path), preprocessed_image)
                            
                            # Convert image to PDF for Docling processing
                            temp_pdf_path = temp_dir / f"page_{page_num}_{int(time.time())}_{id(asyncio.current_task())}.pdf"
                            
                            # Create a simple PDF from the preprocessed image
                            img = Image.open(str(temp_img_path))
                            img.save(str(temp_pdf_path), "PDF", resolution=300.0)
                            
                            # Clean up image file
                            temp_img_path.unlink(missing_ok=True)
                            
                            # Process the preprocessed PDF
                            doc = converter.convert(str(temp_pdf_path)).document
                            page_text = doc.export_to_markdown()
                            
                            # Clean up PDF file
                            temp_pdf_path.unlink(missing_ok=True)
                            
                        else:
                            # Fallback to original method without preprocessing
                            temp_pdf = fitz.open()
                            temp_pdf.insert_pdf(pdf_doc, from_page=page_num, to_page=page_num)

                            temp_dir = Path(tempfile.gettempdir()) / "docling_ocr"
                            temp_dir.mkdir(exist_ok=True)
                            temp_path = temp_dir / f"page_{page_num}_{int(time.time())}_{id(asyncio.current_task())}.pdf"
                            temp_pdf.save(str(temp_path))
                            temp_pdf.close()

                            doc = converter.convert(str(temp_path)).document
                            page_text = doc.export_to_markdown()
                            
                            temp_path.unlink(missing_ok=True)

                        # Clear memory after each page
                        self._clear_memory_caches()
                        logger.debug(f"Memory cleared after processing page {page_num + 1}")

                        return (page_num, page_text)

                    except Exception as e:
                        logger.error(f"Failed to process page {page_num + 1}: {e}")
                        return (page_num, "")

            # Process all pages concurrently
            tasks = [process_single_page(page_num) for page_num in range(total_pages)]
            page_results = await asyncio.gather(*tasks, return_exceptions=True)

            pdf_doc.close()

            # Combine results in correct order
            total_text = ""
            processed_pages = 0
            
            for result in page_results:
                if isinstance(result, Exception):
                    logger.error(f"Page processing failed: {result}")
                    continue
                    
                page_num, page_text = result
                if page_text.strip():
                    total_text += f"\n\n--- Page {page_num + 1} ---\n\n" + page_text
                    processed_pages += 1

            # Chunk the combined text
            chunks = self._chunk_markdown_by_headings(total_text)

            logger.info(f"Concurrent page processing completed: {processed_pages}/{total_pages} pages processed")

            # Build proper metadata
            from models.metadata import MetadataBuilder, ProcessingMethod, SourceType, ProcessingStatus

            metadata = (
                MetadataBuilder()
                .set_source_info(
                    source_id=filename,
                    source_name=filename,
                    source_type=SourceType.FILE,
                )
                .set_file_info(file_extension=".pdf", file_size_bytes=len(content))
                .set_processing_info(
                    method=ProcessingMethod.DOCLING,
                    status=ProcessingStatus.SUCCESS,
                    processing_time=time.time() - start_time,
                )
                .set_content_stats(total_chunks=len(chunks), total_characters=len(total_text))
                .set_ocr_info(ocr_enabled=True, ocr_used=True)
                .set_content_features(
                    has_tables="|" in total_text,
                    has_images="![" in total_text,
                    has_links="[" in total_text and "]" in total_text,
                )
            ).build()

            # Add custom page-by-page info to metadata dict
            metadata_dict = metadata.model_dump() if hasattr(metadata, 'model_dump') else metadata.dict() if hasattr(metadata, 'dict') else dict(metadata)
            metadata_dict["total_pages"] = total_pages
            metadata_dict["processed_pages"] = processed_pages
            metadata_dict["processing_method"] = "concurrent_page_ocr"
            metadata_dict["concurrent_pages"] = max_concurrent_pages
            metadata_dict["concurrency_config"] = {
                "pages_per_pdf": max_concurrent_pages,
                "max_concurrent_files": Config.OCR.OCR_MAX_CONCURRENT_FILES()
            }
            metadata_dict["preprocessing_enabled"] = self._preprocessing_enabled
            metadata_dict["advanced_preprocessing_enabled"] = self._advanced_preprocessing_enabled
            metadata_dict["ocr_forced_all_pdfs"] = Config.OCR.OCR_FORCE_ALL_PDFS()
            metadata_dict["converter_reuse"] = True

            return {
                "documents": chunks,
                "metadata": metadata_dict
            }

        except Exception as e:
            logger.error(f"Concurrent page PDF processing failed: {e}")
            raise

    async def _process_large_pdf_in_chunks(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process large PDFs by splitting them into smaller chunks to prevent memory issues.
        This is a fallback method for PDFs that are too large for normal processing.
        """
        try:
            import fitz  # PyMuPDF
            import tempfile
            from pathlib import Path
            
            # Open PDF and get page count
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            total_pages = len(pdf_doc)
            
            logger.info(f"Large PDF detected ({total_pages} pages), processing in chunks...")
            
            # Process in chunks of 3 pages to prevent memory issues (reduced from 5)
            chunk_pages = 3
            all_chunks = []
            total_text = ""
            
            for start_page in range(0, total_pages, chunk_pages):
                end_page = min(start_page + chunk_pages, total_pages)
                
                logger.info(f"Processing pages {start_page + 1}-{end_page} of {total_pages}")
                
                # Create a temporary PDF with only these pages
                temp_pdf = fitz.open()
                temp_pdf.insert_pdf(pdf_doc, from_page=start_page, to_page=end_page - 1)
                
                # Save temporary PDF to a more accessible location
                temp_dir = Path(tempfile.gettempdir()) / "docling_ocr"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / f"chunk_{start_page}_{end_page}_{int(time.time())}.pdf"
                temp_pdf.save(str(temp_path))
                
                temp_pdf.close()
                
                try:
                    # Process this chunk with Docling
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.do_ocr = True
                    pipeline_options.do_table_structure = True  # Keep table structure for proper table extraction
                    pipeline_options.table_structure_options.do_cell_matching = True

                    ocr_options = TesseractCliOcrOptions(
                        force_full_page_ocr=True,
                        lang=["vie"],
                        tesseract_cmd=Config.OCR.TESSERACT_CMD()
                    )
                    pipeline_options.ocr_options = ocr_options

                    converter = DocumentConverter(
                        format_options={
                            InputFormat.PDF: PdfFormatOption(
                                pipeline_options=pipeline_options,
                            )
                        }
                    )
                    
                    # Process this chunk
                    doc = converter.convert(str(temp_path)).document
                    chunk_text = doc.export_to_markdown()
                    total_text += f"\n\n--- Page {start_page + 1}-{end_page} ---\n\n" + chunk_text
                    
                    # Clear memory after each chunk
                    self._clear_memory_caches()
                    logger.debug(f"Memory cleared after processing pages {start_page + 1}-{end_page}")
                    
                finally:
                    # Clean up temporary file
                    try:
                        if temp_path.exists():
                            temp_path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.debug(f"Could not remove temp file {temp_path}: {e}")
            
            pdf_doc.close()
            
            # Chunk the combined text
            chunks = self._chunk_markdown_by_headings(total_text)
            
            return {
                "documents": chunks,
                "metadata": {
                    "total_pages": total_pages,
                    "chunks_processed": (total_pages + chunk_pages - 1) // chunk_pages,
                    "processing_method": "chunked_ocr"
                }
            }
            
        except Exception as e:
            logger.error(f"Chunked PDF processing failed: {e}")
            raise

    async def process_document(
        self,
        content: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Process document content using Docling with simple OCR configuration.
        Uses the exact same approach as the test file for optimal OCR quality.
        """
        start_time = time.time()
        if DocumentConverter is None:
            raise RuntimeError("Docling DocumentConverter is not available. Install 'docling'.")

        # Write to a temp file for Docling to consume
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        text_md: str = ""
        ocr_used: bool = False
        
        try:
            # Determine processing method based on file type
            is_pdf = suffix.lower() == '.pdf'
            
            if is_pdf and self.tesseract_available and TesseractCliOcrOptions and PdfPipelineOptions and InputFormat and PdfFormatOption:
                # Check if we should force OCR on all PDFs
                force_ocr_all = Config.OCR.OCR_FORCE_ALL_PDFS()
                
                if force_ocr_all:
                    # Force OCR on all PDFs regardless of text content
                    logger.info("Processing PDF with forced OCR (OCR_FORCE_ALL_PDFS=true)...")
                    
                    # Get reusable PDF converter
                    converter = await self._get_pdf_converter()
                    
                    doc = converter.convert(str(temp_path)).document
                    text_md = doc.export_to_markdown()
                    ocr_used = True
                    
                    # Clear memory immediately after OCR processing
                    self._clear_memory_caches()
                    logger.debug("Memory cleared after forced OCR processing")
                    
                else:
                    # Use direct PDF processing with converter reuse (original behavior)
                    logger.info("Processing PDF with direct OCR pipeline using cached converter")
                    
                    # Get reusable PDF converter
                    converter = await self._get_pdf_converter()
                    
                    logger.info("Processing PDF with OCR enabled (direct method for best quality)...")
                    doc = converter.convert(str(temp_path)).document
                    text_md = doc.export_to_markdown()
                    ocr_used = True
                    
                    # Clear memory immediately after OCR processing
                    self._clear_memory_caches()
                    logger.debug("Memory cleared after OCR processing")
                
            else:
                # Other file types or PDF without Tesseract - use normal Docling extraction
                converter = await self._get_normal_converter()
                doc = converter.convert(str(temp_path)).document
                text_md = doc.export_to_markdown()
                ocr_used = False
                
                if is_pdf:
                    logger.info("Processing PDF with normal extraction (Tesseract not available)")
                else:
                    logger.info(f"Processing {suffix} file with normal extraction")

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
                .set_ocr_info(ocr_enabled=is_pdf and self.tesseract_available, ocr_used=ocr_used)
                .add_custom_metadata("ocr_forced_all_pdfs", Config.OCR.OCR_FORCE_ALL_PDFS())
                .set_content_features(
                    has_tables="|" in text_md,
                    has_images="![" in text_md,
                    has_links="[" in text_md and "]" in text_md,
                )
                .set_quality_metrics(conversion_success=True)
                .add_custom_metadata("markdown_length", len(text_md))
                .add_custom_metadata("preprocessing_enabled", self._preprocessing_enabled)
                .add_custom_metadata("advanced_preprocessing_enabled", self._advanced_preprocessing_enabled)
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
        finally:
            try:
                temp_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
            # Clear memory after processing
            self._clear_memory_caches()

    def _chunk_markdown_by_headings(self, markdown_text: str) -> List[str]:
        """
        Split markdown by headings only - no size limits, no fallback chunking.
        - Recognizes headings starting with '#' (ATX-style) at any level.
        - Each heading and its content becomes one chunk.
        - No size limits - keeps content together under each heading.
        - If no headings found, returns entire document as one chunk.
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

        # Convert sections to strings, trimming leading/trailing blank lines
        chunks: List[str] = []
        for block in sections:
            # Trim
            while block and not block[0].strip():
                block.pop(0)
            while block and not block[-1].strip():
                block.pop()
            chunk = "\n".join(block).strip()
            if chunk:
                chunks.append(chunk)

        # If no headings found, return the entire document as one chunk
        if not chunks:
            chunks = [markdown_text.strip()]

        logger.info(f"Created {len(chunks)} chunks based on headings only (no size limits)")
        return chunks

    def _chunk_text_intelligently(self, text: str) -> List[str]:
        """DEPRECATED: No longer used - chunking is now heading-only."""
        """
        Intelligently chunk text by paragraphs, sentences, and structure.
        Preserves text flow and prevents single-line chunks.
        """
        if not text or len(text.strip()) < 100:
            return [text.strip()] if text.strip() else []

        # First, try to split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would make chunk too large, start a new chunk
            if current_chunk and len(current_chunk) + len(paragraph) > 1500:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If we still have very large chunks, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 2000:
                sentence_chunks = self._chunk_by_sentences(chunk)
                final_chunks.extend(sentence_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks

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


class AsyncDoclingProcessor(DoclingProcessor):
    """Async wrapper to align with existing async pipeline interfaces."""

    async def process(self, content: bytes, filename: Optional[str] = None) -> List[str]:
        result = await self.process_document(content, filename or "unknown")
        return result.get("documents", [])