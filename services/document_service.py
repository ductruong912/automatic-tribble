"""
Document processing service for orchestrating file processing operations.
Provides batch processing capabilities with efficient vector store management.
Uses Docling for basic document processing.
"""

# Standard library imports
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
import time

# Local imports
from core.document_processing import MainDocumentProcessor, FileManager
from core.storage.vector_stores import OptimizedVectorStore
from config.settings import Config

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service for document processing orchestration with batch capabilities.
    Handles file processing and vector store management.
    Uses Docling for basic document conversion.
    """

    def __init__(self, processor=None, file_manager=None, enable_ocr: bool = None, llm_client=None, llm_model: str = None, preprocessing_config: str = "ocr_optimized"):
        """
        Initialize document service with processing components.
        - PDFs: Automatically use OCR
        - Other formats: Use normal Docling extraction
        
        Args:
            processor: Main document processor instance
            file_manager: File manager instance
            enable_ocr: Deprecated - OCR is now automatic for PDFs
            llm_client: OpenAI client for enhanced processing
            llm_model: LLM model to use for enhanced processing
            preprocessing_config: Preprocessing configuration name ("ocr_optimized", "fast", "high_quality", "default")
        """
        self.file_manager = file_manager if file_manager is not None else FileManager()
        self.processor = (
            processor
            if processor is not None
            else MainDocumentProcessor(
                file_manager=self.file_manager,
                enable_ocr=False,  # OCR is now automatic based on file type
                llm_client=llm_client,
                llm_model=llm_model,
                preprocessing_config=preprocessing_config
            )
        )
        
        # OCR concurrency control - configurable to prevent memory issues
        max_concurrent_files = min(2, Config.OCR.OCR_MAX_CONCURRENT_FILES())  # Limit to 2 for better memory management
        self._ocr_semaphore = asyncio.Semaphore(max_concurrent_files)
        logger.info(f"OCR concurrency control: max {max_concurrent_files} concurrent PDF files")
        
        # Initialize vector store
        self.vector_store = OptimizedVectorStore()

    async def _process_with_ocr_control(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a single file with OCR concurrency control to prevent memory exhaustion.
        Uses semaphore to limit concurrent OCR operations and aggressive memory management.
        """
        async with self._ocr_semaphore:
            logger.info(f"🔒 [DocumentService] Acquired OCR slot for {filename}")
            try:
                # Clear memory before processing
                await self._aggressive_memory_cleanup()
                
                # Process the file
                result = await self.processor.process_file(file_content, filename)
                
                # Clear memory after OCR processing
                await self._aggressive_memory_cleanup()
                
                logger.info(f"🔓 [DocumentService] Released OCR slot for {filename}")
                return result
            except Exception as e:
                logger.error(f" [DocumentService] OCR processing failed for {filename}: {e}")
                # Clear memory even on failure
                await self._aggressive_memory_cleanup()
                raise

    async def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup to prevent memory exhaustion."""
        try:
            import gc
            import torch
            
            # Clear Python garbage collection
            gc.collect()
            
            # Clear PyTorch GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            # Clear any potential caches in the processor
            if hasattr(self.processor, 'docling_processor'):
                self.processor.docling_processor._clear_memory_caches()
            
            # Small delay to allow memory to be freed
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.debug(f"Memory cleanup warning: {e}")

    async def process_multiple_documents(
        self, file_data_list: List[tuple], rebuild_at_end: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple documents efficiently with batch vector store update.
        Optimizes performance by processing files in memory and updating vector store once.

        Args:
            file_data_list: List of (file_content, filename) tuples
            rebuild_at_end: Whether to rebuild vector store once at the end

        Returns:
            Dict containing batch processing results with success/failure metrics
        """
        start_time = time.time()
        logger.info(
            f"📥 [DocumentService] Received {len(file_data_list)} files for batch processing"
        )
        logger.info(f"🔄 Vector store rebuild at end: {rebuild_at_end}")

        results = []
        successful_documents = []
        successful_count = 0
        failed_count = 0
        total_documents = 0

        # Process files with OCR concurrency control
        async def process_single_file(file_data: tuple) -> Dict[str, Any]:
            """Process a single file with OCR concurrency control."""
            file_content, filename = file_data
            logger.debug(f"🔗 [DocumentService] Processing file: {filename}")
            try:
                # Use OCR concurrency control for PDFs, direct processing for others
                if filename.lower().endswith('.pdf'):
                    result = await self._process_with_ocr_control(file_content, filename)
                else:
                    result = await self.processor.process_file(file_content, filename)
                
                doc_count = len(result["documents"])
                ocr_time = result["metadata"].get("ocr_time")
                logger.info(
                    f" [DocumentService] Successfully processed {filename}: {doc_count} chunks"
                )
                # Log document previews for debugging
                for j, doc in enumerate(result["documents"][:3]):  # Show first 3 chunks
                    preview = doc[:100].replace("\n", " ").replace("\r", " ")
                    logger.debug(f"  Chunk {j+1}: {preview}...")
                
                return {
                    "filename": filename,
                    "success": True,
                    "document_count": doc_count,
                    "metadata": result["metadata"],
                    "process_time": ocr_time,
                    "result": result
                }
            except Exception as e:
                logger.error(f" [DocumentService Error] {str(e)} (file={filename})")
                return {
                    "filename": filename,
                    "success": False,
                    "error": str(e),
                    "document_count": 0,
                    "metadata": {},
                    "process_time": None,
                    "result": None
                }
        
        # Process all files concurrently with OCR control
        logger.info(f" [DocumentService] Processing {len(file_data_list)} files with OCR concurrency control")
        tasks = [process_single_file(file_data) for file_data in file_data_list]
        file_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, file_result in enumerate(file_results):
            if isinstance(file_result, Exception):
                # Handle exceptions from gather
                filename = file_data_list[i][1] if i < len(file_data_list) else "unknown"
                logger.error(f" [DocumentService] Exception processing {filename}: {file_result}")
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(file_result),
                    "document_count": 0,
                    "metadata": {},
                    "process_time": None,
                })
                failed_count += 1
            else:
                # Handle normal results
                if file_result["success"]:
                    successful_documents.append({
                        "documents": file_result["result"]["documents"], 
                        "metadata": file_result["result"]["metadata"]
                    })
                    successful_count += 1
                    total_documents += file_result["document_count"]
                
                results.append(file_result)
                if not file_result["success"]:
                    failed_count += 1

        # Batch update vector store for efficiency
        logger.info(
            f"🔄 [DocumentService] Updating vector store with {len(successful_documents)} documents"
        )
        vector_store_success = self._batch_update_vector_store(
            successful_documents, rebuild_at_end
        )

        if vector_store_success:
            logger.info(
                f" Vector store updated successfully with {total_documents} total document chunks"
            )
        else:
            logger.warning(" Vector store update failed or was skipped")

        response = self._create_batch_response(
            len(file_data_list),
            successful_count,
            failed_count,
            total_documents,
            vector_store_success,
            results,
        )
        response["process_time"] = time.time() - start_time
        return response

    def _batch_update_vector_store(
        self, successful_documents: List[Dict], rebuild_at_end: bool
    ) -> bool:
        """
        Handle batch vector store updates for efficiency.
        Updates vector store once with all successful documents to avoid repeated rebuilds.
        """
        if successful_documents and rebuild_at_end:
            updated = self.vector_store.add_documents_batch(
                successful_documents, rebuild_at_end=True
            )
            if updated:
                self._publish_rebuilt_vector_store()
            return updated
        return False

    def _publish_rebuilt_vector_store(self):
        """
        Hand the freshly rebuilt store to the preloader.
        Without this, chat keeps answering from the copy loaded at startup.
        """
        from utils.performance import get_model_preloader

        preloader = get_model_preloader()
        try:
            preloader.refresh_vector_store(self.vector_store)
        except Exception as e:
            logger.error(f"Could not publish rebuilt vector store: {e}")
            # Fall back to marking it stale so the next chat request reloads from disk
            try:
                preloader.invalidate_vector_store()
            except Exception as inner:
                logger.error(f"Could not invalidate preloaded vector store: {inner}")

    def _create_batch_response(
        self,
        total_files: int,
        successful_files: int,
        failed_files: int,
        total_documents: int,
        vector_store_updated: bool,
        results: List,
    ) -> Dict[str, Any]:
        """
        Create standardized batch processing response with comprehensive metrics.
        Provides consistent response format for batch operations.
        """
        return {
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_documents": total_documents,
            "vector_store_updated": vector_store_updated,
            "results": results,
        }

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return self.processor.get_supported_extensions()

    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported."""
        return self.processor.is_supported_file(filename)
    
    def close(self):
        """Close all resources to prevent leaks."""
        return None