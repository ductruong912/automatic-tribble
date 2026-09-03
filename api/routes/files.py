"""
File upload endpoints for knowledge base ingestion.
"""

# Standard library imports
import logging
from typing import List

# Third-party imports
from fastapi import APIRouter, UploadFile, File, Depends

# Local imports
from api.dependencies import get_upload_service
from services import UploadService
from models.responses import MultipleFileUploadResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=MultipleFileUploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    upload_service: UploadService = Depends(get_upload_service),
):
    """
    Batch file upload endpoint for knowledge base ingestion.
    Supports multiple file formats: PDF, DOCX, TXT, CSV, XLSX, XLS.
    Applies file size limits and batch processing constraints.
    """
    return await upload_service.process_file_uploads(files)
