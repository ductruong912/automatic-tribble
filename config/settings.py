"""
Centralized configuration management for the RAG chatbot application.
Consolidates all configuration classes into a single, easy-to-manage module.
"""

import os
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig:
    """Database and storage related configuration."""

    @staticmethod
    def CHUNKS_DIR():
        return os.getenv("CHUNKS_DIR", "data/chunks")

    @staticmethod
    def VECTORS_DIR():
        return os.getenv("VECTORS_DIR", "data/vectors")

    @staticmethod
    def TEMP_DIR():
        return os.getenv("TEMP_DIR", "data/temp")

    @staticmethod
    def VECTOR_STORE_PATH():
        return os.getenv("VECTOR_STORE_PATH", "data/vectors/vector_store.pkl")

    @staticmethod
    def LOG_DIR():
        return os.getenv("LOG_DIR", "data/logs")


class LLMConfig:
    """LLM and AI model configuration."""

    @staticmethod
    def OPENAI_API_KEY():
        return os.getenv("OPENAI_API_KEY")

    @staticmethod
    def OPENAI_MODEL():
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @staticmethod
    def EMBEDDING_MODEL():
        return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    @staticmethod
    def RERANKER_MODEL():
        return os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    @staticmethod
    def MAX_TOKENS():
        return int(os.getenv("MAX_TOKENS", "4000"))

    @staticmethod
    def TEMPERATURE():
        return float(os.getenv("TEMPERATURE", "0.1"))

    @staticmethod
    def HISTORY_LENGTH():
        return int(os.getenv("LLM_HISTORY_LENGTH", "9"))

    @staticmethod
    def MAX_CONTEXT_LENGTH():
        return int(os.getenv("MAX_CONTEXT_LENGTH", "5000"))

    @staticmethod
    def OPENAI_MAX_TOKENS():
        return int(os.getenv("OPENAI_MAX_TOKENS", "4000"))

    @staticmethod
    def OPENAI_TEMPERATURE():
        return float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

    @staticmethod
    def OPENAI_TIMEOUT():
        return int(os.getenv("OPENAI_TIMEOUT", "30"))

    @staticmethod
    def LLM_CACHE_TTL():
        return int(os.getenv("LLM_CACHE_TTL", "3600"))

    @staticmethod
    def LLM_CACHE_MAX_ENTRIES():
        return int(os.getenv("LLM_CACHE_MAX_ENTRIES", "1000"))

    @staticmethod
    def LLM_CACHE_MAX_SIZE():
        return int(os.getenv("LLM_CACHE_MAX_SIZE", "100"))

    @staticmethod
    def LLM_MAX_WORKERS():
        return int(os.getenv("LLM_MAX_WORKERS", "5"))

    @staticmethod
    def LLM_HISTORY_LENGTH():
        return int(os.getenv("LLM_HISTORY_LENGTH", "9"))


class FileConfig:
    """File processing and upload configuration."""

    @staticmethod
    def MAX_FILE_SIZE():
        return int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB

    @staticmethod
    def ALLOWED_EXTENSIONS():
        return os.getenv(
            "ALLOWED_EXTENSIONS", ".txt,.pdf,.docx,.csv,.xlsx"
        ).split(",")

    @staticmethod
    def MAX_FILES_PER_BATCH():
        return int(os.getenv("MAX_FILES_PER_BATCH", "10"))

    @staticmethod
    def MAX_TOTAL_BATCH_SIZE():
        return int(os.getenv("MAX_TOTAL_BATCH_SIZE", "209715200"))  # 200MB

    @staticmethod
    def CHUNK_SIZE():
        return int(os.getenv("CHUNK_SIZE", "1000"))

    @staticmethod
    def CHUNK_OVERLAP():
        return int(os.getenv("CHUNK_OVERLAP", "0"))

    @staticmethod
    def TEMP_DIR():
        return os.getenv("TEMP_DIR", "data/temp")

    @staticmethod
    def CHUNKS_DIR():
        return os.getenv("CHUNKS_DIR", "data/chunks")

    @staticmethod
    def VECTORS_DIR():
        return os.getenv("VECTORS_DIR", "data/vectors")

    @staticmethod
    def VECTOR_STORE_PATH():
        return os.getenv("VECTOR_STORE_PATH", "data/vectors/vector_store.pkl")


class ServerConfig:
    """Server and API configuration."""

    @staticmethod
    def HOST():
        return os.getenv("HOST", "0.0.0.0")

    @staticmethod
    def PORT():
        return int(os.getenv("PORT", "8500"))

    @staticmethod
    def DEBUG():
        return os.getenv("DEBUG", "False").lower() == "true"

    @staticmethod
    def UVICORN_WORKERS():
        return int(os.getenv("UVICORN_WORKERS", "1"))

    @staticmethod
    def CORS_ORIGINS():
        origins = os.getenv("CORS_ORIGINS", "*")
        return origins.split(",") if origins != "*" else ["*"]

    @staticmethod
    def CORS_ALLOW_CREDENTIALS():
        return os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true"


class RAGConfig:
    """RAG and retrieval configuration."""

    @staticmethod
    def TOP_K_RESULTS():
        return int(os.getenv("TOP_K_RESULTS", "5"))

    @staticmethod
    def SIMILARITY_THRESHOLD():
        return float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

    @staticmethod
    def QUERY_EXPANSION_ENABLED():
        return os.getenv("QUERY_EXPANSION_ENABLED", "True").lower() == "true"

    @staticmethod
    def RERANKING_ENABLED():
        return os.getenv("RERANKING_ENABLED", "True").lower() == "true"

    @staticmethod
    def QUERY_ADAPTER_PATH():
        return os.getenv("QUERY_ADAPTER_PATH", "data/query_adapter.pkl")

    @staticmethod
    def RETRIEVAL_TOP_K():
        return int(os.getenv("RETRIEVAL_TOP_K", "3"))

    @staticmethod
    def SEMANTIC_WEIGHT():
        return float(os.getenv("SEMANTIC_WEIGHT", "0.5"))

    @staticmethod
    def MAX_CONTEXT_CHUNKS():
        return int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))

    @staticmethod
    def EMBEDDING_MODEL():
        return os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    @staticmethod
    def RERANKER_MODEL():
        return os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

    @staticmethod
    def VIETNAMESE_PREPROCESSING_ENABLED():
        return os.getenv("VIETNAMESE_PREPROCESSING_ENABLED", "True").lower() == "true"

    @staticmethod
    def CLEAN_SPECIAL_CHARS():
        return os.getenv("CLEAN_SPECIAL_CHARS", "True").lower() == "true"

    @staticmethod
    def EXTRACT_CONTENT_WORDS():
        return os.getenv("EXTRACT_CONTENT_WORDS", "True").lower() == "true"

    @staticmethod
    def CONTEXT_EXPANSION_ENABLED():
        return os.getenv("CONTEXT_EXPANSION_ENABLED", "True").lower() == "true"

    @staticmethod
    def CONTEXT_EXPANSION_RADIUS():
        return int(os.getenv("CONTEXT_EXPANSION_RADIUS", "1"))

    @staticmethod
    def MIN_CONTEXT_CHUNKS():
        return int(os.getenv("MIN_CONTEXT_CHUNKS", "2"))

    @staticmethod
    def USE_FAISS_INDEX():
        return os.getenv("USE_FAISS_INDEX", "True").lower() == "true"


class ChatConfig:
    """Chat functionality configuration."""

    @staticmethod
    def CHAT_MODE():
        return os.getenv("CHAT_MODE", "query_only")

    @staticmethod
    def ENABLE_HISTORY():
        return ChatConfig.CHAT_MODE().lower() == "with_history"

    @staticmethod
    def MAX_HISTORY_LENGTH():
        return int(os.getenv("MAX_HISTORY_LENGTH", "10"))


class OCRConfig:
    """OCR and document processing configuration."""

    @staticmethod
    def DOCLING_OCR_ENABLED():
        return os.getenv("DOCLING_OCR_ENABLED", "false").lower() == "true"

    @staticmethod
    def TESSERACT_CMD():
        return os.getenv(
            "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    @staticmethod
    def TESSERACT_TESSDATA_DIR():
        return os.getenv(
            "TESSERACT_TESSDATA_DIR", str(Path(__file__).parent.parent / "tessdata")
        )

    @staticmethod
    def OCR_FORCE_FULL_PAGE():
        return os.getenv("OCR_FORCE_FULL_PAGE", "true").lower() == "true"

    @staticmethod
    def OCR_FORCE_ALL_PDFS():
        return os.getenv("OCR_FORCE_ALL_PDFS", "false").lower() == "true"

    @staticmethod
    def OCR_CONCURRENT_PAGES():
        return int(os.getenv("OCR_CONCURRENT_PAGES", "2"))

    @staticmethod
    def OCR_MAX_CONCURRENT_FILES():
        return int(os.getenv("OCR_MAX_CONCURRENT_FILES", "1"))

    @staticmethod
    def TESSERACT_CMD_EXISTS():
        import shutil

        return shutil.which(OCRConfig.TESSERACT_CMD()) is not None

    @staticmethod
    def get_config_by_name(config_name: str):
        """Legacy method for backward compatibility."""
        return {"name": config_name}


class LoggingConfig:
    """Logging configuration."""

    @staticmethod
    def LOG_LEVEL():
        return os.getenv("LOG_LEVEL", "INFO")

    @staticmethod
    def LOG_TO_FILE():
        return os.getenv("LOG_TO_FILE", "True").lower() == "true"

    @staticmethod
    def LOG_MAX_SIZE():
        return int(os.getenv("LOG_MAX_SIZE", "10485760"))  # 10MB

    @staticmethod
    def LOG_DIR():
        return os.getenv("LOG_DIR", "data/logs")

    @staticmethod
    def LOG_BACKUP_COUNT():
        return int(os.getenv("LOG_BACKUP_COUNT", "5"))


class ConfidenceConfig:
    """Confidence scoring configuration."""

    @staticmethod
    def CONFIDENCE_SCORING_ENABLED():
        return os.getenv("CONFIDENCE_SCORING_ENABLED", "True").lower() == "true"

    @staticmethod
    def CONFIDENCE_CONTEXT_WEIGHT():
        return float(os.getenv("CONFIDENCE_CONTEXT_WEIGHT", "0.35"))

    @staticmethod
    def CONFIDENCE_LENGTH_WEIGHT():
        return float(os.getenv("CONFIDENCE_LENGTH_WEIGHT", "0.20"))

    @staticmethod
    def CONFIDENCE_COHERENCE_WEIGHT():
        return float(os.getenv("CONFIDENCE_COHERENCE_WEIGHT", "0.25"))

    @staticmethod
    def CONFIDENCE_CITATION_WEIGHT():
        return float(os.getenv("CONFIDENCE_CITATION_WEIGHT", "0.10"))

    @staticmethod
    def CONFIDENCE_UNCERTAINTY_WEIGHT():
        return float(os.getenv("CONFIDENCE_UNCERTAINTY_WEIGHT", "0.10"))


# Centralized configuration class for easy access
class HealthConfig:
    """Health monitoring configuration."""

    @staticmethod
    def SERVICE_SUCCESS_RATE_THRESHOLD():
        return float(os.getenv("SERVICE_SUCCESS_RATE_THRESHOLD", "80.0"))

    @staticmethod
    def SERVICE_MIN_REQUESTS_FOR_HEALTH():
        return int(os.getenv("SERVICE_MIN_REQUESTS_FOR_HEALTH", "10"))


class Config:
    """Centralized configuration access point."""

    Database = DatabaseConfig
    LLM = LLMConfig
    File = FileConfig
    Server = ServerConfig
    RAG = RAGConfig
    Chat = ChatConfig
    OCR = OCRConfig
    Logging = LoggingConfig
    Confidence = ConfidenceConfig
    Health = HealthConfig

    @staticmethod
    def validate():
        """Validate critical configuration settings."""
        logger = __import__("logging").getLogger(__name__)

        # Check for required OpenAI API key
        if not Config.LLM.OPENAI_API_KEY():
            logger.warning(
                " Warning: OPENAI_API_KEY not set. Chat functionality may not work."
            )
            return False

        # Create necessary directories
        from utils.file_operations.file_manager import FileManager

        directories = [
            DatabaseConfig.CHUNKS_DIR(),
            DatabaseConfig.VECTORS_DIR(),
            DatabaseConfig.TEMP_DIR(),
        ]
        for directory in directories:
            FileManager.ensure_directory_exists(directory)

        # Ensure parent directory for vector store exists
        FileManager.ensure_directory_exists(
            os.path.dirname(DatabaseConfig.VECTOR_STORE_PATH())
        )
        return True
