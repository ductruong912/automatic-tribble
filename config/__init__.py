"""
Configuration Package
Centralized configuration management organized by domain.
"""

from .settings import Config

# Legacy imports for backward compatibility
# These map to the centralized Config object
from .settings import Config as LegacyConfig

# Create legacy aliases for backward compatibility
LLMConfig = Config.LLM
ConfidenceConfig = Config.Confidence
ChatConfig = Config.Chat
DoclingConfig = Config.OCR  # OCR config for document processing
PreprocessingConfigManager = Config.OCR
PreprocessingSettings = Config.OCR
FileConfig = Config.File
RAGConfig = Config.RAG
VietnamesePreprocessingConfig = Config.RAG
ServerConfig = Config.Server
LoggingConfig = Config.Logging
HealthConfig = Config.Server  # Health config is part of Server config

__all__ = [
    "Config",
    # Legacy aliases for backward compatibility
    "LLMConfig",
    "ConfidenceConfig", 
    "ChatConfig",
    "DoclingConfig",
    "PreprocessingConfigManager",
    "PreprocessingSettings",
    "FileConfig",
    "RAGConfig",
    "VietnamesePreprocessingConfig",
    "ServerConfig",
    "LoggingConfig",
    "HealthConfig",
]