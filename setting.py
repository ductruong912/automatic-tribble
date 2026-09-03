# Standard library imports
import os
from dotenv import load_dotenv

# Load environment variables first, before importing any config classes
load_dotenv()

# Import centralized configuration
from config.settings import Config

# Note: OpenAI client is now initialized per-request in the chatbot service
# This avoids the deprecated openai.api_key global configuration

# Directory validation utility
from utils.file_operations.file_manager import FileManager


def validate_config():
    """Validate critical configuration settings and create necessary directories."""
    return Config.validate()


# Validate configuration on import
validate_config()

# Legacy compatibility - will be removed in future versions
class LegacyConfig:
    """Legacy configuration wrapper for backward compatibility."""
    LLMConfig = Config.LLM
    ServerConfig = Config.Server
    LoggingConfig = Config.Logging
    FileConfig = Config.File
    RAGConfig = Config.RAG
    VietnamesePreprocessingConfig = Config.RAG
    ConfidenceConfig = Config.Confidence
    HealthConfig = Config.Server
    ChatConfig = Config.Chat
