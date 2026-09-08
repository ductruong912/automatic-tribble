#!/usr/bin/env python3
"""
Setup script for the RAG chatbot application.
Handles environment setup, dependency installation, and initial configuration.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or higher is required")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = [
        "data/chunks",
        "data/vectors", 
        "data/temp",
        "data/logs",
        "SSL"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def setup_environment():
    """Create .env from .env.example when it does not exist yet."""
    env_file = Path(".env")
    example = Path(".env.example")
    if env_file.exists():
        print(".env file already exists")
        return
    if not example.exists():
        print("Missing .env.example - cannot create .env")
        return
    env_file.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    print("Created .env from .env.example - please update with your API keys")


def main():
    """Main setup function."""
    print("Setting up RAG Chatbot...")
    print("=" * 50)
    
    check_python_version()
    create_directories()
    setup_environment()
    install_dependencies()
    
    print("=" * 50)
    print("Setup completed successfully!")
    print("Don't forget to:")
    print("   1. Update your .env file with API keys")
    print("   2. Run 'python main.py' to start the server")
    print("   3. Run 'streamlit run app.py' to start the UI")


if __name__ == "__main__":
    main()

