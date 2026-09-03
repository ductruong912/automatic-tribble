@echo off
echo ========================================
echo RAG Chatbot - Virtual Environment Setup
echo Domain-Driven Architecture
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo.
echo Step 1: Setting up environment file...
if not exist ".env" (
    echo Creating .env file...
    echo OPENAI_API_KEY=your_api_key_here > .env
    echo OPENAI_MODEL=gpt-4o-mini >> .env
    echo HOST=0.0.0.0 >> .env
    echo PORT=8500 >> .env
    echo DEBUG=False >> .env
    echo ✓ Environment file created
    echo ⚠️  Please edit .env and add your OpenAI API key
) else (
    echo ✓ Environment file already exists
)

echo.
echo Step 2: Checking for existing virtual environment...
if exist "venv" (
    echo ✓ Found existing venv directory - using it
    goto :activate_venv
) else (
    echo Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created successfully
)

:activate_venv
echo.
echo Step 3: Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated

echo.
echo Step 4: Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip tools, continuing anyway...
)

echo.
echo Step 5: Installing PyTorch with CUDA support...
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo.
    echo WARNING: PyTorch CUDA installation failed, trying CPU version...
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install PyTorch
        pause
        exit /b 1
    )
)

echo.
echo Step 6: Installing other dependencies with wheel preference...
pip install --prefer-binary -r requirements.txt
if errorlevel 1 (
    echo.
    echo WARNING: Some dependencies failed to install
    echo Trying alternative installation method...
    pip install --prefer-binary -r requirements.txt -i https://pypi.org/simple
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies
        echo You may need to install Python 3.11 for better compatibility
        echo Or install missing build tools (Rust, Visual Studio Build Tools)
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo 🚀 Quick Start:
echo   python main.py          # Start FastAPI server (http://localhost:8500)
echo   streamlit run app.py    # Start Streamlit UI (http://localhost:8501)
echo.
echo 📝 Don't forget to:
echo   1. Edit .env file and add your OpenAI API key
echo   2. Activate environment: venv\Scripts\activate
echo.
echo 📁 Project Structure:
echo   core/                   # Domain-organized core modules
echo   config/                 # Centralized configuration
echo   api/                    # FastAPI routes
echo   services/               # Business logic
echo   utils/                  # Shared utilities
echo.
echo 🔧 Environment Management:
echo   venv\Scripts\activate   # Activate environment
echo   deactivate              # Deactivate environment
echo.
echo 📖 Documentation:
echo   See README.md for detailed usage instructions
echo.
pause
