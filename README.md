# RAG Chatbot

A modular Retrieval-Augmented Generation (RAG) chatbot for intelligent Q&A over your documents. Upload files and chat with your data using AI. **Powered by Docling for document conversion, with a pluggable OCR pipeline for scanned PDFs.**

## Architecture

This project follows a **domain-driven design** approach with clear separation of concerns:

### 🏗️ **Domain Organization**

- **AI Services Domain**: LLM integration, embeddings, confidence scoring
- **Document Processing Domain**: File handling, format conversion, text extraction
- **Retrieval Domain**: Search, RAG operations, query processing
- **Storage Domain**: Vector storage, document metadata, caching
- **Infrastructure Domain**: Caching, performance monitoring

### 🔧 **Key Benefits**

- **Maintainability**: Clear domain boundaries make code easier to understand and modify
- **Scalability**: Easy to add new features within appropriate domains
- **Testability**: Each domain can be tested independently
- **Team Development**: Different teams can work on different domains
- **Code Reusability**: Well-organized modules promote code reuse

---

## Features

- **Multi-format Support:** PDF, DOCX, DOC, TXT, CSV, XLSX, XLS, PPTX, PPT, Images, Audio, EPUB, ZIP
- **Docling Integration:** Document conversion to LLM-friendly Markdown format
- **Scan-aware PDF handling:** Pages are classified per page - a real text layer is read directly, only scanned pages go through OCR
- **Pluggable OCR:** Switch engines with `OCR_ENGINE` (`easyocr` for Vietnamese accuracy, `rapidocr` for speed); models download automatically, no external binary to install
- **RAG-Powered Chat:** Context-aware answers from your knowledge base
- **Hybrid Search:** Combines semantic and keyword search
- **Query Expansion:** Improves retrieval with automatic query variations
- **Confidence Scoring:** Quality assessment for every response
- **Batch Processing:** Efficient vector storage and updates
- **Configurable:** 50+ environment variables

---

## Quick Start

**1. Clone & Configure**
```bash
git clone https://github.com/lcduc/automatic-octo-computing-machines.git
cd automatic-octo-computing-machine

# Create environment file
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "OPENAI_MODEL=gpt-4o-mini" >> .env
echo "HOST=0.0.0.0" >> .env
echo "PORT=8500" >> .env
echo "DEBUG=False" >> .env
```

**2. Run**

**Option A: Docker (Recommended)**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application at http://localhost:8500
# API docs at http://localhost:8500/docs
```

**Option B: Local Installation**

**Automated Setup (Windows):**
```bash
# Run the automated setup script
scripts\setup_venv.bat

# Activate environment and run
venv\Scripts\activate
python main.py
```

**Manual Installation:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install PyTorch (choose based on your system)
# CPU only:
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# CUDA 11.8:
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch==2.7.1+cu121 torchvision==0.22.1+cu121 torchaudio==2.7.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

**3. Configure Environment**

Create a `.env` file in the project root with your configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your settings
# Most importantly, set your OpenAI API key:
OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: The `.env.example` file contains all available configuration options with their default values. Copy it to `.env` and customize as needed.

**4. Access the Application**

- **FastAPI Server**: http://localhost:8500
- **API Documentation**: http://localhost:8500/docs
- **Streamlit UI**: `streamlit run app.py` (runs on http://localhost:8501)

**To check your CUDA version:**  
- Run `nvidia-smi` in your terminal  
- Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for more help

---

## API Endpoints

### Health & Status
| Endpoint           | Method | Description                        |
|--------------------|--------|------------------------------------|
| `/`                | GET    | Health check                       |
| `/performance`     | GET    | Performance metrics                |
| `/cache-stats`     | GET    | Cache statistics                   |

### Chat & AI
| Endpoint           | Method | Description                        |
|--------------------|--------|------------------------------------|
| `/chat`            | POST   | Ask questions (RAG-powered)        |

### File Management
| Endpoint           | Method | Description                        |
|--------------------|--------|------------------------------------|
| `/files/upload`    | POST   | Upload files for processing        |

### Models & Maintenance
| Endpoint           | Method | Description                        |
|--------------------|--------|------------------------------------|
| `/models`          | GET    | Model status and availability      |
| `/cleanup/vectors/rebuild` | POST | Rebuild vector store            |
| `/cleanup/query-adapter/update` | POST | Update query adapter         |

### Documentation
| Endpoint           | Method | Description                        |
|--------------------|--------|------------------------------------|
| `/docs`            | GET    | Interactive API documentation      |
| `/redoc`           | GET    | Alternative API documentation      |

---

## Project Structure

```
automatic-octo-computing-machine/
├── api/                            # FastAPI routes and middleware
│   └── routes/                     # API endpoints (chat, files, health)
├── core/                           # Core functionality modules (domain-organized)
│   ├── ai_services/                # AI and ML services
│   │   ├── llm/                    # Language model integration
│   │   ├── embeddings/             # Text embedding generation
│   │   └── confidence/             # Response confidence scoring
│   ├── document_processing/        # Document handling
│   │   ├── processors/             # Format processors (Docling-based)
│   │   ├── extractors/             # Content extractors
│   │   └── managers/               # Processing orchestration
│   ├── retrieval/                  # Search and RAG
│   │   ├── search/                 # Document retrieval
│   │   ├── similarity/             # Similarity calculations
│   │   └── query_expansion/        # Query enhancement
│   ├── storage/                    # Data persistence
│   │   ├── vector_stores/          # Vector storage
│   │   └── metadata_stores/        # Document metadata
│   └── infrastructure/             # System infrastructure
│       └── caching/                # Performance caching
├── config/                         # Configuration management (domain-organized)
│   ├── settings.py                 # Centralized configuration
│   ├── llm/                        # AI services config
│   ├── document_processing/        # Document processing config
│   ├── file/                       # File handling config
│   ├── rag/                        # RAG config
│   └── server/                     # Server config
├── services/                       # Business logic layer
├── utils/                          # Shared utilities (domain-organized)
│   ├── file_operations/            # File management utilities
│   ├── text_processing/            # Text processing utilities
│   └── performance/                # Performance monitoring
├── models/                         # Data models and schemas
├── scripts/                        # Utility scripts
├── data/                           # Runtime data storage
│   ├── chunks/                     # Document chunks
│   ├── vectors/                    # Vector embeddings
│   ├── logs/                       # Application logs
│   └── temp/                       # Temporary files
├── main.py                         # FastAPI application entry point
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Docker orchestration
├── Dockerfile                      # Container definition
└── README.md                       # This file
```

---

## Environment Configuration

The application uses environment variables for configuration. Key variables include:

### 🔑 **Required Variables**
- `OPENAI_API_KEY`: Your OpenAI API key for LLM functionality

### ⚙️ **Optional Variables**
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8500)
- `EMBEDDING_MODEL`: Embedding model (default: paraphrase-multilingual-MiniLM-L12-v2)
- `RERANKER_MODEL`: Reranker model (default: cross-encoder/ms-marco-MiniLM-L6-v2)
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 52428800)
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `SIMILARITY_THRESHOLD`: Search similarity threshold (default: 0.5)

See `.env.example` for a complete list of available configuration options.

---

## Requirements

### System Requirements
- **Python**: 3.10+ (3.11 recommended)
- **RAM**: 8GB+ (16GB recommended for large documents)
- **Storage**: 2GB+ free space
- **GPU**: 8GB+ VRAM NVIDIA GPU (optional, for faster processing)

### API Keys
- **OpenAI API Key**: Required for chat functionality
- **Optional**: Hugging Face token for model downloads

### System Dependencies (for local installation)
- **Windows**: Visual Studio Build Tools (for compiling packages)
- **Linux**: `gcc`, `g++`, `libgl1-mesa-glx`, `poppler-utils`
- **macOS**: Xcode Command Line Tools

### OCR
No external OCR binary is required. Both engines ship as Python packages and
download their models on first use.

Measured on real Vietnamese scans (a contract price table and a delivery note):

| `OCR_ENGINE` | Vietnamese prose | Figures in a price table | Speed per A4 page (CPU) |
|--------------|------------------|--------------------------|--------------------------|
| `easyocr` (default) | 22/33 phrases | 14/19 | ~38s |
| `rapidocr` (PP-OCRv6) | 1/33 phrases | 19/19 | ~7s |

`easyocr` is the default because PP-OCRv6 deletes the letters that carry tone
marks - "thiet ke" comes out as "thit k", which neither a reader nor the
accent-stripped BM25 index can match. Switch to `rapidocr` for forms, invoices
and price tables, where its figures are better and it is five times faster.

Installing a CUDA build of PyTorch lets `easyocr` use the GPU; `OCR_GPU=auto`
picks it up with no further configuration.

---

For advanced configuration, see [CONFIGURATION.md](CONFIGURATION.md).
