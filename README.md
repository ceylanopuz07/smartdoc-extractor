# SmartDoc Extractor

Production-ready SmartDoc Extractor with FastAPI and Docker deployment.

## Overview

SmartDoc Extractor is an AI-powered document extraction system that uses LLM-based extraction with RAG enhancement. It supports:
- **LLM-based Extraction**: Primary extraction using OpenAI GPT models
- **RAG Enhancement**: Retrieval-augmented generation for improved accuracy
- **OCR Processing**: Tesseract OCR for text extraction from images and PDFs
- **Chat Detection**: Automatic detection and extraction from chat conversations
- **PDF Support**: Direct text extraction and OCR fallback for PDFs
- **FastAPI**: Production REST API
- **Docker**: Easy deployment support

## Features

- **LLM-based Extraction**: Primary extraction using OpenAI GPT-4o-mini
- **RAG Enhancement**: ChromaDB retrieval for improved extraction accuracy
- **OCR Processing**: Tesseract OCR for images and PDFs
- **PDF Support**: Direct text extraction with OCR fallback
- **Chat Detection**: Automatic detection of WhatsApp/chat conversations
- **Multi-format Support**: PNG, JPG, PDF documents
- **REST API**: FastAPI with automatic Swagger documentation
- **Web Interface**: User-friendly drag-and-drop interface
- **Production Ready**: Docker container with health checks

## Installation

### Prerequisites

- Python 3.12+
- Tesseract OCR
- Poppler (for PDF processing)
- OpenAI API Key (for LLM extraction)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/ceylanopuz07/smartdoc-extractor.git
cd smartdoc-extractor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install system dependencies**

For OCR and PDF processing, you need to install:

**macOS:**
```bash
# Install Tesseract OCR
brew install tesseract

# Install Poppler for PDF processing
brew install poppler
```

**Ubuntu/Debian:**
```bash
# Install Tesseract OCR
sudo apt-get install tesseract-ocr

# Install Poppler for PDF processing
sudo apt-get install poppler-utils
```

**Windows:**
- Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Download and install Poppler from: https://github.com/oschwartz10612/poppler-windows

4. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

5. **Configure OpenAI API Key**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

6. **Start the API**
```bash
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

7. **Open Web Interface**
```bash
open web_interface.html
```

### Optional: ML Models and RAG Knowledge Base

If you want to use ML models and RAG knowledge base for enhanced extraction:

1. **Download and prepare data**
```bash
python src/download_dataset.py
python src/explore_benchmark.py
python src/preprocess_benchmark.py
```

2. **Train ML models**
```bash
python src/train_benchmark_models.py
```

3. **Build RAG knowledge base**
```bash
python src/rag_knowledge_base.py
```

### Docker Deployment

1. **Build the Docker image**
```bash
docker build -t rag-document-extraction .
```

2. **Run the container**
```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  rag-document-extraction
```

3. **Or use docker-compose**
```bash
docker-compose up -d
```

## API Usage

### Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Swagger documentation
- `POST /extract-file` - Extract information from uploaded document file (PNG, JPG, PDF)

### Example Request

```bash
curl -X POST http://localhost:8000/extract-file \
  -F "file=@document.pdf"
```

### Example Response

```json
{
  "extraction_results": {
    "document_type": "invoice",
    "date": "2024-01-15",
    "amount": "$30.00",
    "names": ["ABC Company"],
    "email_addresses": ["contact@abc.com"],
    "phone_numbers": ["+1-555-123-4567"],
    "reference_numbers": ["12345"],
    "total_amount": "$30.00",
    "items": ["Widget A - $10.00", "Widget B - $20.00"],
    "other_fields": {}
  },
  "rag_context": [],
  "success": true,
  "message": "Extraction completed successfully"
}
```

## Project Structure

```
rag-document-extraction/
├── src/
│   ├── api.py                  # FastAPI application
│   ├── llm_extractor.py        # LLM-based extraction
│   ├── rag_extractor.py        # RAG-enhanced extractor (optional)
│   ├── ocr_processor.py        # OCR processing for images and PDFs
│   ├── config.py               # Configuration
│   ├── preprocessing.py        # Text preprocessing (optional)
│   ├── base_extractor.py       # ML extraction (optional)
│   ├── download_dataset.py     # Dataset download (optional)
│   ├── explore_benchmark.py    # Data exploration (optional)
│   └── train_benchmark_models.py # ML training (optional)
├── data/
│   ├── raw/                    # Raw dataset (optional)
│   ├── processed/              # Processed data (optional)
│   └── knowledge_base/         # ChromaDB storage (optional)
├── models/
│   └── ml_models/              # Trained ML models (optional)
├── web_interface.html         # User interface
├── Dockerfile                  # Docker configuration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Dataset (Optional)

For enhanced extraction with RAG knowledge base, the system can use the **thoughtworks/document-processing-benchmark** dataset from Hugging Face:
- **Size**: 4,522 documents
- **Sources**: invoices, receipts, forms from 6 different datasets
- **Diversity**: Multiple document types and layouts
- **Fields**: 26 unique extraction fields

This dataset is optional. The system works with LLM-based extraction without it.

## Extraction Process

1. **OCR Processing**: Extract text from images and PDFs using Tesseract
2. **Document Type Detection**: Automatically detect document type (invoice, receipt, chat, etc.)
3. **LLM Extraction**: Use OpenAI GPT-4o-mini for structured extraction
4. **RAG Enhancement** (optional): Retrieve similar documents for improved accuracy
5. **Fallback Extraction**: Rule-based extraction when LLM is unavailable

## Performance

- **LLM Extraction**: High accuracy with GPT-4o-mini
- **OCR Processing**: Fast text extraction from images and PDFs
- **API Response**: < 2 seconds for typical documents
- **Document Support**: PNG, JPG, PDF formats
- **Chat Detection**: Automatic conversation extraction

## Development

### Adding New Document Types

The LLM-based extraction automatically adapts to different document types. For better results with specific document types, you can:

1. Add example documents to the RAG knowledge base
2. Rebuild the RAG knowledge base with new documents
3. The LLM will learn from the context provided by similar documents

### Customizing Extraction Fields

Edit the prompt in `src/llm_extractor.py` to modify the extraction fields and prompts used by the LLM.

## Troubleshooting

### API not starting
- Check if port 8000 is available
- Verify all dependencies are installed
- Ensure Tesseract and Poppler are installed

### OCR not working
- Verify Tesseract is installed and in PATH
- Check that image files are valid
- For PDFs, ensure Poppler is installed

### LLM extraction not working
- Verify OPENAI_API_KEY is set in .env file
- Check your OpenAI API credits
- The system will fall back to rule-based extraction if LLM is unavailable

### PDF upload failing
- Ensure Poppler is installed and in PATH
- Check that PDF files are not corrupted
- Verify the API is running

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or support, please open an issue on GitHub.
