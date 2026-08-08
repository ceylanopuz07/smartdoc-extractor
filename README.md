# SmartDoc Extractor

Production-ready SmartDoc Extractor with FastAPI and Docker deployment.

## Overview

SmartDoc Extractor is an AI-powered document extraction system that uses pure LLM-based extraction with enhanced prompts and few-shot learning. It supports:
- **LLM-based Extraction**: Primary extraction using OpenAI GPT-4o-mini with document type-specific prompts
- **Few-shot Learning**: Example-based extraction for improved accuracy
- **OCR Processing**: Tesseract OCR for text extraction from images and PDFs
- **Chat Detection**: Automatic detection and extraction from chat conversations
- **PDF Support**: Direct text extraction and OCR fallback for PDFs
- **FastAPI**: Production REST API
- **Docker**: Easy deployment support

## Features

- **LLM-based Extraction**: Primary extraction using OpenAI GPT-4o-mini with document type-specific prompts
- **Document Type Detection**: Automatic detection of invoices, receipts, contracts, forms, chat conversations, letters, reports, certificates, and statements
- **Few-shot Examples**: Context-aware extraction with example-based learning
- **Enhanced Prompts**: Specialized extraction instructions for each document type
- **OCR Processing**: Tesseract OCR for images and PDFs
- **PDF Support**: Direct text extraction with OCR fallback
- **Chat Detection**: Automatic detection of WhatsApp/chat conversations
- **Multi-format Support**: PNG, JPG, PDF documents
- **REST API**: FastAPI with automatic Swagger documentation
- **Web Interface**: User-friendly drag-and-drop interface
- **Production Ready**: Docker container with health checks
- **Fallback Extraction**: Rule-based extraction when LLM is unavailable

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

### Testing

Run the test suite to verify extraction functionality:
```bash
python test_llm_extraction.py
```

This will test extraction on sample documents (invoice, receipt, chat, contract).

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
- `GET /health` - Health check (OCR and LLM initialization status)
- `GET /docs` - Swagger documentation
- `POST /extract-file` - Extract information from uploaded document file (PNG, JPG, PDF) using LLM-based extraction

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
    "date": ["2024-01-15", "2024-02-15"],
    "amount": ["$500.00", "$500.00", "$1,000.00", "$80.00", "$1,080.00"],
    "names": ["TechCorp Inc.", "ABC Company"],
    "addresses": ["123 Business Ave, Suite 100, San Francisco, CA 94105", "456 Industry Blvd, New York, NY 10001"],
    "phone_numbers": ["+1-555-123-4567"],
    "email_addresses": ["billing@techcorp.com"],
    "reference_numbers": ["INV-2024-001"],
    "total_amount": "$1,080.00",
    "items": [
      {"name": "Software License", "quantity": 5, "unit_price": "$100.00", "total_price": "$500.00"},
      {"name": "Support Services", "quantity": 10, "unit_price": "$50.00", "total_price": "$500.00"}
    ],
    "other_fields": {
      "payment_terms": "Net 30",
      "tax_rate": "8%",
      "bank": "Chase Bank",
      "account": "****1234"
    }
  },
  "success": true,
  "message": "Extraction completed successfully"
}
```

## Project Structure

```
rag-document-extraction/
├── src/
│   ├── api.py                  # FastAPI application
│   ├── llm_extractor.py        # LLM-based extraction with enhanced prompts
│   ├── ocr_processor.py        # OCR processing for images and PDFs
│   ├── config.py               # Configuration
│   ├── rag_extractor.py        # Legacy RAG-enhanced extractor (deprecated)
│   ├── rag_knowledge_base.py   # Legacy RAG knowledge base (deprecated)
│   ├── base_extractor.py       # Legacy ML extraction (deprecated)
│   ├── preprocessing.py        # Legacy text preprocessing (deprecated)
│   ├── download_dataset.py     # Legacy dataset download (deprecated)
│   ├── explore_benchmark.py    # Legacy data exploration (deprecated)
│   └── train_benchmark_models.py # Legacy ML training (deprecated)
├── data/
│   ├── raw/                    # Raw dataset (legacy)
│   ├── processed/              # Processed data (legacy)
│   └── knowledge_base/         # ChromaDB storage (legacy)
├── models/
│   └── ml_models/              # Trained ML models (legacy)
├── web_interface.html         # User interface
├── test_llm_extraction.py     # LLM extraction test suite
├── Dockerfile                  # Docker configuration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Architecture

SmartDoc Extractor uses a pure LLM-based architecture:

1. **OCR Processing**: Extract text from images and PDFs using Tesseract
2. **Document Type Detection**: Automatically detect document type (invoice, receipt, chat, contract, form, letter, report, certificate, statement)
3. **LLM Extraction**: Use OpenAI GPT-4o-mini with:
   - Document type-specific prompts
   - Few-shot examples for common patterns
   - Enhanced extraction rules
4. **Fallback Extraction**: Rule-based extraction when LLM is unavailable

The system no longer uses ML models or RAG enhancement, relying entirely on LLM-based extraction with enhanced prompts for better accuracy and maintainability.

## Extraction Process

1. **OCR Processing**: Extract text from images and PDFs using Tesseract
2. **Document Type Detection**: Automatically detect document type (invoice, receipt, chat, contract, form, letter, report, certificate, statement)
3. **LLM Extraction**: Use OpenAI GPT-4o-mini with document type-specific prompts and few-shot examples
4. **Fallback Extraction**: Rule-based extraction when LLM is unavailable

## Performance

- **LLM Extraction**: High accuracy with GPT-4o-mini and enhanced prompts
- **Document Type Detection**: Automatic detection of 9 document types
- **OCR Processing**: Fast text extraction from images and PDFs
- **API Response**: < 2 seconds for typical documents
- **Document Support**: PNG, JPG, PDF formats
- **Chat Detection**: Automatic conversation extraction
- **Fallback**: Graceful degradation to rule-based extraction

## Development

### Adding New Document Types

To add support for a new document type:

1. Add document type detection logic in `_detect_document_type()` in `src/llm_extractor.py`
2. Add type-specific instructions in `_get_type_specific_instructions()`
3. Add few-shot examples in `_get_few_shot_examples()` if desired
4. Test with `python test_llm_extraction.py`

### Customizing Extraction Fields

Edit the prompt in `_create_extraction_prompt()` in `src/llm_extractor.py` to modify the extraction fields and prompts used by the LLM.

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
