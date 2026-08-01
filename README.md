# RAG Document Extraction API

Production-ready RAG-enhanced document extraction system with FastAPI and Docker deployment.

## Overview

This system combines machine learning predictions with retrieval-augmented generation (RAG) to extract structured information from documents. It uses:
- **ChromaDB** for vector storage and similarity search
- **Sentence Transformers** for document embeddings
- **Random Forest** models for ML-based extraction
- **FastAPI** for production REST API
- **Docker** for easy deployment

## Features

- **RAG Knowledge Base**: Indexed 4,522 documents from diverse sources (invoices, receipts, forms)
- **ML Extraction**: Trained Random Forest models on 10 extraction fields
- **RAG Enhancement**: Retrieves similar documents to improve extraction accuracy
- **REST API**: FastAPI with automatic Swagger documentation
- **Production Ready**: Docker container with health checks
- **Scalable**: Can handle multiple document types

## Installation

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/ceylanopuz07/rag-document-extraction.git
cd rag-document-extraction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download and prepare data**
```bash
python src/download_dataset.py
python src/explore_benchmark.py
python src/preprocess_benchmark.py
```

5. **Train ML models**
```bash
python src/train_benchmark_models.py
```

6. **Build RAG knowledge base**
```bash
python src/rag_knowledge_base.py
```

7. **Start the API**
```bash
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
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
- `POST /extract` - Extract information from document

### Example Request

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "doc_type": "receipt",
    "image_w_px": 432,
    "image_h_px": 648,
    "image_bytes_len": 50000,
    "gt_token_count_cl100k": 100
  }'
```

### Example Response

```json
{
  "ml_results": {
    "menu_nm": 0.0,
    "menu_price": 0.0,
    "total_total_price": 0.0,
    ...
  },
  "rag_context": [
    {
      "labels": {
        "menu_nm": "Kupon 3",
        "menu_price": "28,636",
        "total_total_price": "31,500",
        ...
      },
      "distance": 0.062,
      "metadata": {
        "doc_type": "receipt",
        "source_dataset": "cord_v2"
      }
    }
  ],
  "enhanced_results": {
    "menu_nm": 0.0,
    "menu_price": 0.0,
    "total_total_price": 0.0,
    "meta_split_rag": "test",
    "meta_version_rag": "2.0.0"
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
│   ├── rag_knowledge_base.py   # ChromaDB knowledge base
│   ├── rag_extractor.py        # RAG-enhanced extractor
│   ├── config.py               # Configuration
│   ├── preprocessing.py        # Text preprocessing
│   ├── base_extractor.py       # ML extraction
│   ├── download_dataset.py     # Dataset download
│   ├── explore_benchmark.py    # Data exploration
│   └── train_benchmark_models.py # ML training
├── data/
│   ├── raw/                    # Raw dataset
│   ├── processed/              # Processed data
│   └── knowledge_base/         # ChromaDB storage
├── models/
│   └── ml_models/              # Trained ML models
├── notebooks/                  # Jupyter notebooks
├── Dockerfile                  # Docker configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Dataset

The system uses the **thoughtworks/document-processing-benchmark** dataset from Hugging Face:
- **Size**: 4,522 documents
- **Sources**: invoices, receipts, forms from 6 different datasets
- **Diversity**: Multiple document types and layouts
- **Fields**: 26 unique extraction fields

## RAG Enhancement Process

1. **ML Extraction**: Random Forest models predict extraction fields
2. **Query Generation**: ML results are converted to query text
3. **Similarity Search**: ChromaDB retrieves top-k similar documents
4. **Context Enhancement**: Retrieved document labels enhance predictions
5. **Combined Output**: Returns ML-only and RAG-enhanced results

## Performance

- **ML Models**: Trained on 4,522 documents
- **RAG Retrieval**: Sub-second similarity search
- **API Response**: < 1 second for typical requests
- **Knowledge Base**: 4,522 indexed documents

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Document Types

1. Add documents to the dataset
2. Re-run preprocessing
3. Retrain ML models
4. Rebuild RAG knowledge base

### Customizing Extraction Fields

Edit `src/config.py` to modify target fields and model parameters.

## Troubleshooting

### API not starting
- Check if port 8000 is available
- Verify all dependencies are installed
- Ensure data and models directories exist

### RAG retrieval not working
- Verify ChromaDB knowledge base is built
- Check if documents are indexed
- Test with `python src/rag_knowledge_base.py`

### ML predictions are poor
- Retrain models with more data
- Check feature engineering in preprocessing
- Verify training data quality

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or support, please open an issue on GitHub.
