"""
FastAPI interface for RAG-enhanced document extraction
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_extractor import RAGEnhancedExtractor
from ocr_processor import OCRProcessor
from llm_extractor import LLMExtractor

app = FastAPI(
    title="RAG Document Extraction API",
    description="Production-ready RAG-enhanced document extraction system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global extractor instance
extractor = None
ocr_processor = None
llm_extractor = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG extractor, OCR processor, and LLM extractor on startup"""
    global extractor, ocr_processor, llm_extractor
    print("Initializing RAG Enhanced Extractor...")
    extractor = RAGEnhancedExtractor()
    print("Initializing OCR Processor...")
    ocr_processor = OCRProcessor()
    print("Initializing LLM Extractor...")
    llm_extractor = LLMExtractor()
    print("API ready!")

class DocumentRequest(BaseModel):
    """Request model for document extraction"""
    doc_type: Optional[str] = "unknown"
    image_w_px: Optional[int] = 0
    image_h_px: Optional[int] = 0
    image_bytes_len: Optional[int] = 0
    gt_token_count_cl100k: Optional[int] = 0

class ExtractionResponse(BaseModel):
    """Response model for extraction results"""
    extraction_results: Dict[str, Any]
    rag_context: list
    success: bool
    message: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Document Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/extract": "POST - Extract information from document metadata",
            "/extract-file": "POST - Extract information from uploaded document file (supports LLM)",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "extractor_initialized": extractor is not None,
        "ocr_initialized": ocr_processor is not None,
        "llm_initialized": llm_extractor is not None
    }

@app.post("/extract", response_model=ExtractionResponse)
async def extract(document: DocumentRequest):
    """
    Extract information from document using RAG-enhanced extraction
    
    - **doc_type**: Document type (invoice, receipt, form)
    - **image_w_px**: Image width in pixels
    - **image_h_px**: Image height in pixels
    - **image_bytes_len**: Image size in bytes
    - **gt_token_count_cl100k**: Token count for the document
    """
    if extractor is None:
        raise HTTPException(status_code=503, detail="Extractor not initialized")
    
    try:
        # Convert request to dict
        doc_data = document.dict()
        
        # Run extraction
        results = extractor.extract(doc_data)
        
        return ExtractionResponse(
            extraction_results=results.get('llm_results', {}),
            rag_context=results.get('rag_context', []),
            success=True,
            message="Extraction completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/extract-file", response_model=ExtractionResponse)
async def extract_from_file(file: UploadFile = File(...), use_llm: bool = True):
    """
    Extract information from uploaded document file
    
    - **file**: Document image file (PNG, JPG, PDF)
    - **use_llm**: Use LLM-based extraction (default: True)
    """
    if extractor is None or ocr_processor is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Read file
        image_bytes = await file.read()
        
        # Extract text using OCR
        text = ocr_processor.extract_text_from_image(image_bytes)
        
        # Choose extraction method
        if use_llm and llm_extractor:
            # Use LLM extraction with RAG context
            rag_results = extractor.extract({
                'doc_type': 'unknown',
                'extracted_text': text,
                'text_length': len(text),
                'image_w_px': 0,
                'image_h_px': 0,
                'image_bytes_len': len(image_bytes),
                'gt_token_count_cl100k': len(text.split())
            })
            
            # Use LLM with RAG context
            results = llm_extractor.extract(text, rag_results.get('rag_context', []))
        else:
            # Use ML-based extraction
            doc_data = {
                'doc_type': 'unknown',
                'extracted_text': text,
                'text_length': len(text),
                'image_w_px': 0,
                'image_h_px': 0,
                'image_bytes_len': len(image_bytes),
                'gt_token_count_cl100k': len(text.split())
            }
            results = extractor.extract(doc_data)
            results['llm_results'] = {}
        
        return ExtractionResponse(
            extraction_results=results.get('llm_results', results.get('ml_results', {})),
            rag_context=results.get('rag_context', []),
            success=results.get('success', True),
            message=results.get('message', 'Extraction completed successfully')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File extraction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
