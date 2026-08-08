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

from ocr_processor import OCRProcessor
from llm_extractor import LLMExtractor

app = FastAPI(
    title="SmartDoc Extractor API",
    description="Production-ready SmartDoc Extractor with LLM-based extraction using enhanced prompts and few-shot learning",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
ocr_processor = None
llm_extractor = None

@app.on_event("startup")
async def startup_event():
    """Initialize OCR processor and LLM extractor on startup"""
    global ocr_processor, llm_extractor
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
    success: bool
    message: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SmartDoc Extractor API",
        "version": "1.0.0",
        "endpoints": {
            "/extract-file": "POST - Extract information from uploaded document file (LLM-based)",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ocr_initialized": ocr_processor is not None,
        "llm_initialized": llm_extractor is not None
    }


@app.post("/extract-file", response_model=ExtractionResponse)
async def extract_from_file(file: UploadFile = File(...)):
    """
    Extract information from uploaded document file using LLM-based extraction
    
    - **file**: Document file (PNG, JPG, PDF)
    """
    if ocr_processor is None or llm_extractor is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Read file
        file_bytes = await file.read()
        filename = file.filename
        
        print(f"Processing file: {filename}, size: {len(file_bytes)} bytes")
        
        # Extract text using OCR (handles both images and PDFs)
        text = ocr_processor.extract_text(file_bytes, filename)
        print(f"Extracted text length: {len(text)} characters")
        
        # Use LLM extraction with fallback to rule-based extraction
        results = llm_extractor.extract(text)
        
        return ExtractionResponse(
            extraction_results=results.get('llm_results', results.get('enhanced_results', {})),
            success=results.get('success', True),
            message=results.get('message', 'Extraction completed successfully')
        )
    except Exception as e:
        import traceback
        print(f"Error in extract_from_file: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"File extraction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
