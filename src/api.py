"""
FastAPI interface for RAG-enhanced document extraction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_extractor import RAGEnhancedExtractor

app = FastAPI(
    title="RAG Document Extraction API",
    description="Production-ready RAG-enhanced document extraction system",
    version="1.0.0"
)

# Global extractor instance
extractor = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG extractor on startup"""
    global extractor
    print("Initializing RAG Enhanced Extractor...")
    extractor = RAGEnhancedExtractor()
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
    ml_results: Dict[str, Any]
    rag_context: list
    enhanced_results: Dict[str, Any]
    success: bool
    message: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Document Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/extract": "POST - Extract information from document",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "extractor_initialized": extractor is not None
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
            ml_results=results['ml_results'],
            rag_context=results['rag_context'],
            enhanced_results=results['enhanced_results'],
            success=True,
            message="Extraction completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
