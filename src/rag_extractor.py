"""
RAG-Enhanced Document Extractor combining ML predictions with RAG retrieval
"""
import pandas as pd
import numpy as np
import json
import ast
import joblib
import os
from typing import Dict, Any, List
from rag_knowledge_base import RAGKnowledgeBase

class RAGEnhancedExtractor:
    """RAG-enhanced document extractor"""
    
    def __init__(self):
        """Initialize RAG-enhanced extractor"""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        
        # Initialize RAG knowledge base
        self.kb = RAGKnowledgeBase()
        
        # Load trained ML models
        self.models_dir = os.path.join(self.project_root, "models", "ml_models")
        self.ml_models = {}
        self.load_ml_models()
        
        print("RAG-Enhanced Extractor initialized")
    
    def load_ml_models(self):
        """Load trained ML models"""
        if not os.path.exists(self.models_dir):
            print("No trained models found. Training required first.")
            return
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
        print(f"Loading {len(model_files)} ML models...")
        
        for model_file in model_files:
            model_name = model_file.replace('.joblib', '').replace('benchmark_rf_', '')
            model_path = os.path.join(self.models_dir, model_file)
            self.ml_models[model_name] = joblib.load(model_path)
            print(f"  Loaded: {model_name}")
    
    def prepare_features(self, doc_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for ML prediction"""
        features = []
        
        # Document type one-hot encoding
        doc_type = doc_data.get('doc_type', 'unknown')
        features.extend([
            1 if doc_type == 'form' else 0,
            1 if doc_type == 'invoice' else 0,
            1 if doc_type == 'receipt' else 0
        ])
        
        # Image dimensions
        features.append(doc_data.get('image_w_px', 0))
        features.append(doc_data.get('image_h_px', 0))
        features.append(doc_data.get('image_bytes_len', 0))
        
        # Token count
        features.append(doc_data.get('gt_token_count_cl100k', 0))
        
        return np.array([features])
    
    def ml_extract(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract using ML models only"""
        ml_results = {}
        
        if not self.ml_models:
            return ml_results
        
        # Prepare features
        features = self.prepare_features(doc_data)
        
        # Get predictions from all models
        for field_name, model in self.ml_models.items():
            try:
                prediction = model.predict(features)[0]
                ml_results[field_name] = prediction
            except Exception as e:
                print(f"Error predicting {field_name}: {e}")
        
        return ml_results
    
    def rag_enhance(self, doc_data: Dict[str, Any], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance ML results with RAG retrieval"""
        # Create query from ML results
        query_text = json.dumps(ml_results, indent=2)
        
        # Retrieve similar documents
        similar_docs = self.kb.retrieve_similar_documents(query_text, n_results=3)
        
        # Extract context from similar documents
        rag_context = []
        for doc in similar_docs:
            try:
                doc_labels = json.loads(doc['document'])
                rag_context.append({
                    'labels': doc_labels,
                    'distance': doc['distance'],
                    'metadata': doc['metadata']
                })
            except:
                continue
        
        # Enhance results using retrieved context
        enhanced_results = ml_results.copy()
        
        # Simple enhancement: if ML prediction is uncertain, use similar document values
        for field_name in ml_results.keys():
            # Get values from similar documents
            similar_values = []
            for context in rag_context:
                if field_name in context['labels']:
                    similar_values.append(context['labels'][field_name])
            
            # If we have similar values and they're consistent, use them
            if similar_values and len(set(similar_values)) == 1:
                enhanced_results[f"{field_name}_rag"] = similar_values[0]
        
        return {
            'ml_results': ml_results,
            'rag_context': rag_context,
            'enhanced_results': enhanced_results
        }
    
    def extract(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Full extraction pipeline: ML + RAG"""
        # ML extraction
        ml_results = self.ml_extract(doc_data)
        
        # RAG enhancement
        rag_results = self.rag_enhance(doc_data, ml_results)
        
        return rag_results

def main():
    """Test RAG-enhanced extractor"""
    print("Testing RAG-Enhanced Extractor...")
    
    # Initialize extractor
    extractor = RAGEnhancedExtractor()
    
    # Create sample document data
    sample_doc = {
        'doc_type': 'receipt',
        'image_w_px': 432,
        'image_h_px': 648,
        'image_bytes_len': 50000,
        'gt_token_count_cl100k': 100
    }
    
    print("\nSample document:")
    print(json.dumps(sample_doc, indent=2))
    
    # Extract
    print("\nRunning extraction...")
    results = extractor.extract(sample_doc)
    
    print("\n=== ML Results ===")
    print(json.dumps(results['ml_results'], indent=2))
    
    print("\n=== RAG Context ===")
    print(f"Retrieved {len(results['rag_context'])} similar documents")
    for i, context in enumerate(results['rag_context']):
        print(f"\n  Context {i+1}:")
        print(f"    Distance: {context['distance']:.4f}")
        print(f"    Metadata: {context['metadata']}")
        print(f"    Labels: {list(context['labels'].keys())}")
    
    print("\n=== Enhanced Results ===")
    print(json.dumps(results['enhanced_results'], indent=2))

if __name__ == "__main__":
    main()
