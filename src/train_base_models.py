"""
Train base ML models (Random Forest and SVM)
"""
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED_DATA_FILE
from base_extractor import BaseMLExtractor
from preprocessing import TextPreprocessor

def main():
    """Train and evaluate base ML models"""
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    
    # Parse ground truth dictionaries
    import ast
    df['ground_truth_dict'] = df['ground_truth_dict'].apply(ast.literal_eval)
    
    # Prepare training data with labels
    print("Preparing training data...")
    preprocessor = TextPreprocessor()
    
    # Extract labels first
    df['labels'] = df['ground_truth_dict'].apply(preprocessor.extract_labels_from_ground_truth)
    
    print(f"Dataset shape: {df.shape}")
    
    # Train Random Forest
    print("\n" + "="*50)
    print("Training Random Forest Models")
    print("="*50)
    rf_extractor = BaseMLExtractor(model_type='random_forest')
    rf_extractor.train(df)
    
    print("\n" + "="*50)
    print("Random Forest Evaluation")
    print("="*50)
    rf_results = rf_extractor.evaluate(df)
    
    # Save Random Forest models
    rf_extractor.save_models()
    
    # Train SVM
    print("\n" + "="*50)
    print("Training SVM Models")
    print("="*50)
    svm_extractor = BaseMLExtractor(model_type='svm')
    svm_extractor.train(df)
    
    print("\n" + "="*50)
    print("SVM Evaluation")
    print("="*50)
    svm_results = svm_extractor.evaluate(df)
    
    # Save SVM models
    svm_extractor.save_models()
    
    # Test prediction
    print("\n" + "="*50)
    print("Test Prediction")
    print("="*50)
    sample_text = df['claim_text'].iloc[0]
    print(f"Sample text: {sample_text[:200]}...")
    
    prediction = rf_extractor.predict(sample_text)
    print(f"\nPrediction: {prediction}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
