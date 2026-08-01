"""
Train ML models on thoughtworks benchmark dataset
"""
import pandas as pd
import sys
import os
import ast
import json

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED_DATA_FILE
from base_extractor import BaseMLExtractor

def main():
    """Train and evaluate ML models on benchmark dataset"""
    print("Loading processed data...")
    
    # Load the benchmark processed data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "processed", "document_benchmark_processed.csv")
    
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Parse labels
    df['labels'] = df['labels'].apply(ast.literal_eval)
    
    # For this dataset, we'll use metadata as features since we don't have OCR text yet
    # Create simple features from available columns
    print("Creating features...")
    
    # Feature columns we can use
    feature_cols = ['doc_type', 'task_type', 'image_w_px', 'image_h_px', 'image_bytes_len']
    
    # One-hot encode categorical features
    features_df = pd.DataFrame()
    
    # Document type
    if 'doc_type' in df.columns:
        doc_type_dummies = pd.get_dummies(df['doc_type'], prefix='doc_type')
        features_df = pd.concat([features_df, doc_type_dummies], axis=1)
    
    # Image dimensions
    if 'image_w_px' in df.columns:
        features_df['image_w_px'] = df['image_w_px'].fillna(0)
    if 'image_h_px' in df.columns:
        features_df['image_h_px'] = df['image_h_px'].fillna(0)
    if 'image_bytes_len' in df.columns:
        features_df['image_bytes_len'] = df['image_bytes_len'].fillna(0)
    
    # Add token counts as features
    if 'gt_token_count_cl100k' in df.columns:
        features_df['gt_token_count'] = df['gt_token_count_cl100k'].fillna(0)
    
    print(f"Feature columns: {list(features_df.columns)}")
    print(f"Features shape: {features_df.shape}")
    
    # Prepare labels - select most common fields
    print("Preparing labels...")
    
    # Get all label keys and their frequency
    label_counts = {}
    for labels in df['labels']:
        for key in labels.keys():
            label_counts[key] = label_counts.get(key, 0) + 1
    
    # Select top N most common labels
    top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top labels: {top_labels}")
    
    # Create label matrix for top labels
    label_matrix = {}
    for key, _ in top_labels:
        label_matrix[key] = df['labels'].apply(lambda x: x.get(key, ''))
    
    labels_df = pd.DataFrame(label_matrix)
    
    # Encode categorical labels
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    
    for col in labels_df.columns:
        if labels_df[col].dtype == 'object':
            labels_df[col] = labels_df[col].fillna('')
            le = LabelEncoder()
            labels_df[col] = le.fit_transform(labels_df[col].astype(str))
            label_encoders[col] = le
        else:
            labels_df[col] = pd.to_numeric(labels_df[col], errors='coerce').fillna(0)
    
    print(f"Labels shape: {labels_df.shape}")
    
    # Train Random Forest
    print("\n" + "="*50)
    print("Training Random Forest Models")
    print("="*50)
    
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    import numpy as np
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels_df,
        test_size=0.2,
        random_state=42
    )
    
    models = {}
    results = {}
    
    for target_col in labels_df.columns:
        print(f"\nTraining model for {target_col}...")
        
        # Check if target is continuous or categorical
        unique_values = y_train[target_col].nunique()
        is_continuous = unique_values > 20 or y_train[target_col].dtype in ['float64', 'int64']
        
        if is_continuous:
            print(f"  Using regressor (continuous values, {unique_values} unique)")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train[target_col])
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test[target_col], y_pred)
            results[target_col] = {'type': 'regression', 'metric': 'MSE', 'value': mse}
            print(f"  MSE: {mse:.4f}")
        else:
            print(f"  Using classifier (categorical values, {unique_values} unique)")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train[target_col])
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test[target_col], y_pred)
            results[target_col] = {'type': 'classification', 'metric': 'accuracy', 'value': accuracy}
            print(f"  Accuracy: {accuracy:.4f}")
        
        models[target_col] = model
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    for col, result in results.items():
        print(f"{col}: {result['metric']} = {result['value']:.4f}")
    
    # Calculate average for classification tasks only
    classification_results = [r['value'] for r in results.values() if r['type'] == 'classification']
    if classification_results:
        print(f"\nAverage classification accuracy: {np.mean(classification_results):.4f}")
    
    # Save models
    import joblib
    models_dir = os.path.join(project_root, "models", "ml_models")
    os.makedirs(models_dir, exist_ok=True)
    
    for col, model in models.items():
        model_path = os.path.join(models_dir, f"benchmark_rf_{col}.joblib")
        joblib.dump(model, model_path)
    
    print(f"\nModels saved to {models_dir}")

if __name__ == "__main__":
    main()
