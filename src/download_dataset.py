"""
Download thoughtworks document-processing-benchmark dataset from Hugging Face
"""
from datasets import load_dataset
import pandas as pd
import os

def download_dataset():
    """Download the thoughtworks document-processing-benchmark dataset"""
    print("Downloading thoughtworks/document-processing-benchmark dataset from Hugging Face...")
    
    # Load dataset (using test config for manageable size)
    dataset = load_dataset("thoughtworks/document-processing-benchmark", "test")
    
    print(f"Dataset loaded successfully!")
    print(f"Dataset size: {len(dataset)}")
    
    # Convert to pandas for easier exploration
    df = pd.DataFrame(dataset)
    
    # Save to data/raw directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    output_path = os.path.join(raw_dir, "document_benchmark_raw.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    df = download_dataset()
    print("\nFirst few rows:")
    print(df.head())
