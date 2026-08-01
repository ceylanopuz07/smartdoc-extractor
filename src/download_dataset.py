"""
Download insurance-claims-extraction dataset from Hugging Face
"""
from datasets import load_dataset
import pandas as pd
import os

def download_dataset():
    """Download the insurance claims extraction dataset"""
    print("Downloading insurance-claims-extraction dataset from Hugging Face...")
    
    # Load dataset
    dataset = load_dataset("Cleanlab/insurance-claims-extraction")
    
    print(f"Dataset loaded successfully!")
    print(f"Train split size: {len(dataset['train'])}")
    
    # Convert to pandas for easier exploration
    train_df = pd.DataFrame(dataset['train'])
    
    # Save to data/raw directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    output_path = os.path.join(raw_dir, "insurance_claims_raw.csv")
    train_df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    
    return train_df

if __name__ == "__main__":
    df = download_dataset()
    print("\nFirst few rows:")
    print(df.head())
