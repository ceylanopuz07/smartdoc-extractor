"""
Explore thoughtworks document-processing-benchmark dataset structure
"""
import pandas as pd
import json
import ast
import os

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "data", "raw", "document_benchmark_raw.csv")

df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Parse the first record
first_record = ast.literal_eval(df['test'].iloc[0])
print("\n=== First Record Structure ===")
print(json.dumps(first_record, indent=2))

# Check keys in the record
print(f"\n=== Keys in Record ===")
print(list(first_record.keys()))

# Check source datasets
source_datasets = df['test'].apply(lambda x: ast.literal_eval(x)['source_dataset'])
print(f"\n=== Source Datasets ===")
print(source_datasets.value_counts())

# Check document types
doc_types = df['test'].apply(lambda x: ast.literal_eval(x).get('doc_type', 'unknown'))
print(f"\n=== Document Types ===")
print(doc_types.value_counts())

# Examine ground truth structure
if 'ground_truth_json' in first_record:
    print("\n=== Ground Truth Structure ===")
    print(json.dumps(first_record['ground_truth_json'], indent=2)[:500])
