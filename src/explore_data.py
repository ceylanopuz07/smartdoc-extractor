"""
Data exploration script for insurance claims extraction dataset
"""
import pandas as pd
import json
import ast
from collections import Counter
import os

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "data", "raw", "insurance_claims_raw.csv")

df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Examine first sample
print("\n=== First Claim Text ===")
print(df['claim_text'].iloc[0][:500])
print("\n=== First Ground Truth ===")
print(df['ground_truth'].iloc[0][:500])

# Parse ground truth (handle both JSON and Python dict formats)
ground_truths = []
for gt_str in df['ground_truth']:
    try:
        gt = json.loads(gt_str)
    except:
        try:
            gt = ast.literal_eval(gt_str)
        except:
            gt = {}
    ground_truths.append(gt)

# Extract all keys from ground truth
all_keys = set()
for gt in ground_truths:
    def extract_keys(obj, keys_set):
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys_set.add(k)
                extract_keys(v, keys_set)
        elif isinstance(obj, list):
            for item in obj:
                extract_keys(item, keys_set)
    extract_keys(gt, all_keys)

print(f"\n=== All unique keys in ground truth ===")
print(sorted(all_keys))

# Analyze structure of ground truth
print("\n=== Ground Truth Structure Analysis ===")
for i, gt in enumerate(ground_truths):
    print(f"\nSample {i}:")
    print(json.dumps(gt, indent=2))
    if i >= 2:  # Show first 3 samples
        break

# Text analysis
df['text_length'] = df['claim_text'].apply(len)
print(f"\n=== Text length statistics ===")
print(df['text_length'].describe())

# Check for different claim types/channels
channels = []
for gt in ground_truths:
    if 'header' in gt and 'channel' in gt['header']:
        channels.append(gt['header']['channel'])

print(f"\n=== Channels ===")
print(Counter(channels))

# Analyze coverage types
coverage_types = []
for gt in ground_truths:
    if gt and 'policy_details' in gt and gt['policy_details'] and 'coverage_type' in gt['policy_details']:
        coverage_types.append(gt['policy_details']['coverage_type'])

print(f"\n=== Coverage types ===")
print(Counter(coverage_types))

# Save processed data for training
processed_dir = os.path.join(project_root, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

processed_df = df.copy()
processed_df['ground_truth_dict'] = ground_truths
processed_path = os.path.join(processed_dir, "insurance_claims_processed.csv")
processed_df.to_csv(processed_path, index=False)
print(f"\nProcessed data saved to {processed_path}")
