#!/usr/bin/env python
# coding: utf-8

# # Data Exploration - Insurance Claims Extraction Dataset

# In[ ]:


import pandas as pd
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('../data/raw/insurance_claims_raw.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")


# In[ ]:


# Examine first sample
print("=== First Claim Text ===")
print(df['claim_text'].iloc[0][:500])
print("\n=== First Ground Truth ===")
print(json.dumps(json.loads(df['ground_truth'].iloc[0]), indent=2))


# In[ ]:


# Analyze ground truth structure
ground_truths = df['ground_truth'].apply(json.loads)

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

print(f"All unique keys in ground truth: {sorted(all_keys)}")


# In[ ]:


# Analyze structure of ground truth
print("=== Ground Truth Structure Analysis ===")
for i, gt in enumerate(ground_truths):
    print(f"\nSample {i}:")
    print(json.dumps(gt, indent=2))
    if i >= 2:  # Show first 3 samples
        break


# In[ ]:


# Text analysis
df['text_length'] = df['claim_text'].apply(len)
print(f"Text length statistics:")
print(df['text_length'].describe())


# In[ ]:


# Check for different claim types/channels
channels = []
for gt in ground_truths:
    if 'header' in gt and 'channel' in gt['header']:
        channels.append(gt['header']['channel'])

print(f"Channels: {Counter(channels)}")


# In[ ]:


# Analyze coverage types
coverage_types = []
for gt in ground_truths:
    if 'policy_details' in gt and 'coverage_type' in gt['policy_details']:
        coverage_types.append(gt['policy_details']['coverage_type'])

print(f"Coverage types: {Counter(coverage_types)}")


# In[ ]:


# Save processed data for training
processed_df = df.copy()
processed_df['ground_truth_dict'] = ground_truths
processed_df.to_csv('../data/processed/insurance_claims_processed.csv', index=False)
print("Processed data saved to ../data/processed/insurance_claims_processed.csv")

