"""
Preprocess thoughtworks document-processing-benchmark dataset
"""
import pandas as pd
import json
import ast
import os
from collections import Counter

class BenchmarkPreprocessor:
    """Preprocess thoughtworks benchmark dataset"""
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
    
    def load_data(self):
        """Load the raw dataset"""
        data_path = os.path.join(self.project_root, "data", "raw", "document_benchmark_raw.csv")
        df = pd.read_csv(data_path)
        return df
    
    def parse_records(self, df):
        """Parse the nested JSON records"""
        records = df['test'].apply(ast.literal_eval)
        parsed_df = pd.DataFrame(records.tolist())
        return parsed_df
    
    def extract_labels_from_ground_truth(self, gt_json):
        """Extract labels from ground truth JSON"""
        if not gt_json:
            return {}
        
        try:
            gt = json.loads(gt_json) if isinstance(gt_json, str) else gt_json
            labels = {}
            
            # Extract from gt_parse
            if 'gt_parse' in gt:
                parse = gt['gt_parse']
                
                # Flatten nested structure
                def flatten_dict(d, parent_key='', sep='_'):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)
                
                labels = flatten_dict(parse)
            
            # Add metadata
            if 'meta' in gt:
                labels['meta_version'] = gt['meta'].get('version', '')
                labels['meta_split'] = gt['meta'].get('split', '')
            
            return labels
        except:
            return {}
    
    def preprocess_dataframe(self, df):
        """Preprocess the entire dataframe"""
        print("Parsing records...")
        parsed_df = self.parse_records(df)
        
        print("Extracting labels from ground truth...")
        parsed_df['labels'] = parsed_df['ground_truth_json'].apply(self.extract_labels_from_ground_truth)
        
        # For now, skip OCR (slow) - we'll use structured labels
        # Later can add OCR for text features
        
        print(f"Processed shape: {parsed_df.shape}")
        return parsed_df
    
    def save_processed_data(self, df):
        """Save processed data"""
        processed_dir = os.path.join(self.project_root, "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        output_path = os.path.join(processed_dir, "document_benchmark_processed.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        
        return output_path

def main():
    """Main preprocessing pipeline"""
    preprocessor = BenchmarkPreprocessor()
    
    # Load data
    print("Loading data...")
    df = preprocessor.load_data()
    print(f"Loaded {len(df)} records")
    
    # Preprocess
    processed_df = preprocessor.preprocess_dataframe(df)
    
    # Save
    preprocessor.save_processed_data(processed_df)
    
    # Show sample
    print("\n=== Sample Labels ===")
    print(processed_df['labels'].iloc[0])
    
    print("\n=== Label Statistics ===")
    all_labels = {}
    for labels in processed_df['labels']:
        for key in labels.keys():
            all_labels[key] = all_labels.get(key, 0) + 1
    
    print(f"Total unique label keys: {len(all_labels)}")
    print("Top 20 most common keys:")
    for key, count in sorted(all_labels.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {key}: {count}")

if __name__ == "__main__":
    main()
