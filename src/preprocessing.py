"""
Text preprocessing pipeline for insurance claims extraction
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import config

class TextPreprocessor:
    """Preprocess text data for ML extraction"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            'date': r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{2}-\d{2}-\d{4}\b',
            'currency': r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?',
            'claim_id': r'\bCLM-\d{6}\b',
            'policy_number': r'\bPOL-\d{9}\b'
        }
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_patterns(self, text: str) -> Dict[str, bool]:
        """Extract pattern features from text"""
        features = {}
        for pattern_name, pattern in self.patterns.items():
            features[f'has_{pattern_name}'] = bool(re.search(pattern, text))
        return features
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract numerical text features"""
        text = self.clean_text(text)
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        features = {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'char_count': len(text.replace(' ', '')),
        }
        
        return features
    
    def extract_field_candidates(self, text: str) -> Dict[str, List[str]]:
        """Extract potential field values using regex patterns"""
        candidates = {}
        
        # Claim IDs
        candidates['claim_id'] = re.findall(self.patterns['claim_id'], text)
        
        # Policy numbers
        candidates['policy_number'] = re.findall(self.patterns['policy_number'], text)
        
        # Dates
        candidates['dates'] = re.findall(self.patterns['date'], text)
        
        # Currency amounts
        candidates['amounts'] = re.findall(self.patterns['currency'], text)
        
        return candidates
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataframe"""
        processed_df = df.copy()
        
        # Remove existing feature columns if present to avoid duplicates
        feature_cols_to_remove = ['cleaned_text', 'field_candidates'] + \
                                [f'has_{p}' for p in self.patterns.keys()] + \
                                ['text_length', 'word_count', 'sentence_count', 'avg_word_length', 'char_count']
        
        for col in feature_cols_to_remove:
            if col in processed_df.columns:
                processed_df = processed_df.drop(columns=[col])
        
        # Clean text
        processed_df['cleaned_text'] = processed_df['claim_text'].apply(self.clean_text)
        
        # Extract text features
        text_features = processed_df['cleaned_text'].apply(self.extract_text_features)
        text_features_df = pd.DataFrame(text_features.tolist())
        
        # Extract pattern features
        pattern_features = processed_df['cleaned_text'].apply(self.extract_patterns)
        pattern_features_df = pd.DataFrame(pattern_features.tolist())
        
        # Combine all features
        all_features = pd.concat([text_features_df, pattern_features_df], axis=1)
        
        # Remove duplicate columns if any
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        processed_df = pd.concat([processed_df, all_features], axis=1)
        
        # Extract field candidates
        field_candidates = processed_df['cleaned_text'].apply(self.extract_field_candidates)
        processed_df['field_candidates'] = field_candidates
        
        return processed_df
    
    def extract_labels_from_ground_truth(self, ground_truth_dict: Dict) -> Dict[str, Any]:
        """Extract target labels from ground truth dictionary"""
        labels = {}
        
        if not ground_truth_dict:
            return labels
        
        # Header fields
        if 'header' in ground_truth_dict:
            header = ground_truth_dict['header']
            labels['claim_id'] = header.get('claim_id', '')
            labels['report_date'] = header.get('report_date', '')
            labels['incident_date'] = header.get('incident_date', '')
            labels['reported_by'] = header.get('reported_by', '')
            labels['channel'] = header.get('channel', '')
        
        # Policy details
        if 'policy_details' in ground_truth_dict and ground_truth_dict['policy_details']:
            policy = ground_truth_dict['policy_details']
            labels['policy_number'] = policy.get('policy_number', '')
            labels['policyholder_name'] = policy.get('policyholder_name', '')
            labels['coverage_type'] = policy.get('coverage_type', '')
        
        # Incident description
        if 'incident_description' in ground_truth_dict:
            incident = ground_truth_dict['incident_description']
            labels['incident_type'] = incident.get('incident_type', '')
            labels['estimated_damage_amount'] = incident.get('estimated_damage_amount', '')
        
        return labels
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare training data with features and labels"""
        processed_df = self.preprocess_dataframe(df)
        
        # Extract labels
        processed_df['labels'] = processed_df['ground_truth_dict'].apply(
            self.extract_labels_from_ground_truth
        )
        
        return processed_df
