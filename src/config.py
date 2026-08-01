"""
Configuration management for RAG-enhanced document extraction
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
MODELS_DIR = PROJECT_ROOT / "models"
ML_MODELS_DIR = MODELS_DIR / "ml_models"
BERT_MODELS_DIR = MODELS_DIR / "bert_models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, KNOWLEDGE_BASE_DIR, 
                 ML_MODELS_DIR, BERT_MODELS_DIR, EMBEDDINGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_NAME = "Cleanlab/insurance-claims-extraction"
RAW_DATA_FILE = RAW_DATA_DIR / "insurance_claims_raw.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "insurance_claims_processed.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# BERT configuration
BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# ML configuration
N_ESTIMATORS = 100
SVM_C = 1.0
SVM_KERNEL = 'rbf'

# Feature extraction
TEXT_FEATURES = ['text_length', 'word_count', 'sentence_count', 'avg_word_length']
PATTERN_FEATURES = ['has_email', 'has_phone', 'has_date', 'has_currency', 'has_claim_id']

# Target fields for extraction
TARGET_FIELDS = [
    'claim_id',
    'policy_number', 
    'policyholder_name',
    'coverage_type',
    'incident_type',
    'estimated_damage_amount',
    'report_date',
    'incident_date'
]
