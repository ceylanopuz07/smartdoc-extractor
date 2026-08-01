"""
Base ML extraction module using traditional ML algorithms
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import joblib
import config
from preprocessing import TextPreprocessor
import json

class BaseMLExtractor:
    """Base ML extractor using traditional algorithms"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.preprocessor = TextPreprocessor()
        self.label_encoders = {}
        self.models = {}
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for ML"""
        # Preprocess data
        processed_df = self.preprocessor.preprocess_dataframe(df)
        
        # Select feature columns
        text_features = config.TEXT_FEATURES
        pattern_features = [f'has_{p}' for p in self.preprocessor.patterns.keys()]
        
        self.feature_columns = text_features + pattern_features
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in processed_df.columns:
                processed_df[col] = 0
        
        feature_matrix = processed_df[self.feature_columns].fillna(0)
        
        return feature_matrix, processed_df
    
    def prepare_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare label matrix for ML"""
        labels_df = df['labels'].apply(lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series())
        
        # Encode categorical labels
        for col in labels_df.columns:
            if labels_df[col].dtype == 'object':
                # Fill NaN with empty string for encoding
                labels_df[col] = labels_df[col].fillna('')
                le = LabelEncoder()
                labels_df[col] = le.fit_transform(labels_df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # For numeric columns, fill NaN with 0 and ensure it's numeric
                labels_df[col] = pd.to_numeric(labels_df[col], errors='coerce').fillna(0)
        
        return labels_df
    
    def train(self, df: pd.DataFrame):
        """Train ML models"""
        print(f"Training {self.model_type} models...")
        
        # Prepare features and labels
        feature_matrix, processed_df = self.prepare_features(df)
        labels_df = self.prepare_labels(processed_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, labels_df, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE
        )
        
        # Train models for each target field
        for target_col in labels_df.columns:
            print(f"Training model for {target_col}...")
            
            if self.model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=config.N_ESTIMATORS,
                    random_state=config.RANDOM_STATE
                )
            elif self.model_type == 'svm':
                model = SVC(
                    C=config.SVM_C,
                    kernel=config.SVM_KERNEL,
                    random_state=config.RANDOM_STATE
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Train model
            model.fit(X_train, y_train[target_col])
            self.models[target_col] = model
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test[target_col], y_pred)
            print(f"  {target_col} accuracy: {accuracy:.4f}")
        
        print("Training completed!")
        return self.models
    
    def predict(self, text: str) -> dict:
        """Make prediction on single text"""
        # Create dataframe with single text
        df = pd.DataFrame({'claim_text': [text]})
        
        # Prepare features
        feature_matrix, _ = self.prepare_features(df)
        
        # Make predictions for each field
        predictions = {}
        for target_col, model in self.models.items():
            pred = model.predict(feature_matrix)[0]
            
            # Decode label if categorical
            if target_col in self.label_encoders:
                pred = self.label_encoders[target_col].inverse_transform([pred])[0]
            
            predictions[target_col] = pred
        
        return predictions
    
    def evaluate(self, df: pd.DataFrame):
        """Evaluate models on test set"""
        feature_matrix, processed_df = self.prepare_features(df)
        labels_df = self.prepare_labels(processed_df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, labels_df,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        results = {}
        for target_col, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test[target_col], y_pred)
            
            # Cross-validation (use fewer splits for small datasets)
            cv_splits = min(3, len(X_train) // 2)
            if cv_splits >= 2:
                cv_scores = cross_val_score(model, X_train, y_train[target_col], cv=cv_splits)
            else:
                cv_scores = np.array([accuracy])
            
            results[target_col] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"\n{target_col}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return results
    
    def save_models(self):
        """Save trained models"""
        for target_col, model in self.models.items():
            model_path = config.ML_MODELS_DIR / f"{self.model_type}_{target_col}.joblib"
            joblib.dump(model, model_path)
        
        # Save label encoders
        encoders_path = config.ML_MODELS_DIR / f"{self.model_type}_label_encoders.joblib"
        joblib.dump(self.label_encoders, encoders_path)
        
        print(f"Models saved to {config.ML_MODELS_DIR}")
    
    def load_models(self):
        """Load trained models"""
        # Load label encoders
        encoders_path = config.ML_MODELS_DIR / f"{self.model_type}_label_encoders.joblib"
        self.label_encoders = joblib.load(encoders_path)
        
        # Load models
        for target_col in self.label_encoders.keys():
            model_path = config.ML_MODELS_DIR / f"{self.model_type}_{target_col}.joblib"
            self.models[target_col] = joblib.load(model_path)
        
        print(f"Models loaded from {config.ML_MODELS_DIR}")
