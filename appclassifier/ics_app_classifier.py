#!/usr/bin/env python3
"""
ICS App Classification using Multiple ML Models
Classifies apps as ICS vs non-ICS based on description text using various ML models.

Dataset:
- training_data_ics.csv: ICS apps (positive class)
- training_data_non_ics.csv: non-ICS apps (negative class)
- Training: 220 apps each class
- Testing: 13 apps each class

Models implemented:
1. Logistic Regression
2. Support Vector Machines
3. Naive Bayes
4. Random Forest
5. Recurrent Neural Network (RNN)
6. Long Short Term Memory (LSTM)
"""

import pandas as pd
import numpy as np
import re
import os
import gc
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ICSAppClassifier:
    def __init__(self, data_dir=None):
        # Default to the directory containing this file if no data_dir is provided
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.ics_file = os.path.join(data_dir, "training_data_ics.csv")
        self.non_ics_file = os.path.join(data_dir, "training_data_non_ics.csv")
        self.results_file = os.path.join(data_dir, "ml_models_results.txt")
        
        # Model parameters
        self.model_params = {
            'logistic_regression': {
                'C': 0.5,
                'max_iter': 100,
                'random_state': 42
            },
            'svm': {
                'C': 0.5,
                'kernel': 'linear',
                'random_state': 42
            },
            'naive_bayes': {
                'alpha': 1.0
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 15,
                'random_state': 42
            },
            #'lstm': {
            #    'vocab_size': 8000,
            #    'embedding_dim': 128,
            #    'lstm_units': 64,
            #    'max_length': 150,
            #    'epochs': 25,
            #    'batch_size': 32,
            #    'patience': 3
            #}
            'lstm': {
                'vocab_size': 8000,
                'embedding_dim': 32,
                'lstm_units': 16,
                'max_length': 10,
                'epochs': 10,
                'batch_size': 8,
                'patience': 10
            }
        }
        
        self.results = {}
        
    def _cleanup_tf(self) -> None:
        """Clear TensorFlow/Keras session and collect garbage to free memory/caches."""
        try:
            K.clear_session()
        finally:
            gc.collect()

    def load_data(self):
        """Load and combine ICS and non-ICS data"""
        print("Loading data...")
        
        # Load ICS apps (positive class)
        ics_df = pd.read_csv(self.ics_file)
        ics_df['label'] = 1
        ics_df['class'] = 'ICS'
        
        # Load non-ICS apps (negative class)
        non_ics_df = pd.read_csv(self.non_ics_file)
        non_ics_df['label'] = 0
        non_ics_df['class'] = 'non-ICS'
        
        # Combine datasets
        self.df = pd.concat([ics_df, non_ics_df], ignore_index=True)
        
        # Clean description column
        self.df['description_clean'] = self.df['descriptionHtml'].fillna('')
        
        print(f"Total apps loaded: {len(self.df)}")
        print(f"ICS apps: {len(ics_df)}")
        print(f"Non-ICS apps: {len(non_ics_df)}")
        
        return self.df
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove special characters and digits, keep letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_features(self):
        """Prepare TF-IDF features and neural network sequences"""
        print("Preprocessing text and extracting features...")
        
        # Clean text descriptions
        self.df['description_processed'] = self.df['description_clean'].apply(self.preprocess_text)
        
        # Remove empty descriptions
        self.df = self.df[self.df['description_processed'].str.len() > 0]
        
        # Prepare data for classical ML models (TF-IDF)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.X_tfidf = self.tfidf_vectorizer.fit_transform(self.df['description_processed'])
        self.y = self.df['label'].values
        
        # Split data for classical models
        self.X_train_tfidf, self.X_test_tfidf, self.y_train, self.y_test = train_test_split(
            self.X_tfidf, self.y, test_size=0.1, random_state=42, stratify=self.y
        )
        
        # Prepare data for neural networks (tokenized sequences)
        self.tokenizer = Tokenizer(num_words=self.model_params['rnn']['vocab_size'], oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.df['description_processed'])
        
        sequences = self.tokenizer.texts_to_sequences(self.df['description_processed'])
        self.X_seq = pad_sequences(sequences, maxlen=self.model_params['rnn']['max_length'])
        
        # Split data for neural networks
        self.X_train_seq, self.X_test_seq, self.y_train_seq, self.y_test_seq = train_test_split(
            self.X_seq, self.y, test_size=0.1, random_state=42, stratify=self.y
        )
        
        print(f"TF-IDF feature matrix shape: {self.X_tfidf.shape}")
        print(f"Training set size: {self.X_train_tfidf.shape[0]}")
        print(f"Test set size: {self.X_test_tfidf.shape[0]}")
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        
        # For neural networks, convert probabilities to binary predictions
        if model_name in ['RNN', 'LSTM']:
            y_pred = (y_pred > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, target_names=['non-ICS', 'ICS'])
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        
        params = self.model_params['logistic_regression']
        model = LogisticRegression(**params)
        model.fit(self.X_train_tfidf, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_tfidf, self.y_train, cv=5)
        
        # Evaluation
        results = self.evaluate_model(model, self.X_test_tfidf, self.y_test, 'Logistic Regression')
        results['cv_scores'] = cv_scores
        results['model_params'] = params
        
        self.results['Logistic Regression'] = results
        return model, results
    
    def train_svm(self):
        """Train Support Vector Machine model"""
        print("Training Support Vector Machine...")
        
        params = self.model_params['svm']
        model = SVC(**params)
        model.fit(self.X_train_tfidf, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_tfidf, self.y_train, cv=5)
        
        # Evaluation
        results = self.evaluate_model(model, self.X_test_tfidf, self.y_test, 'SVM')
        results['cv_scores'] = cv_scores
        results['model_params'] = params
        
        self.results['SVM'] = results
        return model, results
    
    def train_naive_bayes(self):
        """Train Naive Bayes model"""
        print("Training Naive Bayes...")
        
        params = self.model_params['naive_bayes']
        model = MultinomialNB(**params)
        model.fit(self.X_train_tfidf, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_tfidf, self.y_train, cv=5)
        
        # Evaluation
        results = self.evaluate_model(model, self.X_test_tfidf, self.y_test, 'Naive Bayes')
        results['cv_scores'] = cv_scores
        results['model_params'] = params
        
        self.results['Naive Bayes'] = results
        return model, results
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        params = self.model_params['random_forest']
        model = RandomForestClassifier(**params)
        model.fit(self.X_train_tfidf, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_tfidf, self.y_train, cv=5)
        
        # Evaluation
        results = self.evaluate_model(model, self.X_test_tfidf, self.y_test, 'Random Forest')
        results['cv_scores'] = cv_scores
        results['model_params'] = params
        
        self.results['Random Forest'] = results
        return model, results
    
    def train_lstm(self):
        """Train LSTM model"""
        print("Training LSTM...")
        
        params = self.model_params['lstm']
        
        # Build LSTM model - Regularized to prevent overfitting
        model = Sequential([
            Embedding(params['vocab_size'], params['embedding_dim'], input_length=params['max_length']),
            LSTM(params['lstm_units'], dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
            LSTM(params['lstm_units'] // 2, dropout=0.3, recurrent_dropout=0.3),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
        
        # Train model
        history = model.fit(
            self.X_train_seq, self.y_train_seq,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluation
        results = self.evaluate_model(model, self.X_test_seq, self.y_test_seq, 'LSTM')
        results['model_params'] = params
        results['training_history'] = history.history
        
        self.results['LSTM'] = results
        # Clear TF caches/session to avoid state carryover between runs
        del model
        self._cleanup_tf()
        return None, results
    
    def save_results(self):
        """Save detailed results to file"""
        print(f"Saving results to {self.results_file}...")
        
        with open(self.results_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ICS APP CLASSIFICATION - ML MODELS RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET INFORMATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total apps: {len(self.df)}\n")
            f.write(f"ICS apps: {sum(self.df['label'] == 1)}\n")
            f.write(f"Non-ICS apps: {sum(self.df['label'] == 0)}\n")
            f.write(f"Training set size: {self.X_train_tfidf.shape[0]}\n")
            f.write(f"Test set size: {self.X_test_tfidf.shape[0]}\n")
            f.write(f"TF-IDF features: {self.X_tfidf.shape[1]}\n\n")
            
            # Summary table
            f.write("RESULTS SUMMARY:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            f.write("-"*70 + "\n")
            
            for model_name, results in self.results.items():
                f.write(f"{model_name:<20} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                       f"{results['recall']:<10.4f} {results['f1_score']:<10.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED RESULTS BY MODEL\n")
            f.write("="*80 + "\n\n")
            
            # Detailed results for each model
            for model_name, results in self.results.items():
                f.write(f"{model_name.upper()}\n")
                f.write("-" * len(model_name) + "\n")
                
                # Model parameters
                f.write("Model Parameters:\n")
                for param, value in results['model_params'].items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
                
                # Performance metrics
                f.write("Performance Metrics:\n")
                f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
                f.write(f"  Precision: {results['precision']:.4f}\n")
                f.write(f"  Recall:    {results['recall']:.4f}\n")
                f.write(f"  F1-Score:  {results['f1_score']:.4f}\n")
                
                # Cross-validation scores (for classical models)
                if 'cv_scores' in results:
                    f.write(f"  CV Mean:   {results['cv_scores'].mean():.4f} (+/- {results['cv_scores'].std() * 2:.4f})\n")
                
                f.write("\n")
                
                # Confusion Matrix
                f.write("Confusion Matrix:\n")
                f.write("  Predicted:  non-ICS  ICS\n")
                f.write(f"  non-ICS:    {results['confusion_matrix'][0][0]:7d}  {results['confusion_matrix'][0][1]:3d}\n")
                f.write(f"  ICS:        {results['confusion_matrix'][1][0]:7d}  {results['confusion_matrix'][1][1]:3d}\n")
                f.write("\n")
                
                # Classification Report
                f.write("Classification Report:\n")
                f.write(results['classification_report'])
                f.write("\n" + "="*80 + "\n\n")
    
    def run_all_models(self):
        """Run all models and generate results"""
        print("Starting ICS App Classification...")
        
        # Load and prepare data
        self.load_data()
        self.prepare_features()
        
        # Train all models
        print("\nTraining all models...")
        self.train_logistic_regression()
        self.train_svm()
        self.train_naive_bayes()
        self.train_random_forest()
        self.train_lstm()
        
        # Save results
        self.save_results()
        # Final cleanup to ensure TF state is cleared after the full run
        self._cleanup_tf()
        
        # Print summary
        print("\nRESULTS SUMMARY:")
        print("-" * 70)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        
        for model_name, results in self.results.items():
            print(f"{model_name:<20} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                  f"{results['recall']:<10.4f} {results['f1_score']:<10.4f}")
        
        print(f"\nDetailed results saved to: {self.results_file}")

if __name__ == "__main__":
    classifier = ICSAppClassifier()
    classifier.run_all_models()
