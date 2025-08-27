#!/usr/bin/env python3
"""
ICS Application BiLSTM Classifier
"""

import pandas as pd
import numpy as np
import tensorflow as tf
# Use TensorFlow's Keras instead of standalone Keras to avoid compatibility issues
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM, GlobalMaxPool1D, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import json
import os
from typing import Tuple, List
import pickle

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing utilities from the reference implementation
import sys
from preprocess import preprocess_one_description
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ICSClassifier:
    def __init__(self, embedding_dim=300, max_features=3000, max_len=500):
        """
        Initialize the ICS BiLSTM classifier
        
        Args:
            embedding_dim: Dimension of word embeddings (300 for GloVe)
            max_features: Maximum number of words to keep in vocabulary
            max_len: Maximum sequence length
        """
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self.max_len = max_len
        self.tokenizer = None
        self.model = None
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text description using the IoTSpotter preprocessing pipeline
        
        Args:
            text: Raw HTML description text
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Use the preprocessing function from reference implementation
        try:
            processed = preprocess_one_description(str(text), enable_langid=False)
            return processed if processed else ''
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            # Fallback basic preprocessing
            text = str(text).lower()
            text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
            text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
            return text.strip()
    
    def load_and_preprocess_data(self, ics_file: str, non_ics_file: str) -> Tuple[List[str], List[int]]:
        """
        Load and preprocess training data
        
        Args:
            ics_file: Path to ICS training data CSV
            non_ics_file: Path to non-ICS training data CSV
            
        Returns:
            Tuple of (texts, labels)
        """
        print("Loading ICS training data...")
        ics_df = pd.read_csv(ics_file)
        print(f"Loaded {len(ics_df)} ICS applications")
        
        print("Loading non-ICS training data...")
        non_ics_df = pd.read_csv(non_ics_file)
        print(f"Loaded {len(non_ics_df)} non-ICS applications")
        
        # Extract descriptions and create labels
        ics_texts = [self.preprocess_text(desc) for desc in ics_df['descriptionHtml'].values]
        non_ics_texts = [self.preprocess_text(desc) for desc in non_ics_df['descriptionHtml'].values]
        
        # Combine texts and labels
        texts = ics_texts + non_ics_texts
        labels = [1] * len(ics_texts) + [0] * len(non_ics_texts)  # 1 for ICS, 0 for non-ICS
        
        # Filter out empty texts
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(texts, labels):
            if text.strip():
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        print(f"Total samples after preprocessing: {len(filtered_texts)}")
        print(f"ICS samples: {sum(filtered_labels)}, Non-ICS samples: {len(filtered_labels) - sum(filtered_labels)}")
        
        return filtered_texts, filtered_labels
    
    def split_data(self, texts: List[str], labels: List[int], val_size_per_class: int = 13) -> Tuple:
        """
        Split data into training and validation sets with equal class distribution
        
        Args:
            texts: List of preprocessed texts
            labels: List of labels (0 or 1)
            val_size_per_class: Number of samples per class for validation (13 each)
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        # Separate by class
        ics_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]
        non_ics_texts = [texts[i] for i in range(len(texts)) if labels[i] == 0]
        
        print(f"Available ICS samples: {len(ics_texts)}")
        print(f"Available non-ICS samples: {len(non_ics_texts)}")
        
        # Ensure we have enough samples
        if len(ics_texts) < val_size_per_class:
            raise ValueError(f"Not enough ICS samples. Need at least {val_size_per_class}, got {len(ics_texts)}")
        if len(non_ics_texts) < val_size_per_class:
            raise ValueError(f"Not enough non-ICS samples. Need at least {val_size_per_class}, got {len(non_ics_texts)}")
        
        # Ensure balanced training by taking equal number from each class
        min_available = min(len(ics_texts), len(non_ics_texts))
        train_size_per_class = min_available - val_size_per_class
        
        print(f"Using {train_size_per_class} training samples per class for balanced training")
        
        # Split each class - validation comes from the end
        ics_val = ics_texts[-val_size_per_class:]
        ics_train = ics_texts[:train_size_per_class]  # Take exactly train_size_per_class
        
        non_ics_val = non_ics_texts[-val_size_per_class:]
        non_ics_train = non_ics_texts[:train_size_per_class]  # Take exactly train_size_per_class
        
        # Combine training and validation sets
        X_train = ics_train + non_ics_train
        y_train = [1] * len(ics_train) + [0] * len(non_ics_train)
        
        X_val = ics_val + non_ics_val
        y_val = [1] * len(ics_val) + [0] * len(non_ics_val)
        
        print(f"Training set: {len(X_train)} samples ({len(ics_train)} ICS, {len(non_ics_train)} non-ICS)")
        print(f"Validation set: {len(X_val)} samples ({len(ics_val)} ICS, {len(non_ics_val)} non-ICS)")
        
        return X_train, X_val, y_train, y_val
    
    def build_model(self) -> Sequential:
        """
        Build the BiLSTM model architecture based on the reference implementation
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Embedding layer (will be updated with pre-trained weights if available)
        model.add(Embedding(
            input_dim=self.max_features + 1,
            output_dim=self.embedding_dim,
            input_length=self.max_len,
            trainable=True
        ))
        
        # Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=300, return_sequences=True)))
        
        # Global max pooling
        model.add(GlobalMaxPool1D())
        
        # Dense layers with dropout and batch normalization
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def train(self, ics_file: str, non_ics_file: str, epochs: int = 50, batch_size: int = 32) -> dict:
        """
        Train the BiLSTM classifier
        
        Args:
            ics_file: Path to ICS training data CSV
            non_ics_file: Path to non-ICS training data CSV
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history
        """
        print("Starting BiLSTM classifier training...")
        
        # Load and preprocess data
        texts, labels = self.load_and_preprocess_data(ics_file, non_ics_file)
        
        # Split data
        X_train, X_val, y_train, y_val = self.split_data(texts, labels)
        
        # Initialize and fit tokenizer
        print("Fitting tokenizer...")
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train + X_val)
        
        # Convert texts to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len)
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_len)
        
        # Convert labels to numpy arrays
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        
        print(f"Training data shape: {X_train_pad.shape}")
        print(f"Validation data shape: {X_val_pad.shape}")
        
        # Build model
        self.model = self.build_model()
        
        # Train model
        print("Starting model training...")
        history = self.model.fit(
            X_train_pad, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_pad, y_val),
            verbose=1
        )
        
        # Evaluate on validation set
        print("\nEvaluation on validation set:")
        val_loss, val_acc = self.model.evaluate(X_val_pad, y_val, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # Make predictions for detailed evaluation
        y_pred_prob = self.model.predict(X_val_pad)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate detailed metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        accuracy = (y_pred == y_val).mean()
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print("\n" + "="*60)
        print("BILSTM ICS CLASSIFIER - DETAILED RESULTS")
        print("="*60)
        print(f"Overall Accuracy:  {accuracy:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall:    {recall:.4f}")
        print(f"Overall F1-Score:  {f1:.4f}")
        print("="*60)
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Non-ICS', 'ICS']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_val, y_pred)
        print(cm)
        print(f"\nConfusion Matrix Breakdown:")
        print(f"True Negatives (Non-ICS correctly classified): {cm[0,0]}")
        print(f"False Positives (Non-ICS misclassified as ICS): {cm[0,1]}")
        print(f"False Negatives (ICS misclassified as Non-ICS): {cm[1,0]}")
        print(f"True Positives (ICS correctly classified): {cm[1,1]}")
        
        print("\n" + "="*60)
        
        return history.history
    
    def save_model(self, model_path: str, tokenizer_path: str):
        """
        Save the trained model and tokenizer
        
        Args:
            model_path: Path to save the Keras model
            tokenizer_path: Path to save the tokenizer
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_path: str, tokenizer_path: str):
        """
        Load a pre-trained model and tokenizer
        
        Args:
            model_path: Path to the saved Keras model
            tokenizer_path: Path to the saved tokenizer
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from {tokenizer_path}")
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of text descriptions to classify
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded or trained first")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Convert to sequences and pad
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        
        # Make predictions
        probabilities = self.model.predict(padded)
        predictions = (probabilities > 0.5).astype(int).flatten()
        
        return predictions, probabilities.flatten()


def main():
    """
    Main function to train the BiLSTM classifier
    """
    # Initialize classifier
    classifier = ICSClassifier(
        embedding_dim=300,
        max_features=3000,
        max_len=500
    )
    
    # File paths (relative to this script)
    ics_file = os.path.join(BASE_DIR, 'training_data_ics.csv')
    non_ics_file = os.path.join(BASE_DIR, 'training_data_non_ics.csv')
    
    # Train the model
    history = classifier.train(
        ics_file=ics_file,
        non_ics_file=non_ics_file,
        epochs=50,
        batch_size=32
    )
    
    # Save the model under a local 'model' directory beside this script
    model_dir = os.path.join(BASE_DIR, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'ics_bilstm_classifier.h5')
    tokenizer_path = os.path.join(model_dir, 'ics_bilstm_tokenizer.pkl')
    
    classifier.save_model(model_path, tokenizer_path)
    
    print("BiLSTM classifier training completed successfully!")


if __name__ == '__main__':
    main()