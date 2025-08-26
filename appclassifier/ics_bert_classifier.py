#!/usr/bin/env python3
"""
BERT-based ICS Application Classifier
Step 2 of the ICS classification pipeline using transformer-based approach.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import os
import pickle
from typing import Tuple, List, Dict
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing utilities from the reference implementation
import sys
sys.path.append('/home/asurite.ad.asu.edu/asawan15/ad_research/ivdetector/IoTSpotter/classification')
from preprocess import preprocess_one_description


class ICSDataset(Dataset):
    """
    PyTorch Dataset for ICS classification with BERT tokenization
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            # Clean and validate text
            if not text or text.isspace():
                text = "No description available"
            
            # Remove problematic characters and limit length
            text = text.replace('\x00', '').replace('\ufffd', '').strip()
            if len(text) > 10000:  # Reasonable limit before tokenization
                text = text[:10000]
            
            # Tokenize with BERT tokenizer
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=True,
                return_attention_mask=True
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Warning: Error processing text at index {idx}: {e}")
            # Return a safe fallback
            fallback_text = "Error processing text"
            encoding = self.tokenizer(
                fallback_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=True,
                return_attention_mask=True
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }


class BERTICSClassifier(nn.Module):
    """
    BERT-based classifier for ICS application detection
    
    Architecture Rationale:
    - Uses pre-trained BERT-base-uncased for general domain knowledge
    - Adds dropout for regularization to prevent overfitting on small dataset
    - Single linear classification head on [CLS] token representation
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2, dropout_rate: float = 0.3):
        """
        Initialize BERT classifier
        
        Args:
            model_name: Pre-trained BERT model variant
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout probability for regularization
        """
        super(BERTICSClassifier, self).__init__()
        
        # Load pre-trained BERT model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize classifier layer with small random weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class ICSBERTClassifier:
    """
    Main class for training and using BERT-based ICS classifier
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        """
        Initialize BERT ICS classifier
        
        Parameter Rationale:
        - model_name: 'bert-base-uncased' chosen for:
          * Good balance between performance and computational efficiency
          * 110M parameters - suitable for our dataset size (~440 samples)
          * Uncased to handle mixed-case text in app descriptions
          * Well-established baseline for text classification
        
        - max_length: 512 tokens chosen because:
          * BERT's maximum sequence length
          * Allows capturing full context of app descriptions
          * Most app descriptions will fit within this limit
          * Longer than BiLSTM's 500 for fair comparison
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text description for BERT
        
        Note: BERT handles much of the preprocessing internally via its tokenizer,
        but we still apply some domain-specific preprocessing for consistency
        """
        if pd.isna(text) or text == '':
            return ''
        
        try:
            # Use the same preprocessing as BiLSTM for consistency
            processed = preprocess_one_description(str(text), enable_langid=False)
            return processed if processed else str(text).lower()
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            # Fallback: basic preprocessing
            text = str(text).lower()
            text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
            return text.strip()
    
    def load_and_preprocess_data(self, ics_file: str, non_ics_file: str) -> Tuple[List[str], List[int]]:
        """
        Load and preprocess training data
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
        (Same split as BiLSTM for fair comparison)
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
        
        # Split each class - validation comes from the end
        ics_val = ics_texts[-val_size_per_class:]
        ics_train = ics_texts[:-val_size_per_class]
        
        non_ics_val = non_ics_texts[-val_size_per_class:]
        non_ics_train = non_ics_texts[:-val_size_per_class]
        
        # Combine training and validation sets
        X_train = ics_train + non_ics_train
        y_train = [1] * len(ics_train) + [0] * len(non_ics_train)
        
        X_val = ics_val + non_ics_val
        y_val = [1] * len(ics_val) + [0] * len(non_ics_val)
        
        print(f"Training set: {len(X_train)} samples ({len(ics_train)} ICS, {len(non_ics_train)} non-ICS)")
        print(f"Validation set: {len(X_val)} samples ({len(ics_val)} ICS, {len(non_ics_val)} non-ICS)")
        
        return X_train, X_val, y_train, y_val
    
    def create_data_loaders(self, X_train, X_val, y_train, y_val, batch_size: int = 16):
        """
        Create PyTorch DataLoaders
        
        Batch Size Rationale:
        - 16 chosen as optimal balance for BERT fine-tuning:
          * Large enough for stable gradients
          * Small enough to fit in memory with 512 sequence length
          * Commonly used in BERT fine-tuning literature
          * Allows for gradient accumulation if needed
        """
        train_dataset = ICSDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = ICSDataset(X_val, y_val, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, ics_file: str, non_ics_file: str, epochs: int = 4, batch_size: int = 16, 
              learning_rate: float = 2e-5, warmup_steps_ratio: float = 0.1) -> Dict:
        """
        Train the BERT classifier
        
        Training Parameter Rationale:
        
        - epochs=4: 
          * BERT fine-tuning typically requires fewer epochs (3-5)
          * More epochs risk overfitting on small dataset
          * 4 epochs provides good balance between training and generalization
        
        - learning_rate=2e-5:
          * Standard BERT fine-tuning learning rate
          * Smaller than typical training (1e-3) to preserve pre-trained weights
          * Range 2e-5 to 5e-5 is well-established for BERT fine-tuning
          * Prevents catastrophic forgetting of pre-trained representations
        
        - warmup_steps_ratio=0.1:
          * Learning rate warmup is crucial for transformer training
          * Gradual increase from 0 to target LR in first 10% of training
          * Helps stabilize training and improves final performance
          * Standard practice in transformer fine-tuning
        
        - AdamW optimizer:
          * Weight decay regularization prevents overfitting
          * Better than standard Adam for transformer models
          * Decouples weight decay from gradient updates
        """
        
        print("=" * 60)
        print("BERT ICS CLASSIFIER TRAINING")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Max sequence length: {self.max_length}")
        print(f"Training epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Warmup ratio: {warmup_steps_ratio}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Load and preprocess data
        texts, labels = self.load_and_preprocess_data(ics_file, non_ics_file)
        X_train, X_val, y_train, y_val = self.split_data(texts, labels)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(X_train, X_val, y_train, y_val, batch_size)
        
        # Initialize model
        self.model = BERTICSClassifier(self.model_name, num_classes=2, dropout_rate=0.3)
        self.model.to(self.device)
        
        # Optimizer and scheduler setup
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_steps_ratio)
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,  # L2 regularization
            eps=1e-8  # Numerical stability
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_accuracy = 0
        
        print("Starting BERT training...")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in train_pbar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
                for batch in val_pbar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    total_val_loss += loss.item()
                    
                    predictions = torch.argmax(outputs, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_predictions)
            
            # Track best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            
            # Record metrics
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Best Val Accuracy: {best_val_accuracy:.4f}")
        
        # Final evaluation with detailed metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        final_accuracy = accuracy_score(all_labels, all_predictions)
        final_precision = precision_score(all_labels, all_predictions)
        final_recall = recall_score(all_labels, all_predictions)
        final_f1 = f1_score(all_labels, all_predictions)
        
        print("\n" + "=" * 60)
        print("BERT ICS CLASSIFIER - DETAILED RESULTS")
        print("=" * 60)
        print(f"Overall Accuracy:  {final_accuracy:.4f}")
        print(f"Overall Precision: {final_precision:.4f}")
        print(f"Overall Recall:    {final_recall:.4f}")
        print(f"Overall F1-Score:  {final_f1:.4f}")
        print(f"Best Val Accuracy: {best_val_accuracy:.4f}")
        print("=" * 60)
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['Non-ICS', 'ICS']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_predictions)
        print(cm)
        print(f"\nConfusion Matrix Breakdown:")
        print(f"True Negatives (Non-ICS correctly classified): {cm[0,0]}")
        print(f"False Positives (Non-ICS misclassified as ICS): {cm[0,1]}")
        print(f"False Negatives (ICS misclassified as Non-ICS): {cm[1,0]}")
        print(f"True Positives (ICS correctly classified): {cm[1,1]}")
        
        print("\n" + "=" * 60)
        
        return history
    
    def save_model(self, model_path: str, tokenizer_path: str):
        """Save the trained model and tokenizer"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length
        }, model_path)
        print(f"Model saved to {model_path}")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_path: str, tokenizer_path: str):
        """Load a pre-trained model and tokenizer"""
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = BERTICSClassifier(checkpoint['model_name'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Tokenizer loaded from {tokenizer_path}")
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new texts"""
        if self.model is None:
            raise ValueError("Model must be loaded or trained first")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create dataset and dataloader with memory optimization
        dataset = ICSDataset(processed_texts, [0] * len(processed_texts), self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=16, 
            shuffle=False,
            num_workers=2,
            pin_memory=True, 
            drop_last=False
        )
        
        # Make predictions
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # ICS probability
        
        return np.array(all_predictions), np.array(all_probabilities)


def main():
    """Main function to train the BERT classifier"""
    
    # Initialize classifier
    classifier = ICSBERTClassifier(
        model_name='bert-base-uncased',
        max_length=512
    )
    
    # File paths
    ics_file = '/home/asurite.ad.asu.edu/asawan15/ad_research/ivdetector/training_data_ics.csv'
    non_ics_file = '/home/asurite.ad.asu.edu/asawan15/ad_research/ivdetector/training_data_non_ics.csv'
    
    # Train the model
    history = classifier.train(
        ics_file=ics_file,
        non_ics_file=non_ics_file,
        epochs=4,
        batch_size=16,
        learning_rate=2e-5,
        warmup_steps_ratio=0.1
    )
    
    # Save the model
    model_dir = '/home/asurite.ad.asu.edu/asawan15/ad_research/ivdetector/models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'ics_bert_classifier.pth')
    tokenizer_path = os.path.join(model_dir, 'ics_bert_tokenizer')
    
    classifier.save_model(model_path, tokenizer_path)
    
    print("\nBERT classifier training completed successfully!")


if __name__ == '__main__':
    main()