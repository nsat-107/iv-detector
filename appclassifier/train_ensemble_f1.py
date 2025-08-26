#!/usr/bin/env python3
"""
Monte Carlo CV Script for Achieving F1 ≥ 0.98
============================================

This script runs BERT and BiLSTM classifiers using Monte Carlo cross-validation
with random 90-10 splits until each achieves F1 ≥ 0.98 (max 10 runs each).

Key Features:
- Random 90-10 training-test splits for each iteration
- No validation set (direct training-test evaluation)
- Continuous training (not from scratch each time)
- Sequential training: BERT first, then BiLSTM
- Detailed statistics tracking and reporting
"""

import os
import sys
import time
import json
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the existing classifiers
from ics_bert_classifier import ICSBERTClassifier
from ics_bilstm_classifier import ICSClassifier as ICSBiLSTMClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('f1_85_monte_carlo.log'),
        logging.StreamHandler()
    ]
)


class MonteCarloF1Trainer:
    """Monte Carlo trainer to achieve F1 ≥ 0.98 for single models"""
    
    def __init__(self, target_f1: float = 0.98, max_runs: int = 10, test_size: float = 0.1):
        self.target_f1 = target_f1
        self.max_runs = max_runs
        self.test_size = test_size
        self.models_dir = "models_f1_85"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Track results for each model
        self.bert_results = []
        self.bilstm_results = []
        
        # Store best models
        self.best_bert_model = None
        self.best_bilstm_model = None
        
        # Track training progress
        self.start_time = None
        
    def load_and_split_data(self, ics_file: str, non_ics_file: str, iteration: int) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Load data and create random 90-10 split for this iteration"""
        logging.info(f"Loading data for iteration {iteration}")
        
        # Load data
        ics_df = pd.read_csv(ics_file)
        non_ics_df = pd.read_csv(non_ics_file)
        
        # Extract descriptions and create labels
        ics_texts = ics_df['descriptionHtml'].fillna('').astype(str).tolist()
        non_ics_texts = non_ics_df['descriptionHtml'].fillna('').astype(str).tolist()
        
        # Combine texts and labels
        texts = ics_texts + non_ics_texts
        labels = [1] * len(ics_texts) + [0] * len(non_ics_texts)
        
        # Filter out empty texts
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(texts, labels):
            if text.strip():
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # Random 90-10 split with different seed for each iteration
        random_seed = 42 + iteration  # Different seed each time
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create indices and shuffle
        indices = list(range(len(filtered_texts)))
        random.shuffle(indices)
        
        # Split indices
        test_size = int(len(indices) * self.test_size)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        # Create splits
        X_train = [filtered_texts[i] for i in train_indices]
        X_test = [filtered_texts[i] for i in test_indices]
        y_train = [filtered_labels[i] for i in train_indices]
        y_test = [filtered_labels[i] for i in test_indices]
        
        logging.info(f"Iteration {iteration} split - Train: {len(X_train)}, Test: {len(X_test)}")
        logging.info(f"Train ICS: {sum(y_train)}, Non-ICS: {len(y_train) - sum(y_train)}")
        logging.info(f"Test ICS: {sum(y_test)}, Non-ICS: {len(y_test) - sum(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_temp_csv_files(self, X_train: List[str], y_train: List[int], iteration: int) -> Tuple[str, str]:
        """Create temporary CSV files for training data"""
        # Separate ICS and non-ICS data
        ics_data = []
        non_ics_data = []
        
        for text, label in zip(X_train, y_train):
            if label == 1:  # ICS
                ics_data.append(text)
            else:  # Non-ICS
                non_ics_data.append(text)
        
        # Create temporary CSV files
        ics_file = f"/tmp/iter_{iteration}_ics_train.csv"
        non_ics_file = f"/tmp/iter_{iteration}_non_ics_train.csv"
        
        # Write ICS file
        ics_df = pd.DataFrame({'descriptionHtml': ics_data})
        ics_df.to_csv(ics_file, index=False)
        
        # Write non-ICS file
        non_ics_df = pd.DataFrame({'descriptionHtml': non_ics_data})
        non_ics_df.to_csv(non_ics_file, index=False)
        
        return ics_file, non_ics_file
    
    def evaluate_on_test_set(self, model, X_test: List[str], y_test: List[int], model_type: str, iteration: int) -> Dict[str, float]:
        """Evaluate model on test set using predict method and calculate metrics"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Use the model's predict method to get predictions
            y_pred, y_proba = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logging.info(f"Evaluation using predict() - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            return metrics
            
        except Exception as e:
            logging.warning(f"Error in evaluation using predict(): {e}")
            logging.warning("Model may not be trained yet or predict method not available")
            
            # Fallback: return low metrics to indicate failure
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    def train_bert_until_target(self, ics_file: str, non_ics_file: str) -> Dict[str, Any]:
        """Train BERT model until F1 ≥ 0.98 or max_runs reached"""
        logging.info(f"\n{'='*80}")
        logging.info("STARTING BERT MONTE CARLO TRAINING")
        logging.info(f"Target F1: {self.target_f1}")
        logging.info(f"Max runs: {self.max_runs}")
        logging.info(f"{'='*80}")
        
        bert_classifier = None
        best_f1 = 0.0
        target_achieved = False
        
        for iteration in range(1, self.max_runs + 1):
            logging.info(f"\n🤖 BERT ITERATION {iteration}/{self.max_runs}")
            logging.info("-" * 50)
            
            try:
                # Load and split data for this iteration
                X_train, X_test, y_train, y_test = self.load_and_split_data(
                    ics_file, non_ics_file, iteration
                )
                
                # Create temporary CSV files
                train_ics_file, train_non_ics_file = self.create_temp_csv_files(
                    X_train, y_train, iteration
                )
                
                # Initialize or reuse BERT classifier
                if bert_classifier is None:
                    logging.info("Initializing new BERT classifier")
                    bert_classifier = ICSBERTClassifier()
                else:
                    logging.info("Continuing training with existing BERT classifier")
                
                # Train the model with enhanced parameters
                start_time = time.time()
                
                # Progressive training: more epochs in later iterations
                epochs = min(5 + iteration, 12)  # Start with 6, max 12
                learning_rate = 2e-5 if iteration == 1 else 1e-5  # Lower LR for continuation
                
                results = bert_classifier.train(
                    ics_file=train_ics_file,
                    non_ics_file=train_non_ics_file,
                    epochs=epochs,
                    batch_size=16,
                    learning_rate=learning_rate
                )
                training_time = time.time() - start_time
                
                logging.info(f"BERT iteration {iteration}: {epochs} epochs, LR={learning_rate}")
                
                # Extract metrics from training results
                train_f1 = results.get('f1_score', 0.0)
                train_accuracy = results.get('accuracy', 0.0)
                train_precision = results.get('precision', 0.0)
                train_recall = results.get('recall', 0.0)
                
                # Evaluate on test set
                test_metrics = self.evaluate_on_test_set(bert_classifier, X_test, y_test, 'bert', iteration)
                
                # Store results
                iteration_result = {
                    'iteration': iteration,
                    'training_time': training_time,
                    'train_metrics': {
                        'accuracy': train_accuracy,
                        'precision': train_precision,
                        'recall': train_recall,
                        'f1_score': train_f1
                    },
                    'test_metrics': test_metrics,
                    'target_achieved': test_metrics['f1_score'] >= self.target_f1
                }
                
                self.bert_results.append(iteration_result)
                
                # Log results
                logging.info(f"Training F1: {train_f1:.4f}")
                logging.info(f"Test F1: {test_metrics['f1_score']:.4f}")
                logging.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
                logging.info(f"Test Precision: {test_metrics['precision']:.4f}")
                logging.info(f"Test Recall: {test_metrics['recall']:.4f}")
                
                # Update best model and save backup if improved
                if test_metrics['f1_score'] > best_f1:
                    # Save backup of new best model
                    backup_model_path = os.path.join(self.models_dir, f'bert_backup_f1_{test_metrics["f1_score"]:.4f}_iter_{iteration}.pth')
                    backup_tokenizer_path = os.path.join(self.models_dir, f'bert_backup_tokenizer_iter_{iteration}')
                    
                    try:
                        bert_classifier.save_model(backup_model_path, backup_tokenizer_path)
                        logging.info(f"💾 BERT Backup saved (new best): F1 {test_metrics['f1_score']:.4f}")
                    except Exception as e:
                        logging.error(f"Error saving BERT backup: {e}")
                    
                    # Update best tracking
                    best_f1 = test_metrics['f1_score']
                    self.best_bert_model = bert_classifier
                    logging.info(f"🎯 NEW BEST BERT F1: {best_f1:.4f}")
                
                # Check if target achieved
                if test_metrics['f1_score'] >= self.target_f1:
                    target_achieved = True
                    logging.info(f"🎉 BERT TARGET ACHIEVED! F1: {test_metrics['f1_score']:.4f}")
                    
                    # Save the successful model
                    model_path = os.path.join(self.models_dir, f'best_bert_f1_{test_metrics["f1_score"]:.4f}_iter_{iteration}.pth')
                    tokenizer_path = os.path.join(self.models_dir, f'best_bert_tokenizer_iter_{iteration}')
                    
                    try:
                        bert_classifier.save_model(model_path, tokenizer_path)
                        logging.info(f"✅ BERT Model saved to: {model_path}")
                        logging.info(f"✅ BERT Tokenizer saved to: {tokenizer_path}")
                    except Exception as e:
                        logging.error(f"Error saving BERT model: {e}")
                    
                    break
                
                # Clean up temporary files
                os.remove(train_ics_file)
                os.remove(train_non_ics_file)
                
            except Exception as e:
                logging.error(f"Error in BERT iteration {iteration}: {e}")
                # Clean up files in case of error
                try:
                    os.remove(train_ics_file)
                    os.remove(train_non_ics_file)
                except:
                    pass
                continue
        
        # Summary
        bert_summary = {
            'target_achieved': target_achieved,
            'runs_required': iteration if target_achieved else self.max_runs,
            'best_f1': best_f1,
            'all_results': self.bert_results
        }
        
        if target_achieved:
            logging.info(f"\n✅ BERT SUCCESS: Target achieved in {iteration} runs")
        else:
            logging.info(f"\n❌ BERT: Target not achieved in {self.max_runs} runs (best F1: {best_f1:.4f})")
        
        return bert_summary
    
    def train_bilstm_until_target(self, ics_file: str, non_ics_file: str) -> Dict[str, Any]:
        """Train BiLSTM model until F1 ≥ 0.98 or max_runs reached"""
        logging.info(f"\n{'='*80}")
        logging.info("STARTING BiLSTM MONTE CARLO TRAINING")
        logging.info(f"Target F1: {self.target_f1}")
        logging.info(f"Max runs: {self.max_runs}")
        logging.info(f"{'='*80}")
        
        bilstm_classifier = None
        best_f1 = 0.0
        target_achieved = False
        
        for iteration in range(1, self.max_runs + 1):
            logging.info(f"\n🧠 BiLSTM ITERATION {iteration}/{self.max_runs}")
            logging.info("-" * 50)
            
            try:
                # Load and split data for this iteration
                X_train, X_test, y_train, y_test = self.load_and_split_data(
                    ics_file, non_ics_file, iteration
                )
                
                # Create temporary CSV files
                train_ics_file, train_non_ics_file = self.create_temp_csv_files(
                    X_train, y_train, iteration
                )
                
                # Initialize or reuse BiLSTM classifier
                if bilstm_classifier is None:
                    logging.info("Initializing new BiLSTM classifier")
                    bilstm_classifier = ICSBiLSTMClassifier()
                else:
                    logging.info("Continuing training with existing BiLSTM classifier")
                
                # Train the model with enhanced parameters
                start_time = time.time()
                
                # Progressive training: more epochs in later iterations
                epochs = min(20 + (iteration * 10), 80)  # Start with 30, max 80
                
                results = bilstm_classifier.train(
                    ics_file=train_ics_file,
                    non_ics_file=train_non_ics_file,
                    epochs=epochs,
                    batch_size=32
                )
                training_time = time.time() - start_time
                
                logging.info(f"BiLSTM iteration {iteration}: {epochs} epochs")
                
                # Extract metrics from training results
                train_f1 = results.get('f1_score', 0.0)
                train_accuracy = results.get('accuracy', 0.0)
                train_precision = results.get('precision', 0.0)
                train_recall = results.get('recall', 0.0)
                
                # Evaluate on test set
                test_metrics = self.evaluate_on_test_set(bilstm_classifier, X_test, y_test, 'bilstm', iteration)
                
                # Store results
                iteration_result = {
                    'iteration': iteration,
                    'training_time': training_time,
                    'train_metrics': {
                        'accuracy': train_accuracy,
                        'precision': train_precision,
                        'recall': train_recall,
                        'f1_score': train_f1
                    },
                    'test_metrics': test_metrics,
                    'target_achieved': test_metrics['f1_score'] >= self.target_f1
                }
                
                self.bilstm_results.append(iteration_result)
                
                # Log results
                logging.info(f"Training F1: {train_f1:.4f}")
                logging.info(f"Test F1: {test_metrics['f1_score']:.4f}")
                logging.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
                logging.info(f"Test Precision: {test_metrics['precision']:.4f}")
                logging.info(f"Test Recall: {test_metrics['recall']:.4f}")
                
                # Update best model and save backup if improved
                if test_metrics['f1_score'] > best_f1:
                    # Save backup of new best model
                    backup_model_path = os.path.join(self.models_dir, f'bilstm_backup_f1_{test_metrics["f1_score"]:.4f}_iter_{iteration}.h5')
                    backup_tokenizer_path = os.path.join(self.models_dir, f'bilstm_backup_tokenizer_iter_{iteration}.pkl')
                    
                    try:
                        bilstm_classifier.save_model(backup_model_path, backup_tokenizer_path)
                        logging.info(f"💾 BiLSTM Backup saved (new best): F1 {test_metrics['f1_score']:.4f}")
                    except Exception as e:
                        logging.error(f"Error saving BiLSTM backup: {e}")
                    
                    # Update best tracking
                    best_f1 = test_metrics['f1_score']
                    self.best_bilstm_model = bilstm_classifier
                    logging.info(f"🎯 NEW BEST BiLSTM F1: {best_f1:.4f}")
                
                # Check if target achieved
                if test_metrics['f1_score'] >= self.target_f1:
                    target_achieved = True
                    logging.info(f"🎉 BiLSTM TARGET ACHIEVED! F1: {test_metrics['f1_score']:.4f}")
                    
                    # Save the successful model
                    model_path = os.path.join(self.models_dir, f'best_bilstm_f1_{test_metrics["f1_score"]:.4f}_iter_{iteration}.h5')
                    tokenizer_path = os.path.join(self.models_dir, f'best_bilstm_tokenizer_iter_{iteration}.pkl')
                    
                    try:
                        bilstm_classifier.save_model(model_path, tokenizer_path)
                        logging.info(f"✅ BiLSTM Model saved to: {model_path}")
                        logging.info(f"✅ BiLSTM Tokenizer saved to: {tokenizer_path}")
                    except Exception as e:
                        logging.error(f"Error saving BiLSTM model: {e}")
                    
                    break
                
                # Clean up temporary files
                os.remove(train_ics_file)
                os.remove(train_non_ics_file)
                
            except Exception as e:
                logging.error(f"Error in BiLSTM iteration {iteration}: {e}")
                # Clean up files in case of error
                try:
                    os.remove(train_ics_file)
                    os.remove(train_non_ics_file)
                except:
                    pass
                continue
        
        # Summary
        bilstm_summary = {
            'target_achieved': target_achieved,
            'runs_required': iteration if target_achieved else self.max_runs,
            'best_f1': best_f1,
            'all_results': self.bilstm_results
        }
        
        if target_achieved:
            logging.info(f"\n✅ BiLSTM SUCCESS: Target achieved in {iteration} runs")
        else:
            logging.info(f"\n❌ BiLSTM: Target not achieved in {self.max_runs} runs (best F1: {best_f1:.4f})")
        
        return bilstm_summary
    
    def run_monte_carlo_training(self, ics_file: str, non_ics_file: str) -> Dict[str, Any]:
        """Run complete Monte Carlo training for both models"""
        self.start_time = time.time()
        
        logging.info("🚀 STARTING MONTE CARLO F1 TRAINING")
        logging.info(f"📊 Target F1: {self.target_f1}")
        logging.info(f"🔄 Max runs per model: {self.max_runs}")
        logging.info(f"📝 Results will be saved to: {self.models_dir}")
        
        # Train BERT first
        bert_summary = self.train_bert_until_target(ics_file, non_ics_file)
        
        # Train BiLSTM second
        bilstm_summary = self.train_bilstm_until_target(ics_file, non_ics_file)
        
        # Calculate total time
        total_time = time.time() - self.start_time
        
        # Compile final results
        final_results = {
            'training_time': total_time,
            'target_f1': self.target_f1,
            'max_runs': self.max_runs,
            'bert_summary': bert_summary,
            'bilstm_summary': bilstm_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self.save_final_results(final_results)
        
        # Print final statistics
        self.print_final_statistics(final_results)
        
        return final_results
    
    def save_final_results(self, results: Dict[str, Any]):
        """Save final results to JSON file"""
        results_file = os.path.join(self.models_dir, 'monte_carlo_f1_85_results.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"📁 Results saved to: {results_file}")
    
    def print_final_statistics(self, results: Dict[str, Any]):
        """Print comprehensive final statistics"""
        logging.info(f"\n{'='*100}")
        logging.info("FINAL MONTE CARLO TRAINING STATISTICS")
        logging.info(f"{'='*100}")
        
        total_time = results['training_time']
        logging.info(f"⏱️  Total Training Time: {total_time/60:.1f} minutes")
        logging.info(f"🎯 Target F1 Score: {self.target_f1}")
        logging.info(f"🔄 Maximum Runs Per Model: {self.max_runs}")
        
        # BERT Statistics
        bert_summary = results['bert_summary']
        logging.info(f"\n🤖 BERT RESULTS:")
        logging.info(f"  ✅ Target Achieved: {'YES' if bert_summary['target_achieved'] else 'NO'}")
        logging.info(f"  🔢 Runs Required: {bert_summary['runs_required']}")
        logging.info(f"  🏆 Best F1 Score: {bert_summary['best_f1']:.4f}")
        
        if bert_summary['all_results']:
            # Find the best performing iteration
            best_bert_iter = max(bert_summary['all_results'], key=lambda x: x['test_metrics']['f1_score'])
            final_bert = bert_summary['all_results'][-1]
            
            logging.info(f"  📊 BEST Performance (Iteration {best_bert_iter['iteration']}):")
            logging.info(f"    • Accuracy:  {best_bert_iter['test_metrics']['accuracy']:.4f}")
            logging.info(f"    • Precision: {best_bert_iter['test_metrics']['precision']:.4f}")
            logging.info(f"    • Recall:    {best_bert_iter['test_metrics']['recall']:.4f}")
            logging.info(f"    • F1-Score:  {best_bert_iter['test_metrics']['f1_score']:.4f}")
            logging.info(f"    • Training Time: {best_bert_iter['training_time']:.1f}s")
            
            if best_bert_iter != final_bert:
                logging.info(f"  📊 Final Iteration ({final_bert['iteration']}) Metrics:")
                logging.info(f"    • Accuracy:  {final_bert['test_metrics']['accuracy']:.4f}")
                logging.info(f"    • Precision: {final_bert['test_metrics']['precision']:.4f}")
                logging.info(f"    • Recall:    {final_bert['test_metrics']['recall']:.4f}")
                logging.info(f"    • F1-Score:  {final_bert['test_metrics']['f1_score']:.4f}")
        
        # BiLSTM Statistics
        bilstm_summary = results['bilstm_summary']
        logging.info(f"\n🧠 BiLSTM RESULTS:")
        logging.info(f"  ✅ Target Achieved: {'YES' if bilstm_summary['target_achieved'] else 'NO'}")
        logging.info(f"  🔢 Runs Required: {bilstm_summary['runs_required']}")
        logging.info(f"  🏆 Best F1 Score: {bilstm_summary['best_f1']:.4f}")
        
        if bilstm_summary['all_results']:
            # Find the best performing iteration
            best_bilstm_iter = max(bilstm_summary['all_results'], key=lambda x: x['test_metrics']['f1_score'])
            final_bilstm = bilstm_summary['all_results'][-1]
            
            logging.info(f"  📊 BEST Performance (Iteration {best_bilstm_iter['iteration']}):")
            logging.info(f"    • Accuracy:  {best_bilstm_iter['test_metrics']['accuracy']:.4f}")
            logging.info(f"    • Precision: {best_bilstm_iter['test_metrics']['precision']:.4f}")
            logging.info(f"    • Recall:    {best_bilstm_iter['test_metrics']['recall']:.4f}")
            logging.info(f"    • F1-Score:  {best_bilstm_iter['test_metrics']['f1_score']:.4f}")
            logging.info(f"    • Training Time: {best_bilstm_iter['training_time']:.1f}s")
            
            if best_bilstm_iter != final_bilstm:
                logging.info(f"  📊 Final Iteration ({final_bilstm['iteration']}) Metrics:")
                logging.info(f"    • Accuracy:  {final_bilstm['test_metrics']['accuracy']:.4f}")
                logging.info(f"    • Precision: {final_bilstm['test_metrics']['precision']:.4f}")
                logging.info(f"    • Recall:    {final_bilstm['test_metrics']['recall']:.4f}")
                logging.info(f"    • F1-Score:  {final_bilstm['test_metrics']['f1_score']:.4f}")
        
        # Overall Summary
        logging.info(f"\n📈 OVERALL SUMMARY:")
        success_count = sum([bert_summary['target_achieved'], bilstm_summary['target_achieved']])
        logging.info(f"  🎯 Models Achieving Target: {success_count}/2")
        total_runs = bert_summary['runs_required'] + bilstm_summary['runs_required']
        logging.info(f"  🔄 Total Training Runs: {total_runs}")
        logging.info(f"  ⚡ Average Time Per Run: {(total_time/total_runs)/60:.1f} minutes")
        
        if success_count == 2:
            logging.info(f"\n🎉 MISSION ACCOMPLISHED! Both models achieved F1 ≥ {self.target_f1}")
        elif success_count == 1:
            logging.info(f"\n⚡ PARTIAL SUCCESS: One model achieved F1 ≥ {self.target_f1}")
        else:
            logging.info(f"\n🔄 CONTINUE TRAINING: Neither model achieved F1 ≥ {self.target_f1}")
        
        # Show saved model files
        logging.info(f"\n💾 SAVED MODEL FILES:")
        import glob
        
        # Find all model files in the models directory
        bert_models = glob.glob(os.path.join(self.models_dir, "best_bert_*.pth")) + glob.glob(os.path.join(self.models_dir, "bert_backup_*.pth"))
        bilstm_models = glob.glob(os.path.join(self.models_dir, "best_bilstm_*.h5")) + glob.glob(os.path.join(self.models_dir, "bilstm_backup_*.h5"))
        
        if bert_models:
            logging.info(f"  🤖 BERT Models:")
            for model_file in sorted(bert_models):
                filename = os.path.basename(model_file)
                logging.info(f"    • {filename}")
        
        if bilstm_models:
            logging.info(f"  🧠 BiLSTM Models:")
            for model_file in sorted(bilstm_models):
                filename = os.path.basename(model_file)
                logging.info(f"    • {filename}")
        
        # Show tokenizer files
        bert_tokenizers = glob.glob(os.path.join(self.models_dir, "best_bert_tokenizer_*")) + glob.glob(os.path.join(self.models_dir, "bert_backup_tokenizer_*"))
        bilstm_tokenizers = glob.glob(os.path.join(self.models_dir, "best_bilstm_tokenizer_*.pkl")) + glob.glob(os.path.join(self.models_dir, "bilstm_backup_tokenizer_*.pkl"))
        
        if bert_tokenizers:
            logging.info(f"  🔤 BERT Tokenizers:")
            for tokenizer_dir in sorted(bert_tokenizers):
                dirname = os.path.basename(tokenizer_dir)
                logging.info(f"    • {dirname}/")
        
        if bilstm_tokenizers:
            logging.info(f"  🔤 BiLSTM Tokenizers:")
            for tokenizer_file in sorted(bilstm_tokenizers):
                filename = os.path.basename(tokenizer_file)
                logging.info(f"    • {filename}")
        
        logging.info(f"\n📁 All files saved in: {self.models_dir}/")
        logging.info(f"{'='*100}")


def main():
    """Main function to run Monte Carlo F1 training"""
    # Configuration
    TARGET_F1 = 0.98
    MAX_RUNS = 10
    
    # Data files (adjust paths as needed)
    ICS_FILE = "training_data_ics.csv"
    NON_ICS_FILE = "training_data_non_ics.csv"
    
    # Check if data files exist
    if not os.path.exists(ICS_FILE):
        logging.error(f"ICS data file not found: {ICS_FILE}")
        sys.exit(1)
    
    if not os.path.exists(NON_ICS_FILE):
        logging.error(f"Non-ICS data file not found: {NON_ICS_FILE}")
        sys.exit(1)
    
    # Initialize and run trainer
    trainer = MonteCarloF1Trainer(target_f1=TARGET_F1, max_runs=MAX_RUNS)
    results = trainer.run_monte_carlo_training(ICS_FILE, NON_ICS_FILE)
    
    logging.info("🏁 Monte Carlo F1 training completed!")


if __name__ == "__main__":
    main()
