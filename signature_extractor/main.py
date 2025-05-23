import os
import argparse
import numpy as np
import json
import torch
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from extractors import SmaliExtractor, ManifestExtractor, DescriptionExtractor
from graphcodebert_embedder import GraphCodeBertEmbedder
from trainer import SignatureTrainer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract signatures from APKs using GraphCodeBERT')
    
    # Required arguments
    parser.add_argument('--apps_dir', type=str, required=False,
                        help='Directory containing extracted APK directories')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store output files')
    
    # Dataset arguments (new)
    parser.add_argument('--dataset_mode', type=str, choices=['single_dir', 'train_test'], default='single_dir',
                        help='Dataset mode: single_dir or train_test')
    parser.add_argument('--train_ics_dir', type=str, 
                        help='Directory containing ICS apps for training')
    parser.add_argument('--train_non_ics_dir', type=str, 
                        help='Directory containing non-ICS apps for training')
    parser.add_argument('--test_ics_dir', type=str, 
                        help='Directory containing ICS apps for testing')
    parser.add_argument('--test_non_ics_dir', type=str, 
                        help='Directory containing non-ICS apps for testing')
    
    # Optional arguments
    parser.add_argument('--include_description', action='store_true',
                        help='Include app description in the signature (placeholder)')
    parser.add_argument('--train_contrastive', action='store_true',
                        help='Train a contrastive model to generate 128D signatures')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of the final signature vector')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--cluster', action='store_true',
                        help='Perform clustering on the signatures')
    parser.add_argument('--cluster_algorithm', type=str, default='dbscan',
                        choices=['dbscan', 'kmeans'],
                        help='Clustering algorithm to use')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Epsilon parameter for DBSCAN')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Min samples parameter for DBSCAN')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters for K-Means')
    parser.add_argument('--is_ics', action='store_true',
                        help='Label all apps as ICS apps (for training with only ICS apps)')
    parser.add_argument('--all_ics', action='store_true',
                        help='All apps in the dataset are ICS apps')
    parser.add_argument('--ics_labels', type=str, default=None,
                        help='Path to file with ICS labels (if available)')
    
    return parser.parse_args()

def get_app_dirs(base_dir):
    """Get all app directories"""
    return [d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))]

def process_apps(app_dirs, base_dir, include_description=False):
    """Process all apps to extract necessary data"""
    app_data = []
    
    for app_dir in app_dirs:
        app_path = os.path.join(base_dir, app_dir)
        try:
            # Extract smali data
            smali_input = SmaliExtractor.get_graphcodebert_input(app_path)
            
            # Extract manifest data
            manifest_features = ManifestExtractor.extract_features(app_path)
            manifest_input = ManifestExtractor.get_graphcodebert_input(manifest_features)
            
            # Extract description if needed
            description_input = None
            if include_description:
                description_text = DescriptionExtractor.get_description(app_path)
                description_input = description_text
            
            # Create app data entry
            app_entry = {
                'app_name': app_dir,
                'smali_input': smali_input,
                'manifest_input': manifest_input
            }
            
            if include_description:
                app_entry['description_input'] = description_input
                
            app_data.append(app_entry)
            print(f"Processed app: {app_dir}")
            
        except Exception as e:
            print(f"Error processing {app_dir}: {e}")
    
    print(f"Processed {len(app_data)} apps successfully.")
    return app_data

def get_labels(app_names, args):
    """Get labels for apps (ICS or non-ICS)"""
    if args.all_ics:
        # All apps are ICS apps
        return np.ones(len(app_names), dtype=np.int32)
    
    if args.ics_labels:
        # Load labels from file
        try:
            with open(args.ics_labels, 'r') as f:
                label_data = json.load(f)
            
            # Convert to numpy array
            labels = np.array([label_data.get(name, 0) for name in app_names], dtype=np.int32)
            return labels
        except Exception as e:
            print(f"Error loading labels: {e}")
            # Fall back to all apps being ICS
            return np.ones(len(app_names), dtype=np.int32)
    
    # Default: all apps are ICS (since we're processing ICS_Apps directory)
    return np.ones(len(app_names), dtype=np.int32)

def cluster_signatures(signatures, app_names, algorithm='dbscan', 
                      eps=0.5, min_samples=5, n_clusters=5):
    """Cluster app signatures"""
    print(f"Clustering signatures using {algorithm}...")
    
    if algorithm.lower() == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(signatures)
        labels = clustering.labels_
    elif algorithm.lower() == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(signatures)
        labels = clustering.labels_
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
    # Group apps by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(app_names[i])
    
    # Print cluster information
    print(f"Found {len(clusters)} clusters")
    for label, apps in clusters.items():
        if label == -1:
            print(f"Noise points: {len(apps)} apps")
        else:
            print(f"Cluster {label}: {len(apps)} apps")
    
    return clusters, labels

def process_train_test_dataset(args):
    """Process train/test datasets"""
    # Create output directories
    train_output_dir = os.path.join(args.output_dir, 'train')
    test_output_dir = os.path.join(args.output_dir, 'test')
    metrics_dir = os.path.join(args.output_dir, 'metrics')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    train_ics_dirs = get_app_dirs(args.train_ics_dir)
    train_non_ics_dirs = get_app_dirs(args.train_non_ics_dir)
    
    train_ics_data = process_apps(train_ics_dirs, args.train_ics_dir, args.include_description)
    train_non_ics_data = process_apps(train_non_ics_dirs, args.train_non_ics_dir, args.include_description)
    
    # Combine ICS and non-ICS data
    train_data = train_ics_data + train_non_ics_data
    train_app_names = [app['app_name'] for app in train_data]
    
    # Create labels (1 for ICS, 0 for non-ICS)
    train_labels = np.zeros(len(train_data), dtype=np.int32)
    train_labels[:len(train_ics_data)] = 1  # First part is ICS apps
    
    # Initialize GraphCodeBERT embedder
    embedder = GraphCodeBertEmbedder()
    
    # Get embeddings for training data
    print("Generating GraphCodeBERT embeddings for training data...")
    embedding_start_time = time.time()
    train_embeddings = embedder.embed_app_batch(train_data, args.include_description)
    embedding_time = time.time() - embedding_start_time
    print(f"Embedding generation time for training data: {embedding_time:.2f} seconds")
    
    # Save training embeddings
    train_embeddings_file = os.path.join(train_output_dir, 'graphcodebert_embeddings.npy')
    np.save(train_embeddings_file, train_embeddings)
    
    # Save training app names
    train_names_file = os.path.join(train_output_dir, 'app_names.json')
    with open(train_names_file, 'w') as f:
        json.dump(train_app_names, f)
    
    # Save training labels
    train_labels_file = os.path.join(train_output_dir, 'is_ics_labels.npy')
    np.save(train_labels_file, train_labels)
    
    # Train contrastive model if requested
    if args.train_contrastive:
        print("Training contrastive model...")
        trainer = SignatureTrainer(train_output_dir)
        
        training_start_time = time.time()
        model, train_signatures = trainer.train_model(
            train_embeddings, 
            train_labels,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
        training_time = time.time() - training_start_time
        print(f"Training time: {training_time:.2f} seconds")
        
        # Save signatures
        trainer.save_signatures(train_signatures, train_app_names, train_labels)
        
        # Process test data
        print("Processing test data...")
        test_ics_dirs = get_app_dirs(args.test_ics_dir)
        test_non_ics_dirs = get_app_dirs(args.test_non_ics_dir)
        
        test_ics_data = process_apps(test_ics_dirs, args.test_ics_dir, args.include_description)
        test_non_ics_data = process_apps(test_non_ics_dirs, args.test_non_ics_dir, args.include_description)
        
        # Combine ICS and non-ICS test data
        test_data = test_ics_data + test_non_ics_data
        test_app_names = [app['app_name'] for app in test_data]
        
        # Create test labels
        test_labels = np.zeros(len(test_data), dtype=np.int32)
        test_labels[:len(test_ics_data)] = 1  # First part is ICS apps
        
        # Get embeddings for test data
        print("Generating GraphCodeBERT embeddings for test data...")
        test_start_time = time.time()
        test_embeddings = embedder.embed_app_batch(test_data, args.include_description)
        test_embedding_time = time.time() - test_start_time
        print(f"Embedding generation time for test data: {test_embedding_time:.2f} seconds")
        
        # Save test embeddings
        test_embeddings_file = os.path.join(test_output_dir, 'graphcodebert_embeddings.npy')
        np.save(test_embeddings_file, test_embeddings)
        
        # Save test app names
        test_names_file = os.path.join(test_output_dir, 'app_names.json')
        with open(test_names_file, 'w') as f:
            json.dump(test_app_names, f)
        
        # Save test labels
        test_labels_file = os.path.join(test_output_dir, 'is_ics_labels.npy')
        np.save(test_labels_file, test_labels)
        
        # Generate signatures for test data using the trained model
        print("Generating signatures for test data...")
        model.eval()
        inference_start_time = time.time()
        test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(trainer.device)
        with torch.no_grad():
            test_signatures = model.encode(test_embeddings_tensor).cpu().numpy()
        inference_time = time.time() - inference_start_time
        print(f"Inference time for test data: {inference_time:.2f} seconds")
        
        # Save test signatures
        test_signatures_file = os.path.join(test_output_dir, 'app_signatures.npy')
        np.save(test_signatures_file, test_signatures)
        
        # Evaluate signature performance using classifiers
        print("Evaluating signature performance...")
        
        # 1. K-Nearest Neighbors Classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_signatures, train_labels)
        knn_predictions = knn.predict(test_signatures)
        
        # 2. SVM Classifier
        svm = SVC()
        svm.fit(train_signatures, train_labels)
        svm_predictions = svm.predict(test_signatures)
        
        # Calculate metrics
        metrics = {
            "knn": {
                "accuracy": accuracy_score(test_labels, knn_predictions),
                "precision": precision_score(test_labels, knn_predictions),
                "recall": recall_score(test_labels, knn_predictions),
                "f1": f1_score(test_labels, knn_predictions),
                "confusion_matrix": confusion_matrix(test_labels, knn_predictions).tolist()
            },
            "svm": {
                "accuracy": accuracy_score(test_labels, svm_predictions),
                "precision": precision_score(test_labels, svm_predictions),
                "recall": recall_score(test_labels, svm_predictions),
                "f1": f1_score(test_labels, svm_predictions),
                "confusion_matrix": confusion_matrix(test_labels, svm_predictions).tolist()
            },
            "timing": {
                "embedding_generation_train": embedding_time,
                "embedding_generation_test": test_embedding_time,
                "training": training_time,
                "inference": inference_time
            }
        }
        
        # Save metrics to JSON file
        metrics_file = os.path.join(metrics_dir, 'performance_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print classification report
        print("\nClassification Report (KNN):")
        print(classification_report(test_labels, knn_predictions, target_names=['Non-ICS', 'ICS']))
        
        print("\nClassification Report (SVM):")
        print(classification_report(test_labels, svm_predictions, target_names=['Non-ICS', 'ICS']))
        
        # Create confusion matrix plots
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(test_labels, knn_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-ICS', 'ICS'], yticklabels=['Non-ICS', 'ICS'])
        plt.title('KNN Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        cm = confusion_matrix(test_labels, svm_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-ICS', 'ICS'], yticklabels=['Non-ICS', 'ICS'])
        plt.title('SVM Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'confusion_matrices.png'))
        
        # If clustering is requested, cluster both training and test signatures
        if args.cluster:
            # Cluster training signatures
            train_clusters, train_cluster_labels = cluster_signatures(
                train_signatures, 
                train_app_names,
                algorithm=args.cluster_algorithm,
                eps=args.eps,
                min_samples=args.min_samples,
                n_clusters=args.n_clusters
            )
            
            # Save training cluster labels
            train_cluster_labels_file = os.path.join(train_output_dir, 'cluster_labels.npy')
            np.save(train_cluster_labels_file, train_cluster_labels)
            
            # Save training clusters as JSON
            train_clusters_file = os.path.join(train_output_dir, 'clusters.json')
            with open(train_clusters_file, 'w') as f:
                json.dump({str(k): v for k, v in train_clusters.items()}, f, indent=2)
            
            # Cluster test signatures
            test_clusters, test_cluster_labels = cluster_signatures(
                test_signatures, 
                test_app_names,
                algorithm=args.cluster_algorithm,
                eps=args.eps,
                min_samples=args.min_samples,
                n_clusters=args.n_clusters
            )
            
            # Save test cluster labels
            test_cluster_labels_file = os.path.join(test_output_dir, 'cluster_labels.npy')
            np.save(test_cluster_labels_file, test_cluster_labels)
            
            # Save test clusters as JSON
            test_clusters_file = os.path.join(test_output_dir, 'clusters.json')
            with open(test_clusters_file, 'w') as f:
                json.dump({str(k): v for k, v in test_clusters.items()}, f, indent=2)
    
    print("Train/test dataset processing complete!")

def process_single_directory(args):
    """Process apps from a single directory (original functionality)"""
    # Get app directories
    app_dirs = get_app_dirs(args.apps_dir)
    print(f"Found {len(app_dirs)} app directories.")
    
    # Process apps to extract necessary data
    app_data = process_apps(app_dirs, args.apps_dir, args.include_description)
    
    # Get app names
    app_names = [app['app_name'] for app in app_data]
    
    # Initialize GraphCodeBERT embedder
    embedder = GraphCodeBertEmbedder()
    
    # Get embeddings for all apps
    print("Generating GraphCodeBERT embeddings...")
    combined_embeddings = embedder.embed_app_batch(app_data, args.include_description)
    print(f"Generated embeddings of shape: {combined_embeddings.shape}")
    
    # Get labels for contrastive learning
    is_ics_labels = get_labels(app_names, args)
    
    # Save raw embeddings
    embeddings_file = os.path.join(args.output_dir, 'graphcodebert_embeddings.npy')
    np.save(embeddings_file, combined_embeddings)
    
    # Save app names
    names_file = os.path.join(args.output_dir, 'app_names.json')
    with open(names_file, 'w') as f:
        json.dump(app_names, f)
    
    # Train contrastive model if requested
    if args.train_contrastive:
        print("Training contrastive model...")
        trainer = SignatureTrainer(args.output_dir)
        model, signatures = trainer.train_model(
            combined_embeddings, 
            is_ics_labels,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
        
        # Save signatures
        trainer.save_signatures(signatures, app_names, is_ics_labels)
        
        # Use signatures for clustering
        if args.cluster:
            clusters, labels = cluster_signatures(
                signatures, 
                app_names,
                algorithm=args.cluster_algorithm,
                eps=args.eps,
                min_samples=args.min_samples,
                n_clusters=args.n_clusters
            )
            
            # Save cluster labels
            labels_file = os.path.join(args.output_dir, 'cluster_labels.npy')
            np.save(labels_file, labels)
            
            # Save clusters as JSON
            clusters_file = os.path.join(args.output_dir, 'clusters.json')
            with open(clusters_file, 'w') as f:
                json.dump({str(k): v for k, v in clusters.items()}, f, indent=2)
    
    elif args.cluster:
        # Cluster raw embeddings
        clusters, labels = cluster_signatures(
            combined_embeddings, 
            app_names,
            algorithm=args.cluster_algorithm,
            eps=args.eps,
            min_samples=args.min_samples,
            n_clusters=args.n_clusters
        )
        
        # Save cluster labels
        labels_file = os.path.join(args.output_dir, 'cluster_labels.npy')
        np.save(labels_file, labels)
        
        # Save clusters as JSON
        clusters_file = os.path.join(args.output_dir, 'clusters.json')
        with open(clusters_file, 'w') as f:
            json.dump({str(k): v for k, v in clusters.items()}, f, indent=2)
    
    print("Signature extraction complete!")

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process based on dataset mode
    if args.dataset_mode == 'train_test':
        # Check if required directories are provided
        if not all([args.train_ics_dir, args.train_non_ics_dir, args.test_ics_dir, args.test_non_ics_dir]):
            print("Error: For train_test mode, all train and test directories must be provided.")
            return
        
        process_train_test_dataset(args)
    else:
        # Check if apps_dir is provided
        if not args.apps_dir:
            print("Error: For single_dir mode, apps_dir must be provided.")
            return
            
        process_single_directory(args)

if __name__ == "__main__":
    main() 