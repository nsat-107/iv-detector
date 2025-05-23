import os
import re
import argparse
import glob
import xml.etree.ElementTree as ET
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
import torch
from pathlib import Path
import json
import pickle

# Default dimensions for each component of the signature
SMALI_VECTOR_DIM = 300
MANIFEST_VECTOR_DIM = 50
DESCRIPTION_VECTOR_DIM = 768

class SignatureExtractor:
    def __init__(self, extracted_apps_dir, output_dir, include_description=False):
        self.extracted_apps_dir = extracted_apps_dir
        self.output_dir = output_dir
        self.include_description = include_description
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize vectorizers
        self.smali_vectorizer = None
        self.manifest_vectorizer = None
        
        # For storing app data
        self.app_data = []
        
    def process_all_apps(self):
        """Process all extracted apps in the given directory"""
        print(f"Processing all apps in {self.extracted_apps_dir}...")
        
        # Get all app directories
        app_dirs = [d for d in os.listdir(self.extracted_apps_dir) 
                   if os.path.isdir(os.path.join(self.extracted_apps_dir, d))]
        
        for app_dir in app_dirs:
            app_path = os.path.join(self.extracted_apps_dir, app_dir)
            try:
                print(f"Processing app: {app_dir}")
                app_data = self.process_app(app_path)
                if app_data:
                    self.app_data.append(app_data)
            except Exception as e:
                print(f"Error processing {app_dir}: {e}")
                
        print(f"Processed {len(self.app_data)} apps successfully.")
        return self.app_data

    def process_app(self, app_path):
        """Process a single app and extract features"""
        app_name = os.path.basename(app_path)
        
        # Step 1: Extract smali tokens
        smali_tokens = self.extract_smali_tokens(app_path)
        if not smali_tokens:
            print(f"Warning: No smali tokens found for {app_name}")
            smali_document = ""
        else:
            smali_document = " ".join(smali_tokens)
        
        # Step 2: Extract manifest features
        manifest_features = self.extract_manifest_features(app_path)
        
        # Step 3: Get description embedding (optional)
        description_vector = None
        if self.include_description:
            description_vector = self.get_description_embedding(app_path)
        
        return {
            "app_name": app_name,
            "smali_document": smali_document,
            "manifest_features": manifest_features,
            "description_vector": description_vector
        }
    
    def extract_smali_tokens(self, app_path):
        """Extract tokens from all smali files in the app"""
        tokens = []
        smali_files = []
        
        # Find all smali directories
        for root, dirs, files in os.walk(app_path):
            if os.path.basename(root).startswith('smali'):
                # Find all .smali files
                smali_files.extend(glob.glob(os.path.join(root, "**/*.smali"), recursive=True))
        
        # Process each smali file
        for smali_file in smali_files:
            try:
                with open(smali_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Extract opcodes
                    opcodes = re.findall(r'^\s+([\w/-]+)', content, re.MULTILINE)
                    tokens.extend(opcodes)
                    
                    # Extract API calls
                    api_calls = re.findall(r'(L[^;]+;->[\w<>]+)', content)
                    tokens.extend(api_calls)
                    
                    # Extract class signatures
                    class_signatures = re.findall(r'^\.class.* (L[^;]+;)', content, re.MULTILINE)
                    tokens.extend(class_signatures)
                    
                    # Extract method signatures
                    method_signatures = re.findall(r'^\.method.* ([\w<>]+)\(', content, re.MULTILINE)
                    tokens.extend(method_signatures)
            except Exception as e:
                print(f"Error processing smali file {smali_file}: {e}")
                
        return tokens
    
    def extract_manifest_features(self, app_path):
        """Extract features from AndroidManifest.xml"""
        manifest_path = os.path.join(app_path, "AndroidManifest.xml")
        
        if not os.path.exists(manifest_path):
            print(f"Warning: AndroidManifest.xml not found in {app_path}")
            return {}
        
        try:
            # Parse the manifest file
            tree = ET.parse(manifest_path)
            root = tree.getroot()
            
            # Extract namespace
            ns = {'android': 'http://schemas.android.com/apk/res/android'}
            
            # Extract permissions
            permissions = []
            for perm in root.findall(".//uses-permission", ns):
                if 'name' in perm.attrib:
                    permissions.append(perm.attrib['name'])
                elif '{http://schemas.android.com/apk/res/android}name' in perm.attrib:
                    permissions.append(perm.attrib['{http://schemas.android.com/apk/res/android}name'])
            
            # Extract components
            components = {
                'activities': [],
                'services': [],
                'receivers': [],
                'providers': []
            }
            
            # Find the package name
            package_name = root.get('package', '')
            
            # Extract activities
            for activity in root.findall(".//activity", ns):
                act_data = {'exported': False, 'intent_filters': []}
                if 'name' in activity.attrib:
                    act_data['name'] = activity.attrib['name']
                elif '{http://schemas.android.com/apk/res/android}name' in activity.attrib:
                    act_data['name'] = activity.attrib['{http://schemas.android.com/apk/res/android}name']
                
                if 'exported' in activity.attrib:
                    act_data['exported'] = activity.attrib['exported'] == 'true'
                elif '{http://schemas.android.com/apk/res/android}exported' in activity.attrib:
                    act_data['exported'] = activity.attrib['{http://schemas.android.com/apk/res/android}exported'] == 'true'
                
                # Extract intent filters
                for intent_filter in activity.findall(".//intent-filter", ns):
                    intent_data = {'actions': [], 'categories': []}
                    for action in intent_filter.findall(".//action", ns):
                        if 'name' in action.attrib:
                            intent_data['actions'].append(action.attrib['name'])
                        elif '{http://schemas.android.com/apk/res/android}name' in action.attrib:
                            intent_data['actions'].append(action.attrib['{http://schemas.android.com/apk/res/android}name'])
                    
                    for category in intent_filter.findall(".//category", ns):
                        if 'name' in category.attrib:
                            intent_data['categories'].append(category.attrib['name'])
                        elif '{http://schemas.android.com/apk/res/android}name' in category.attrib:
                            intent_data['categories'].append(category.attrib['{http://schemas.android.com/apk/res/android}name'])
                    
                    act_data['intent_filters'].append(intent_data)
                
                components['activities'].append(act_data)
            
            # Similar extraction for services, receivers, providers...
            # (Simplified for brevity)
            
            # Extract features
            features = []
            for feature in root.findall(".//uses-feature", ns):
                if 'name' in feature.attrib:
                    features.append(feature.attrib['name'])
                elif '{http://schemas.android.com/apk/res/android}name' in feature.attrib:
                    features.append(feature.attrib['{http://schemas.android.com/apk/res/android}name'])
            
            return {
                'package_name': package_name,
                'permissions': permissions,
                'components': components,
                'features': features
            }
        
        except Exception as e:
            print(f"Error processing manifest file: {e}")
            return {}
    
    def get_description_embedding(self, app_path):
        """Get embedding for app description (placeholder for now)"""
        if not self.include_description:
            return None
            
        description_path = os.path.join(app_path, "description.txt")
        if not os.path.exists(description_path):
            print(f"Warning: description.txt not found in {app_path}")
            return np.zeros(DESCRIPTION_VECTOR_DIM)
        
        # Placeholder: In a real implementation, you would load a pre-trained
        # sentence-BERT model and embed the description
        try:
            with open(description_path, 'r', encoding='utf-8') as f:
                description = f.read().strip()
                
            # Placeholder: return a random vector of the correct dimension
            # In a real implementation, replace with:
            # model = SentenceTransformer('all-MiniLM-L6-v2')
            # embedding = model.encode(description)
            return np.random.rand(DESCRIPTION_VECTOR_DIM)
            
        except Exception as e:
            print(f"Error processing description: {e}")
            return np.zeros(DESCRIPTION_VECTOR_DIM)
    
    def create_vectorizers(self):
        """Create and fit vectorizers for smali code and manifest features"""
        print("Creating vectorizers...")
        
        # Create TF-IDF vectorizer for smali documents
        self.smali_vectorizer = TfidfVectorizer(
            max_features=SMALI_VECTOR_DIM,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Fit the vectorizer on all smali documents
        smali_docs = [app['smali_document'] for app in self.app_data]
        self.smali_vectorizer.fit(smali_docs)
        
        # Create a vectorizer for manifest features
        # For simplicity, we'll use TF-IDF on permissions and features
        permissions = []
        features = []
        
        for app in self.app_data:
            manifest = app['manifest_features']
            if manifest:
                permissions.extend(manifest.get('permissions', []))
                features.extend(manifest.get('features', []))
        
        # Get unique values
        unique_permissions = list(set(permissions))
        unique_features = list(set(features))
        
        # Create a mapping from feature to index
        self.manifest_feature_mapping = {
            'permission': {perm: i for i, perm in enumerate(unique_permissions)},
            'feature': {feat: i + len(unique_permissions) for i, feat in enumerate(unique_features)}
        }
        
        # Total manifest vector size
        self.manifest_vector_size = len(unique_permissions) + len(unique_features)
        print(f"Manifest vector size: {self.manifest_vector_size}")
    
    def generate_vectors(self):
        """Generate vectors for all apps"""
        print("Generating feature vectors...")
        
        if not self.smali_vectorizer:
            self.create_vectorizers()
        
        vectors = []
        app_names = []
        
        for app in self.app_data:
            app_name = app['app_name']
            app_names.append(app_name)
            
            # Generate smali vector
            smali_doc = app['smali_document']
            if not smali_doc:
                smali_vector = np.zeros(SMALI_VECTOR_DIM)
            else:
                smali_vector = self.smali_vectorizer.transform([smali_doc]).toarray()[0]
                # Ensure dimension is correct
                if len(smali_vector) < SMALI_VECTOR_DIM:
                    padding = np.zeros(SMALI_VECTOR_DIM - len(smali_vector))
                    smali_vector = np.concatenate([smali_vector, padding])
                elif len(smali_vector) > SMALI_VECTOR_DIM:
                    smali_vector = smali_vector[:SMALI_VECTOR_DIM]
            
            # Generate manifest vector
            manifest = app['manifest_features']
            manifest_vector = np.zeros(MANIFEST_VECTOR_DIM)
            
            if manifest:
                # Create a one-hot encoded vector for permissions and features
                manifest_raw_vector = np.zeros(self.manifest_vector_size)
                
                # Set values for permissions
                for perm in manifest.get('permissions', []):
                    if perm in self.manifest_feature_mapping['permission']:
                        idx = self.manifest_feature_mapping['permission'][perm]
                        manifest_raw_vector[idx] = 1
                
                # Set values for features
                for feat in manifest.get('features', []):
                    if feat in self.manifest_feature_mapping['feature']:
                        idx = self.manifest_feature_mapping['feature'][feat]
                        manifest_raw_vector[idx] = 1
                
                # Ensure dimension is correct for manifest vector
                if len(manifest_raw_vector) < MANIFEST_VECTOR_DIM:
                    padding = np.zeros(MANIFEST_VECTOR_DIM - len(manifest_raw_vector))
                    manifest_vector = np.concatenate([manifest_raw_vector, padding])
                elif len(manifest_raw_vector) > MANIFEST_VECTOR_DIM:
                    manifest_vector = manifest_raw_vector[:MANIFEST_VECTOR_DIM]
                else:
                    manifest_vector = manifest_raw_vector
            
            # Get description vector if included
            if self.include_description:
                description_vector = app['description_vector']
                
                # Concatenate all vectors
                app_vector = np.concatenate([smali_vector, manifest_vector, description_vector])
            else:
                # Concatenate without description
                app_vector = np.concatenate([smali_vector, manifest_vector])
            
            vectors.append(app_vector)
        
        # Convert to numpy array
        vectors = np.array(vectors)
        print(f"Generated vectors of shape: {vectors.shape}")
        
        return vectors, app_names
    
    def cluster_apps(self, vectors, app_names, algorithm='dbscan', eps=0.5, min_samples=5, n_clusters=5):
        """Cluster the apps based on their vectors"""
        print(f"Clustering apps using {algorithm}...")
        
        if algorithm.lower() == 'dbscan':
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vectors)
            labels = clustering.labels_
        elif algorithm.lower() == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
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
    
    def save_results(self, vectors, app_names, labels):
        """Save the results to files"""
        # Create a directory for results
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save app vectors
        vectors_file = os.path.join(results_dir, 'app_vectors.npy')
        np.save(vectors_file, vectors)
        
        # Save app names
        app_names_file = os.path.join(results_dir, 'app_names.json')
        with open(app_names_file, 'w') as f:
            json.dump(app_names, f)
        
        # Save cluster labels
        labels_file = os.path.join(results_dir, 'cluster_labels.npy')
        np.save(labels_file, labels)
        
        # Save vectorizers for future use
        vectorizers_file = os.path.join(results_dir, 'vectorizers.pkl')
        with open(vectorizers_file, 'wb') as f:
            pickle.dump({
                'smali_vectorizer': self.smali_vectorizer,
                'manifest_feature_mapping': self.manifest_feature_mapping,
                'manifest_vector_size': self.manifest_vector_size
            }, f)
        
        print(f"Results saved to {results_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract signatures from APKs')
    parser.add_argument('--apps_dir', type=str, required=True, 
                      help='Directory containing extracted APK directories')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to store output files')
    parser.add_argument('--include_description', action='store_true',
                      help='Include app description in the signature (placeholder)')
    parser.add_argument('--cluster_algorithm', type=str, default='dbscan',
                      choices=['dbscan', 'kmeans'],
                      help='Clustering algorithm to use')
    parser.add_argument('--eps', type=float, default=0.5,
                      help='Epsilon parameter for DBSCAN')
    parser.add_argument('--min_samples', type=int, default=5,
                      help='Min samples parameter for DBSCAN')
    parser.add_argument('--n_clusters', type=int, default=5,
                      help='Number of clusters for K-Means')
    
    args = parser.parse_args()
    
    extractor = SignatureExtractor(
        extracted_apps_dir=args.apps_dir,
        output_dir=args.output_dir,
        include_description=args.include_description
    )
    
    # Process all apps
    extractor.process_all_apps()
    
    # Generate vectors
    vectors, app_names = extractor.generate_vectors()
    
    # Cluster apps
    clusters, labels = extractor.cluster_apps(
        vectors, 
        app_names,
        algorithm=args.cluster_algorithm,
        eps=args.eps,
        min_samples=args.min_samples,
        n_clusters=args.n_clusters
    )
    
    # Save results
    extractor.save_results(vectors, app_names, labels)
    
    print("Signature extraction complete!")

if __name__ == "__main__":
    main() 