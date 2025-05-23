import os
import re
import glob
import xml.etree.ElementTree as ET
import numpy as np

class SmaliExtractor:
    """
    Extracts tokens and features from smali code files.
    """
    @staticmethod
    def extract_tokens(app_path):
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
    
    @staticmethod
    def get_smali_document(app_path):
        """Get a text document representation of smali code"""
        tokens = SmaliExtractor.extract_tokens(app_path)
        if not tokens:
            return ""
        return " ".join(tokens)
    
    @staticmethod
    def get_graphcodebert_input(app_path, max_length=5000):
        """
        Generate structured input for GraphCodeBERT from smali files.
        Returns a representative sample of smali code.
        """
        smali_document = SmaliExtractor.get_smali_document(app_path)
        # Limit length to avoid token overflow
        return smali_document[:max_length]


class ManifestExtractor:
    """
    Extracts features from AndroidManifest.xml files.
    """
    @staticmethod
    def extract_features(app_path):
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
            
            # Extract services (similar to activities)
            for service in root.findall(".//service", ns):
                svc_data = {'exported': False, 'intent_filters': []}
                if 'name' in service.attrib:
                    svc_data['name'] = service.attrib['name']
                elif '{http://schemas.android.com/apk/res/android}name' in service.attrib:
                    svc_data['name'] = service.attrib['{http://schemas.android.com/apk/res/android}name']
                
                if 'exported' in service.attrib:
                    svc_data['exported'] = service.attrib['exported'] == 'true'
                elif '{http://schemas.android.com/apk/res/android}exported' in service.attrib:
                    svc_data['exported'] = service.attrib['{http://schemas.android.com/apk/res/android}exported'] == 'true'
                
                # Extract intent filters
                for intent_filter in service.findall(".//intent-filter", ns):
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
                    
                    svc_data['intent_filters'].append(intent_data)
                
                components['services'].append(svc_data)
            
            # Extract receivers (similar to activities)
            for receiver in root.findall(".//receiver", ns):
                rec_data = {'exported': False, 'intent_filters': []}
                if 'name' in receiver.attrib:
                    rec_data['name'] = receiver.attrib['name']
                elif '{http://schemas.android.com/apk/res/android}name' in receiver.attrib:
                    rec_data['name'] = receiver.attrib['{http://schemas.android.com/apk/res/android}name']
                
                if 'exported' in receiver.attrib:
                    rec_data['exported'] = receiver.attrib['exported'] == 'true'
                elif '{http://schemas.android.com/apk/res/android}exported' in receiver.attrib:
                    rec_data['exported'] = receiver.attrib['{http://schemas.android.com/apk/res/android}exported'] == 'true'
                
                # Extract intent filters
                for intent_filter in receiver.findall(".//intent-filter", ns):
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
                    
                    rec_data['intent_filters'].append(intent_data)
                
                components['receivers'].append(rec_data)
            
            # Extract content providers
            for provider in root.findall(".//provider", ns):
                prov_data = {'exported': False}
                if 'name' in provider.attrib:
                    prov_data['name'] = provider.attrib['name']
                elif '{http://schemas.android.com/apk/res/android}name' in provider.attrib:
                    prov_data['name'] = provider.attrib['{http://schemas.android.com/apk/res/android}name']
                
                if 'exported' in provider.attrib:
                    prov_data['exported'] = provider.attrib['exported'] == 'true'
                elif '{http://schemas.android.com/apk/res/android}exported' in provider.attrib:
                    prov_data['exported'] = provider.attrib['{http://schemas.android.com/apk/res/android}exported'] == 'true'
                
                components['providers'].append(prov_data)
            
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
    
    @staticmethod
    def get_graphcodebert_input(manifest_data):
        """
        Generate structured input for GraphCodeBERT from manifest data.
        Returns a string representation of manifest features.
        """
        if not manifest_data:
            return ""
            
        manifest_text = f"package {manifest_data.get('package_name', '')}\n"
        
        # Add permissions
        for perm in manifest_data.get('permissions', []):
            manifest_text += f"permission {perm}\n"
        
        # Add components
        for comp_type, comps in manifest_data.get('components', {}).items():
            for comp in comps:
                manifest_text += f"component {comp_type} {comp.get('name', '')}"
                if comp.get('exported', False):
                    manifest_text += " exported"
                manifest_text += "\n"
                
                # Add intent filters
                for intent in comp.get('intent_filters', []):
                    for action in intent.get('actions', []):
                        manifest_text += f"  action {action}\n"
                    for category in intent.get('categories', []):
                        manifest_text += f"  category {category}\n"
        
        # Add features
        for feature in manifest_data.get('features', []):
            manifest_text += f"feature {feature}\n"
            
        return manifest_text


class DescriptionExtractor:
    """
    Extracts and processes app descriptions.
    Contains a placeholder for embedding descriptions with transformer models.
    """
    @staticmethod
    def get_description(app_path):
        """Get the app description from description.txt"""
        description_path = os.path.join(app_path, "description.txt")
        if not os.path.exists(description_path):
            return ""
        
        try:
            with open(description_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading description file: {e}")
            return ""
    
    @staticmethod
    def get_embedding(description_text, vector_dim=768):
        """
        PLACEHOLDER: Get embedding vector for app description.
        In the future, this will be replaced with a real transformer model.
        """
        if not description_text:
            return np.zeros(vector_dim)
            
        # Placeholder: return a random vector for now
        # In a real implementation, this would use a pre-trained model:
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # embedding = model.encode(description_text)
        # return embedding
        
        return np.random.rand(vector_dim) 