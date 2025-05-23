import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class GraphCodeBertEmbedder:
    """
    Generates embeddings for code and structured data using GraphCodeBERT.
    """
    def __init__(self, model_name="microsoft/graphcodebert-base", device=None):
        """
        Initialize the GraphCodeBERT model and tokenizer.
        
        Args:
            model_name: Pre-trained model identifier
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        
        # Set device (use GPU if available)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading GraphCodeBERT model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def get_embeddings(self, text_inputs, batch_size=8, max_length=512):
        """
        Generate embeddings for a list of text inputs.
        
        Args:
            text_inputs: List of strings to embed
            batch_size: Number of samples per batch
            max_length: Maximum token length per sample
            
        Returns:
            Numpy array of embeddings, shape (n_samples, embedding_dim)
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(text_inputs), batch_size):
            batch = text_inputs[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding as the text representation
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            # Return empty array with correct dimensionality
            return np.array([])
    
    def get_combined_app_embedding(self, smali_input, manifest_input, 
                                  include_description=False, description_input=None,
                                  combination_method='concat'):
        """
        Generate a combined embedding for an app using its smali code and manifest.
        
        Args:
            smali_input: String representation of smali code
            manifest_input: String representation of manifest data
            include_description: Whether to include description embedding
            description_input: String description (if include_description is True)
            combination_method: How to combine embeddings ('concat' or 'mean')
            
        Returns:
            Combined embedding vector as numpy array
        """
        inputs = [smali_input, manifest_input]
        
        if include_description and description_input:
            inputs.append(description_input)
        
        # Get embeddings for all inputs
        embeddings = self.get_embeddings(inputs)
        
        if combination_method == 'mean':
            # Average the embeddings
            return np.mean(embeddings, axis=0)
        else:
            # Concatenate the embeddings (default)
            return np.concatenate(embeddings)
    
    def embed_app_batch(self, app_data_batch, include_description=False):
        """
        Generate embeddings for a batch of apps.
        
        Args:
            app_data_batch: List of dictionaries with 'smali_input' and 'manifest_input'
            include_description: Whether to include description embeddings
            
        Returns:
            Numpy array of combined embeddings
        """
        smali_inputs = [app.get('smali_input', '') for app in app_data_batch]
        manifest_inputs = [app.get('manifest_input', '') for app in app_data_batch]
        
        # Get embeddings for smali code and manifest
        smali_embeddings = self.get_embeddings(smali_inputs)
        manifest_embeddings = self.get_embeddings(manifest_inputs)
        
        if include_description:
            description_inputs = [app.get('description_input', '') for app in app_data_batch]
            description_embeddings = self.get_embeddings(description_inputs)
            
            # Combine all three types of embeddings
            combined_embeddings = np.concatenate([
                smali_embeddings, 
                manifest_embeddings,
                description_embeddings
            ], axis=1)
        else:
            # Combine smali and manifest embeddings
            combined_embeddings = np.concatenate([
                smali_embeddings, 
                manifest_embeddings
            ], axis=1)
        
        return combined_embeddings 