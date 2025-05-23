import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pickle
import json
from sklearn.model_selection import train_test_split

class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning with positive and negative pairs.
    For ICS app detection, we want to pull ICS apps together and push non-ICS apps away.
    """
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        anchor_embedding = self.embeddings[idx]
        anchor_label = self.labels[idx]
        
        # Find a positive example (same class)
        positive_indices = [i for i, label in enumerate(self.labels) 
                           if label == anchor_label and i != idx]
        
        # Find a negative example (different class)
        negative_indices = [i for i, label in enumerate(self.labels) 
                           if label != anchor_label]
        
        # If no positive examples, use the anchor itself
        if not positive_indices:
            positive_idx = idx
        else:
            positive_idx = np.random.choice(positive_indices)
        
        # If no negative examples, use a random example
        if not negative_indices:
            negative_idx = np.random.choice([i for i in range(len(self.labels)) if i != idx])
        else:
            negative_idx = np.random.choice(negative_indices)
        
        positive_embedding = self.embeddings[positive_idx]
        negative_embedding = self.embeddings[negative_idx]
        
        return (
            torch.tensor(anchor_embedding, dtype=torch.float32),
            torch.tensor(positive_embedding, dtype=torch.float32),
            torch.tensor(negative_embedding, dtype=torch.float32),
            torch.tensor(anchor_label, dtype=torch.long)
        )


class ContrastiveAutoencoder(nn.Module):
    """
    Autoencoder that produces compact signatures through contrastive learning.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(ContrastiveAutoencoder, self).__init__()
        
        # Encoder: reduce dimensionality to hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        # Decoder: reconstruct original input
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x):
        # Get encoded representation (signature)
        encoded = self.encoder(x)
        # Reconstruct input
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        """Get just the encoded representation (signature)"""
        return self.encoder(x)


class TripletLoss(nn.Module):
    """
    Triplet loss for contrastive learning with positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Triplet loss: push anchor closer to positive than negative
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        return torch.mean(loss)


class SignatureTrainer:
    """
    Trains a model to generate compact signatures for apps.
    """
    def __init__(self, output_dir, device=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device (use GPU if available)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
    
    def train_model(self, embeddings, is_ics_labels, hidden_dim=128, 
                   batch_size=32, num_epochs=50, learning_rate=0.001,
                   save_model=True):
        """
        Train a contrastive autoencoder model to generate app signatures.
        
        Args:
            embeddings: Array of app embeddings from GraphCodeBERT
            is_ics_labels: Binary labels (1 for ICS apps, 0 for non-ICS)
            hidden_dim: Dimension of the signature vector
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_model: Whether to save the trained model
            
        Returns:
            Trained model and signature vectors for all apps
        """
        # Create dataset
        dataset = ContrastiveDataset(embeddings, is_ics_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create model
        input_dim = embeddings.shape[1]
        model = ContrastiveAutoencoder(input_dim, hidden_dim)
        model.to(self.device)
        
        # Loss functions
        triplet_loss_fn = TripletLoss(margin=1.0)
        recon_loss_fn = nn.MSELoss()
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        print(f"Training contrastive autoencoder on {self.device}...")
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for anchor, positive, negative, _ in dataloader:
                # Move to device
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                anchor_encoded, anchor_decoded = model(anchor)
                positive_encoded, _ = model(positive)
                negative_encoded, _ = model(negative)
                
                # Calculate losses
                triplet_loss = triplet_loss_fn(anchor_encoded, positive_encoded, negative_encoded)
                recon_loss = recon_loss_fn(anchor_decoded, anchor)
                
                # Combined loss (with weight on triplet loss)
                loss = recon_loss + 2.0 * triplet_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Generate signatures for all apps
        print("Generating final signatures...")
        model.eval()
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            signatures = model.encode(embeddings_tensor).cpu().numpy()
        
        # Save model if requested
        if save_model:
            model_path = os.path.join(self.output_dir, 'signature_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'hidden_dim': hidden_dim
            }, model_path)
            print(f"Model saved to {model_path}")
        
        return model, signatures
    
    def save_signatures(self, signatures, app_names, is_ics_labels=None):
        """
        Save generated signatures and metadata to files.
        
        Args:
            signatures: Array of signature vectors
            app_names: List of app names corresponding to signatures
            is_ics_labels: Binary labels (1 for ICS apps, 0 for non-ICS)
        """
        # Save signatures as numpy array
        signatures_file = os.path.join(self.output_dir, 'app_signatures.npy')
        np.save(signatures_file, signatures)
        
        # Save app names
        names_file = os.path.join(self.output_dir, 'app_names.json')
        with open(names_file, 'w') as f:
            json.dump(app_names, f)
        
        # Save labels if provided
        if is_ics_labels is not None:
            labels_file = os.path.join(self.output_dir, 'is_ics_labels.npy')
            np.save(labels_file, is_ics_labels)
        
        print(f"Signatures saved to {signatures_file}")
    
    @staticmethod
    def load_model(model_path, device=None):
        """
        Load a pre-trained signature model.
        
        Args:
            model_path: Path to the saved model
            device: Device to load model on
            
        Returns:
            Loaded model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Load model info
        checkpoint = torch.load(model_path, map_location=device)
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint['hidden_dim']
        
        # Create model and load state
        model = ContrastiveAutoencoder(input_dim, hidden_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model 