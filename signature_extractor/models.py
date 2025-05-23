import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction of app embeddings.
    Compresses input vectors to a fixed-size signature vector.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        """Get just the encoded representation (signature)"""
        return self.encoder(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for learning app signatures.
    Pushes similar apps closer together and dissimilar apps further apart.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, x1, x2, y):
        # Euclidean distance
        dist = torch.sqrt(torch.sum((x1 - x2)**2, dim=1))
        # Contrastive loss: 
        # y=1 for same class (minimize distance)
        # y=0 for different class (push apart up to margin)
        loss = y * dist**2 + (1 - y) * torch.clamp(self.margin - dist, min=0)**2
        return torch.mean(loss) 