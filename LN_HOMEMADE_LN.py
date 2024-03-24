import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    """
    Implementation of Layer Normalization.
    
    Args:
        features (int): Number of features in the input tensor.
        eps (float, optional): Epsilon value to prevent division by zero. Default is 1e-6.
    """
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # Learnable scale parameter
        self.beta = nn.Parameter(torch.zeros(features))  # Learnable shift parameter
        self.eps = eps  # Epsilon value for numerical stability

    def forward(self, x):
        """
        Forward pass of the Layer Normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features).
        
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        mean = x.mean(-1, keepdim=True)  # Compute mean along the last dimension
        std = x.std(-1, keepdim=True)    # Compute standard deviation along the last dimension
        # Apply layer normalization
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
