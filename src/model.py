import torch
import torch.nn as nn
from typing import List


class TabularMLP(nn.Module):
    """
    Multi-Layer Perceptron optimized for Tabular Classification with Full-Batch stability.
    Uses LayerNorm to prevent batch statistics drift between training and inference.
    
    Args:
        in_features (int): Number of input features.
        hidden_dims (List[int]): Dimensions of hidden layers. Default: [128, 64, 32].
        num_classes (int): Number of output classes (2 for binary).
        dropout (float): Dropout probability. Default: 0.1.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [128, 64, 32],
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final classification head (raw logits)
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits (shape: [batch_size, num_classes]).
        """
        return self.network(x)
