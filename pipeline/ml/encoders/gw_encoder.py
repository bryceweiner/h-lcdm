"""
GW Encoder
==========

Encoder for gravitational wave event data.
"""

import torch
import torch.nn as nn


class GWEncoder(nn.Module):
    """Encoder for gravitational wave events."""

    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim), # Input normalization
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x)
