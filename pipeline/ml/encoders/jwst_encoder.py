"""
JWST Encoder
===========

Encoder for JWST galaxy imaging data.
"""

import torch
import torch.nn as nn


class JWSTEncoder(nn.Module):
    """Encoder for JWST galaxy images."""

    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        # Assume input is flattened image features
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim), # Input normalization
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
