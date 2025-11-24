"""
Void Encoder
===========

Encoder for cosmic void catalogs.
"""

import torch
import torch.nn as nn


class VoidEncoder(nn.Module):
    """Encoder for void properties."""

    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
