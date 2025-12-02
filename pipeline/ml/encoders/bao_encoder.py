"""
BAO Encoder
==========

Encoder for BAO distance measurements and redshift evolution.
"""

import torch
import torch.nn as nn


class BAOEncoder(nn.Module):
    """Encoder for BAO measurements."""

    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),  # Input normalization
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
