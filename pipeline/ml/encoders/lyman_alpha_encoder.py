"""
Lyman-alpha Encoder
==================

Encoder for Lyman-alpha forest spectra.
"""

import torch
import torch.nn as nn


class LymanAlphaEncoder(nn.Module):
    """Encoder for Lyman-alpha spectra."""

    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(1), # Normalize input channel
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        return self.encoder(x)
