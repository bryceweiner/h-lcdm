"""
FRB Encoder
==========

Encoder for Fast Radio Burst timing data.
"""

import torch
import torch.nn as nn


class FRBEncoder(nn.Module):
    """Encoder for FRB timing sequences."""

    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        return self.encoder(x)
