"""
CMB Encoder
==========

Convolutional encoder for CMB power spectra in harmonic space.
Handles masking and foreground subtraction effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CMBEncoder(nn.Module):
    """
    Convolutional encoder for CMB E-mode polarization power spectra.

    Designed to handle:
    - Harmonic space representations (ℓ, C_ℓ)
    - Survey masking effects
    - Foreground subtraction uncertainties
    - Cosmic variance
    """

    def __init__(self, input_dim: int, latent_dim: int = 512,
                 n_filters: int = 64, kernel_size: int = 5):
        """
        Initialize CMB encoder.

        Parameters:
            input_dim: Length of power spectrum (number of multipoles)
            latent_dim: Dimension of output latent space
            n_filters: Number of convolutional filters
            kernel_size: Size of convolutional kernels
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            # Input: (batch, 1, input_dim)
            nn.Conv1d(1, n_filters, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(n_filters, n_filters*2, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(n_filters*2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(n_filters*2, n_filters*4, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(n_filters*4),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Global average pooling
            nn.AdaptiveAvgPool1d(1)
        )

        # Flatten and project to latent space
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters*4, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Attention mechanism for multipole importance
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode CMB power spectrum.

        Parameters:
            x: Power spectrum tensor (batch, seq_len) or (batch, 1, seq_len)

        Returns:
            torch.Tensor: Encoded representation (batch, latent_dim)
        """
        # Ensure correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)

        batch_size = x.shape[0]

        # Apply attention weights to emphasize important multipoles
        attention_weights = self.attention(x.squeeze(1))  # (batch, seq_len)
        attention_weights = attention_weights.unsqueeze(1)  # (batch, 1, seq_len)

        # Weight the input
        x_weighted = x * attention_weights

        # Convolutional encoding
        conv_features = self.conv_layers(x_weighted)

        # Project to latent space
        latent = self.projection(conv_features)

        return latent

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        Parameters:
            x: Input power spectrum

        Returns:
            torch.Tensor: Attention weights
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        return self.attention(x.squeeze(1))
