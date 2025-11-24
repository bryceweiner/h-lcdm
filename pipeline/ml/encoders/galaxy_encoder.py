"""
Galaxy Encoder
=============

Hybrid encoder for galaxy catalogs combining photometric and morphological features.
Handles missing data and varying feature sets across surveys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class GalaxyEncoder(nn.Module):
    """
    Encoder for galaxy catalog features.

    Combines:
    - Photometric features (magnitudes, colors)
    - Morphological features (concentration, size)
    - Clustering features (environment)
    - Spectroscopic features (redshift, line strengths)
    """

    def __init__(self, input_dim: int, latent_dim: int = 512):
        """
        Initialize galaxy encoder.

        Parameters:
            input_dim: Number of input features
            latent_dim: Output latent dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Feature type embeddings
        self.feature_types = {
            'photometric': slice(0, 10),      # magnitudes, colors
            'morphological': slice(10, 15),   # concentration, sizes
            'spectroscopic': slice(15, 25),   # redshift, line ratios
            'clustering': slice(25, None)     # environment features
        }

        # Separate encoders for different feature types
        self.photometric_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )

        self.morphological_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )

        self.spectroscopic_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )

        self.clustering_encoder = nn.Sequential(
            nn.Linear(max(1, input_dim - 25), 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )

        # Cross-attention fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )

        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(128 * 4, latent_dim),  # Concatenated features
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )

        # Missing data handling
        self.feature_mask = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode galaxy features.

        Parameters:
            x: Galaxy feature tensor (batch, n_features)

        Returns:
            torch.Tensor: Encoded representation
        """
        batch_size = x.shape[0]

        # Handle missing features with masking
        feature_mask = torch.sigmoid(self.feature_mask)
        x_masked = x * feature_mask.unsqueeze(0)

        # Encode different feature types
        encodings = []

        # Photometric features
        if x.shape[1] >= 10:
            photo_features = x_masked[:, self.feature_types['photometric']]
            photo_encoded = self.photometric_encoder(photo_features)
            encodings.append(photo_encoded)

        # Morphological features
        if x.shape[1] >= 15:
            morph_features = x_masked[:, self.feature_types['morphological']]
            morph_encoded = self.morphological_encoder(morph_features)
            # Project to same dimension
            morph_encoded = F.pad(morph_encoded, (0, 128 - morph_encoded.shape[1]))
            encodings.append(morph_encoded)

        # Spectroscopic features
        if x.shape[1] >= 25:
            spec_features = x_masked[:, self.feature_types['spectroscopic']]
            spec_encoded = self.spectroscopic_encoder(spec_features)
            encodings.append(spec_encoded)

        # Clustering features
        if x.shape[1] > 25:
            clust_features = x_masked[:, self.feature_types['clustering']]
            clust_encoded = self.clustering_encoder(clust_features)
            # Project to same dimension
            clust_encoded = F.pad(clust_encoded, (0, 128 - clust_encoded.shape[1]))
            encodings.append(clust_encoded)

        # Handle case with no encodings
        if not encodings:
            return torch.zeros(batch_size, self.latent_dim, device=x.device)

        # Stack encodings for cross-attention
        encodings_tensor = torch.stack(encodings, dim=1)  # (batch, n_types, 128)

        # Self-attention across feature types
        attended, _ = self.cross_attention(
            encodings_tensor, encodings_tensor, encodings_tensor
        )

        # Flatten and project
        flattened = attended.view(batch_size, -1)
        latent = self.final_projection(flattened)

        return latent

    def get_feature_importance(self) -> torch.Tensor:
        """
        Get learned feature importance weights.

        Returns:
            torch.Tensor: Feature importance scores
        """
        return torch.sigmoid(self.feature_mask)
