"""
Multimodal Fusion
================

Fusion of representations from multiple cosmological modalities.
Implements cross-attention and late fusion for unified latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import numpy as np


class MultimodalFusion(nn.Module):
    """
    Fuse representations from multiple cosmological modalities.

    Implements cross-attention mechanism to combine information from:
    - CMB power spectra
    - BAO measurements
    - Void catalogs
    - Galaxy catalogs
    - FRB data
    - Lyman-alpha spectra
    - JWST catalogs
    """

    def __init__(self, latent_dim: int = 512,
                 fusion_dim: int = 512,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize multimodal fusion.

        Parameters:
            latent_dim: Dimension of individual modality embeddings
            fusion_dim: Dimension of fused representation
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion_dim = fusion_dim
        self.n_heads = n_heads

        # Modality embeddings (learnable)
        self.modality_embeddings = nn.Embedding(7, latent_dim)  # 7 modalities

        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Self-attention for fused representation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        )

        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(7))

        # Modality index mapping
        self.modality_to_idx = {
            'cmb': 0,
            'bao': 1,
            'void': 2,
            'galaxy': 3,
            'frb': 4,
            'lyman_alpha': 5,
            'jwst': 6
        }

    def forward(self, modality_encodings: Dict[str, torch.Tensor],
                return_attention_weights: bool = False) -> Dict[str, Any]:
        """
        Fuse multimodal representations.

        Parameters:
            modality_encodings: Encoded representations for each modality
            return_attention_weights: Whether to return attention weights

        Returns:
            dict: Fused representation and optional attention weights
        """
        # Handle missing modalities
        available_modalities = list(modality_encodings.keys())
        if len(available_modalities) == 0:
            # Get device from first encoding tensor if available
            device = next(iter(modality_encodings.values())).device if modality_encodings else torch.device('cpu')
            return {'fused': torch.zeros(1, self.fusion_dim, device=device)}

        # Stack encodings
        encodings_list = []
        modality_indices = []

        for modality, encoding in modality_encodings.items():
            encodings_list.append(encoding)
            modality_indices.append(self.modality_to_idx[modality])

        # Convert to tensor: (batch_size, n_modalities, latent_dim)
        encodings_tensor = torch.stack(encodings_list, dim=1)
        batch_size = encodings_tensor.shape[0]

        # Add modality embeddings
        modality_idx_tensor = torch.tensor(modality_indices, device=encodings_tensor.device)
        modality_embeds = self.modality_embeddings(modality_idx_tensor)  # (n_modalities, latent_dim)
        modality_embeds = modality_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_mod, latent)

        # Add positional/modality information
        enhanced_encodings = encodings_tensor + modality_embeds

        # Self-attention across modalities
        attended, self_attn_weights = self.self_attention(
            enhanced_encodings, enhanced_encodings, enhanced_encodings,
            need_weights=return_attention_weights
        )

        # Weighted fusion based on modality importance
        weights = F.softmax(self.modality_weights[modality_indices], dim=0)
        weights = weights.unsqueeze(0).unsqueeze(-1)  # (1, n_mod, 1)

        # Weighted sum
        fused = torch.sum(attended * weights, dim=1)  # (batch, latent_dim)

        # Final fusion network
        fused = self.fusion_net(fused)

        result = {'fused': fused}

        if return_attention_weights:
            result['self_attention_weights'] = self_attn_weights
            result['modality_weights'] = weights.squeeze()

        return result

    def get_modality_importance(self) -> Dict[str, float]:
        """
        Get learned modality importance weights.

        Returns:
            dict: Modality importance scores
        """
        weights = F.softmax(self.modality_weights, dim=0)

        importance = {}
        for modality, idx in self.modality_to_idx.items():
            importance[modality] = weights[idx].item()

        return importance

    def cross_modal_attention(self, query_modality: str, key_modalities: List[str],
                             encodings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute cross-modal attention between specific modalities.

        Parameters:
            query_modality: Modality to attend to
            key_modalities: Modalities to attend from
            encodings: Encoded representations

        Returns:
            dict: Cross-attention results
        """
        if query_modality not in encodings or not any(k in encodings for k in key_modalities):
            return {}

        query = encodings[query_modality].unsqueeze(1)  # (batch, 1, latent)
        keys = []
        values = []

        for mod in key_modalities:
            if mod in encodings:
                keys.append(encodings[mod].unsqueeze(1))
                values.append(encodings[mod].unsqueeze(1))

        if not keys:
            return {}

        key_tensor = torch.cat(keys, dim=1)  # (batch, n_keys, latent)
        value_tensor = torch.cat(values, dim=1)  # (batch, n_keys, latent)

        # Cross-attention
        attended, attn_weights = self.cross_attention(
            query, key_tensor, value_tensor,
            need_weights=True
        )

        return {
            'attended': attended.squeeze(1),  # (batch, latent)
            'attention_weights': attn_weights,  # (batch, 1, n_keys)
            'key_modalities': key_modalities
        }


class LateFusion(nn.Module):
    """
    Simple late fusion by concatenation and projection.
    Alternative to attention-based fusion for comparison.
    """

    def __init__(self, latent_dim: int, fusion_dim: int, n_modalities: int = 7):
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion_dim = fusion_dim
        self.n_modalities = n_modalities

        # Projection network for concatenated features
        self.fusion_net = nn.Sequential(
            nn.Linear(latent_dim * n_modalities, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        )

    def forward(self, modality_encodings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Late fusion by concatenation.

        Parameters:
            modality_encodings: Encodings for each modality

        Returns:
            torch.Tensor: Fused representation
        """
        # Handle missing modalities by padding with zeros
        encodings_list = []
        for i in range(self.n_modalities):
            # Find modality by index
            modality = None
            for mod_name, idx in self.modality_to_idx.items():
                if idx == i:
                    modality = mod_name
                    break

            if modality in modality_encodings:
                encodings_list.append(modality_encodings[modality])
            else:
                # Pad with zeros
                encodings_list.append(torch.zeros_like(list(modality_encodings.values())[0]))

        # Concatenate
        concatenated = torch.cat(encodings_list, dim=1)  # (batch, n_mod * latent)

        # Fuse
        fused = self.fusion_net(concatenated)

        return fused

    @property
    def modality_to_idx(self):
        return {
            'cmb': 0,
            'bao': 1,
            'void': 2,
            'galaxy': 3,
            'frb': 4,
            'lyman_alpha': 5,
            'jwst': 6
        }


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns optimal combination weights.
    """

    def __init__(self, latent_dim: int, fusion_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion_dim = fusion_dim

        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(7) / 7)  # Equal initial weights

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(latent_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def forward(self, modality_encodings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Adaptive fusion with learned weights.

        Parameters:
            modality_encodings: Encodings for each modality

        Returns:
            torch.Tensor: Fused representation
        """
        # Normalize weights
        weights = F.softmax(self.combination_weights, dim=0)

        # Weighted sum of available modalities
        weighted_sum = torch.zeros(list(modality_encodings.values())[0].shape,
                                 device=list(modality_encodings.values())[0].device)

        total_weight = 0
        for modality, encoding in modality_encodings.items():
            idx = self.modality_to_idx[modality]
            weight = weights[idx]
            weighted_sum += weight * encoding
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            weighted_sum = weighted_sum / total_weight

        # Final fusion
        fused = self.fusion_net(weighted_sum)

        return fused

    @property
    def modality_to_idx(self):
        return {
            'cmb': 0,
            'bao': 1,
            'void': 2,
            'galaxy': 3,
            'frb': 4,
            'lyman_alpha': 5,
            'jwst': 6
        }
