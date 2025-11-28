"""
Unit Tests for Multimodal Fusion
==================================

Tests for multimodal data fusion module.
"""

import pytest
import numpy as np
import torch

from pipeline.ml.multimodal_fusion import MultimodalFusion


class TestMultimodalFusion:
    """Test MultimodalFusion functionality."""

    def test_initialization(self):
        """Test fusion module initialization."""
        fusion = MultimodalFusion(
            latent_dim=512,
            fusion_dim=256,
            n_heads=8
        )
        
        assert fusion.fusion_dim == 256
        assert fusion.latent_dim == 512
        assert fusion.n_heads == 8

    def test_forward(self):
        """Test forward pass."""
        fusion = MultimodalFusion(
            latent_dim=128,
            fusion_dim=64
        )
        
        # Input: dict of modality embeddings (batch, latent_dim)
        modality_embeddings = {
            'cmb': torch.randn(4, 128),
            'bao': torch.randn(4, 128),
            'void': torch.randn(4, 128)
        }
        
        result = fusion(modality_embeddings)
        fused = result['fused']
        
        assert fused.shape[0] == 4
        assert fused.shape[1] == 64
        assert not torch.isnan(fused).any()

    def test_cross_attention(self):
        """Test cross-attention mechanism."""
        fusion = MultimodalFusion(
            latent_dim=128,
            fusion_dim=64,
            n_heads=4
        )
        
        modality_embeddings = {
            'cmb': torch.randn(8, 128),
            'bao': torch.randn(8, 128)
        }
        
        result = fusion(modality_embeddings)
        fused = result['fused']
        
        assert fused.shape == (8, 64)
        assert not torch.isnan(fused).any()

    def test_multiple_modalities(self):
        """Test fusion with multiple modalities."""
        fusion = MultimodalFusion(
            latent_dim=256,
            fusion_dim=128
        )
        
        # 7 modalities as dict
        modality_names = ['cmb', 'bao', 'void', 'galaxy', 'frb', 'lyman_alpha', 'jwst']
        modality_embeddings = {name: torch.randn(4, 256) for name in modality_names}
        
        result = fusion(modality_embeddings)
        fused = result['fused']
        
        assert fused.shape == (4, 128)

