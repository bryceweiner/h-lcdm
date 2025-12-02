"""
Unit Tests for Multimodal Fusion
==================================

Tests for multimodal data fusion module.
"""

import pytest
import numpy as np
import torch

from pipeline.ml.multimodal_fusion import MultimodalFusion, LateFusion, AdaptiveFusion


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

    def test_forward_with_attention_weights(self):
        """Test forward pass with attention weights."""
        fusion = MultimodalFusion(
            latent_dim=128,
            fusion_dim=64
        )
        
        modality_embeddings = {
            'cmb': torch.randn(4, 128),
            'bao': torch.randn(4, 128)
        }
        
        result = fusion(modality_embeddings, return_attention_weights=True)
        
        assert 'fused' in result
        assert 'self_attention_weights' in result
        assert 'modality_weights' in result

    def test_get_modality_importance(self):
        """Test modality importance retrieval."""
        fusion = MultimodalFusion(
            latent_dim=128,
            fusion_dim=64
        )
        
        importance = fusion.get_modality_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Check that importance sums to ~1 (softmax)
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    def test_cross_modal_attention(self):
        """Test cross-modal attention."""
        fusion = MultimodalFusion(
            latent_dim=128,
            fusion_dim=64
        )
        
        encodings = {
            'cmb': torch.randn(4, 128),
            'bao': torch.randn(4, 128),
            'void': torch.randn(4, 128)
        }
        
        result = fusion.cross_modal_attention('cmb', ['bao', 'void'], encodings)
        
        assert 'attended' in result
        assert 'attention_weights' in result
        assert 'key_modalities' in result
        assert result['attended'].shape == (4, 128)

    def test_cross_modal_attention_missing_modality(self):
        """Test cross-modal attention with missing modality."""
        fusion = MultimodalFusion(
            latent_dim=128,
            fusion_dim=64
        )
        
        encodings = {
            'cmb': torch.randn(4, 128)
        }
        
        result = fusion.cross_modal_attention('cmb', ['missing'], encodings)
        assert result == {}

    def test_cross_modal_attention_no_keys(self):
        """Test cross-modal attention with no valid keys."""
        fusion = MultimodalFusion(
            latent_dim=128,
            fusion_dim=64
        )
        
        encodings = {
            'cmb': torch.randn(4, 128)
        }
        
        # Query modality not in encodings
        result = fusion.cross_modal_attention('missing', ['bao'], encodings)
        assert result == {}
        
        # Empty key_modalities list
        result2 = fusion.cross_modal_attention('cmb', [], encodings)
        assert result2 == {}

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

    def test_empty_modalities(self):
        """Test fusion with empty modality dict."""
        fusion = MultimodalFusion(
            latent_dim=128,
            fusion_dim=64
        )
        # Set device attribute
        fusion.device = torch.device('cpu')
        
        result = fusion({})
        assert 'fused' in result
        assert result['fused'].shape[1] == 64


class TestLateFusion:
    """Test LateFusion functionality."""

    def test_initialization(self):
        """Test LateFusion initialization."""
        fusion = LateFusion(
            latent_dim=128,
            fusion_dim=64,
            n_modalities=3
        )
        
        assert fusion.latent_dim == 128
        assert fusion.fusion_dim == 64
        assert fusion.n_modalities == 3

    def test_forward(self):
        """Test LateFusion forward pass."""
        fusion = LateFusion(
            latent_dim=128,
            fusion_dim=64,
            n_modalities=3
        )
        
        modality_embeddings = {
            'cmb': torch.randn(4, 128),
            'bao': torch.randn(4, 128),
            'void': torch.randn(4, 128)
        }
        
        fused = fusion(modality_embeddings)
        
        assert fused.shape == (4, 64)
        assert not torch.isnan(fused).any()

    def test_forward_partial_modalities(self):
        """Test LateFusion with partial modalities."""
        fusion = LateFusion(
            latent_dim=128,
            fusion_dim=64,
            n_modalities=3
        )
        
        modality_embeddings = {
            'cmb': torch.randn(4, 128),
            'bao': torch.randn(4, 128)
        }
        
        fused = fusion(modality_embeddings)
        
        assert fused.shape == (4, 64)

