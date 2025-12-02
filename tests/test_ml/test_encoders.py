"""
Unit Tests for ML Encoder Modules
==================================

Tests for modality-specific encoders.
"""

import pytest
import torch
import numpy as np

from pipeline.ml.encoders.cmb_encoder import CMBEncoder
from pipeline.ml.encoders.bao_encoder import BAOEncoder
from pipeline.ml.encoders.void_encoder import VoidEncoder
from pipeline.ml.encoders.galaxy_encoder import GalaxyEncoder
from pipeline.ml.encoders.frb_encoder import FRBEncoder
from pipeline.ml.encoders.lyman_alpha_encoder import LymanAlphaEncoder
from pipeline.ml.encoders.jwst_encoder import JWSTEncoder


class TestCMBEncoder:
    """Test CMB encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = CMBEncoder(input_dim=500, latent_dim=128)
        assert encoder.input_dim == 500
        assert encoder.latent_dim == 128

    def test_forward(self):
        """Test forward pass."""
        encoder = CMBEncoder(input_dim=500, latent_dim=128)
        
        # Input: (batch, seq_len)
        x = torch.randn(4, 500)
        output = encoder(x)
        
        assert output.shape == (4, 128)
        assert not torch.isnan(output).any()


class TestBAOEncoder:
    """Test BAO encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = BAOEncoder(input_dim=10, latent_dim=64)
        assert encoder.encoder is not None
        # Test forward to verify it works
        x = torch.randn(4, 10)
        output = encoder(x)
        assert output.shape == (4, 64)

    def test_forward(self):
        """Test forward pass."""
        encoder = BAOEncoder(input_dim=10, latent_dim=64)
        
        x = torch.randn(8, 10)
        output = encoder(x)
        
        assert output.shape == (8, 64)
        assert not torch.isnan(output).any()


class TestVoidEncoder:
    """Test Void encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = VoidEncoder(input_dim=20, latent_dim=64)
        assert encoder.encoder is not None
        # Test forward to verify it works
        x = torch.randn(4, 20)
        output = encoder(x)
        assert output.shape == (4, 64)

    def test_forward(self):
        """Test forward pass."""
        encoder = VoidEncoder(input_dim=20, latent_dim=64)
        
        x = torch.randn(16, 20)
        output = encoder(x)
        
        assert output.shape == (16, 64)
        assert not torch.isnan(output).any()


class TestGalaxyEncoder:
    """Test Galaxy encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = GalaxyEncoder(input_dim=30, latent_dim=128)
        assert encoder.input_dim == 30
        assert encoder.latent_dim == 128

    def test_forward(self):
        """Test forward pass."""
        encoder = GalaxyEncoder(input_dim=30, latent_dim=128)
        
        x = torch.randn(32, 30)
        output = encoder(x)
        
        assert output.shape == (32, 128)
        assert not torch.isnan(output).any()


class TestFRBEncoder:
    """Test FRB encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = FRBEncoder(input_dim=15, latent_dim=64)
        assert encoder.encoder is not None
        # Test forward to verify it works
        x = torch.randn(4, 15)
        output = encoder(x)
        assert output.shape == (4, 64)

    def test_forward(self):
        """Test forward pass."""
        encoder = FRBEncoder(input_dim=15, latent_dim=64)
        
        # Input: (batch, seq_len)
        x = torch.randn(8, 15)
        output = encoder(x)
        
        assert output.shape == (8, 64)
        assert not torch.isnan(output).any()


class TestLymanAlphaEncoder:
    """Test Lyman-alpha encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = LymanAlphaEncoder(input_dim=100, latent_dim=128)
        assert encoder.encoder is not None
        # Test forward to verify it works
        x = torch.randn(4, 100)
        output = encoder(x)
        assert output.shape == (4, 128)

    def test_forward(self):
        """Test forward pass."""
        encoder = LymanAlphaEncoder(input_dim=100, latent_dim=128)
        
        # Input: (batch, seq_len)
        x = torch.randn(4, 100)
        output = encoder(x)
        
        assert output.shape == (4, 128)
        assert not torch.isnan(output).any()


class TestJWSTEncoder:
    """Test JWST encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = JWSTEncoder(input_dim=25, latent_dim=128)
        assert encoder.encoder is not None
        # Test forward to verify it works
        x = torch.randn(4, 25)
        output = encoder(x)
        assert output.shape == (4, 128)

    def test_forward(self):
        """Test forward pass."""
        encoder = JWSTEncoder(input_dim=25, latent_dim=128)
        
        # Input: flattened features (batch, features)
        # JWSTEncoder expects flattened 1D input
        x = torch.randn(4, 25)
        output = encoder(x)
        
        assert output.shape == (4, 128)
        assert not torch.isnan(output).any()

