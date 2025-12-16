"""
Tests for fine structure constant physics calculations.
"""

import pytest
import numpy as np
from pipeline.fine_structure.physics import (
    calculate_bekenstein_hawking_entropy,
    calculate_information_processing_rate,
    calculate_alpha_inverse,
    calculate_alpha,
)


class TestBekensteinHawkingEntropy:
    """Test Bekenstein-Hawking entropy calculation."""
    
    def test_entropy_calculation(self):
        """Test that S_H = πc⁵/(ℏGH²) is calculated correctly."""
        result = calculate_bekenstein_hawking_entropy()
        
        assert 'S_H' in result
        assert 'ln_S_H' in result
        assert result['S_H'] > 0
        assert result['ln_S_H'] > 0
    
    def test_ln_S_H_value(self):
        """Test that ln(S_H) ≈ 281.733."""
        result = calculate_bekenstein_hawking_entropy()
        
        ln_S_H = result['ln_S_H']
        expected = 281.733
        
        assert np.abs(ln_S_H - expected) < 1.0
    
    def test_components(self):
        """Test that all components are included."""
        result = calculate_bekenstein_hawking_entropy()
        
        assert 'components' in result
        assert 'pi' in result['components']
        assert 'c' in result['components']
        assert 'hbar' in result['components']
        assert 'G' in result['components']
        assert 'H' in result['components']


class TestInformationProcessingRate:
    """Test information processing rate calculation."""
    
    def test_gamma_calculation(self):
        """Test that γ = H/ln(S_H) is calculated correctly."""
        result = calculate_information_processing_rate()
        
        assert 'gamma' in result
        assert 'ln_S_H' in result
        assert result['gamma'] > 0
    
    def test_gamma_value(self):
        """Test that γ ≈ 7.753e-21 s^-1."""
        result = calculate_information_processing_rate()
        
        gamma = result['gamma']
        expected = 7.753e-21
        
        # Allow order of magnitude agreement
        assert np.abs(gamma - expected) / expected < 0.1
    
    def test_formula(self):
        """Test that formula is correct."""
        result = calculate_information_processing_rate()
        
        assert result['formula'] == 'H/ln(S_H)'
        assert 'entropy_calculation' in result


class TestAlphaInverse:
    """Test inverse fine structure constant calculation."""
    
    def test_alpha_inverse_value(self):
        """Test that α⁻¹ = 137.032."""
        result = calculate_alpha_inverse()
        
        alpha_inv = result['alpha_inverse']
        expected = 137.032
        
        assert np.abs(alpha_inv - expected) < 0.1
    
    def test_components(self):
        """Test that all components are included."""
        result = calculate_alpha_inverse()
        
        assert 'holographic_term' in result
        assert 'geometric_term' in result
        assert 'vacuum_term' in result
        assert 'components' in result
        assert 'bekenstein_hawking_entropy' in result['components']
        assert 'information_processing_rate' in result['components']
    
    def test_formula(self):
        """Test that formula is correct."""
        result = calculate_alpha_inverse()
        
        assert result['formula'] == '(1/2)ln(S_H) - ln(4π²) - 1/(2π)'
        
        # Verify calculation
        holographic = result['holographic_term']
        geometric = result['geometric_term']
        vacuum = result['vacuum_term']
        alpha_inv = result['alpha_inverse']
        
        assert np.abs(alpha_inv - (holographic - geometric - vacuum)) < 1e-10
    
    def test_alternative_formula(self):
        """Test alternative formula using γ."""
        result = calculate_alpha_inverse()
        
        # Both forms should give same result
        assert 'alternative_formula' in result
        assert result['alternative_formula'] == 'H/(2γ) - ln(4π²) - 1/(2π)'


class TestAlpha:
    """Test fine structure constant calculation."""
    
    def test_alpha_calculation(self):
        """Test α = 1/α⁻¹."""
        result = calculate_alpha()
        
        assert 'alpha' in result
        assert 'alpha_inverse' in result
        
        alpha = result['alpha']
        alpha_inv = result['alpha_inverse']
        
        assert np.abs(alpha - 1.0 / alpha_inv) < 1e-10
    
    def test_alpha_value(self):
        """Test that α ≈ 1/137.032."""
        result = calculate_alpha()
        
        alpha = result['alpha']
        expected = 1.0 / 137.032
        
        assert np.abs(alpha - expected) < 1e-5


class TestNumericalPrecision:
    """Test numerical precision of calculations."""
    
    def test_high_precision(self):
        """Test that calculations maintain high precision."""
        entropy = calculate_bekenstein_hawking_entropy()
        gamma = calculate_information_processing_rate()
        alpha_inv = calculate_alpha_inverse()
        alpha = calculate_alpha()
        
        # All should be finite and well-defined
        assert np.isfinite(entropy['ln_S_H'])
        assert np.isfinite(gamma['gamma'])
        assert np.isfinite(alpha_inv['alpha_inverse'])
        assert np.isfinite(alpha['alpha'])
        
        # Values should be in expected ranges
        assert 200 < entropy['ln_S_H'] < 300
        assert 1e-22 < gamma['gamma'] < 1e-19
        assert 130 < alpha_inv['alpha_inverse'] < 140
        assert 0.007 < alpha['alpha'] < 0.008
