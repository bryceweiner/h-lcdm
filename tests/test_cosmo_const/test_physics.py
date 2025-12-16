"""
Tests for cosmological constant physics calculations.
"""

import pytest
import numpy as np
from pipeline.cosmo_const.physics import (
    calculate_geometric_entropy,
    calculate_irreversibility_fraction,
    calculate_omega_lambda,
    calculate_lambda,
)


class TestGeometricEntropy:
    """Test geometric entropy calculation."""
    
    def test_geometric_entropy_value(self):
        """Test that S_geom = (11 ln 2 - 3 ln 3)/4 = 1.0822."""
        result = calculate_geometric_entropy()
        
        S_geom = result['S_geom']
        expected = (11 * np.log(2) - 3 * np.log(3)) / 4
        
        assert np.abs(S_geom - expected) < 1e-10
        assert np.abs(S_geom - 1.0822) < 0.0001
    
    def test_dimension_weights(self):
        """Test dimension weight assignments."""
        result = calculate_geometric_entropy()
        weights = result['dimension_weights']
        
        assert weights['w_N_plus'] == 3.0
        assert weights['w_N_minus'] == 3.0
        assert weights['w_sigma'] == 2.0
        assert weights['w_tot'] == 8.0
    
    def test_probabilities(self):
        """Test probability calculations."""
        result = calculate_geometric_entropy()
        probs = result['probabilities']
        
        assert np.abs(probs['p_N_plus'] - 3/8) < 1e-10
        assert np.abs(probs['p_N_minus'] - 3/8) < 1e-10
        assert np.abs(probs['p_sigma'] - 1/4) < 1e-10
    
    def test_closed_form_agreement(self):
        """Test that numerical and closed-form calculations agree."""
        result = calculate_geometric_entropy()
        
        S_numerical = result['S_geom_numerical']
        S_closed = result['S_geom_closed_form']
        
        assert np.abs(S_numerical - S_closed) < 1e-10


class TestIrreversibilityFraction:
    """Test irreversibility fraction calculation."""
    
    def test_irreversibility_fraction_value(self):
        """Test that f_irrev = 1 - exp(-1) = 0.6321."""
        result = calculate_irreversibility_fraction()
        
        f_irrev = result['f_irrev']
        expected = 1 - np.exp(-1)
        
        assert np.abs(f_irrev - expected) < 1e-10
        assert np.abs(f_irrev - 0.6321) < 0.0001
    
    def test_formula(self):
        """Test that formula is correct."""
        result = calculate_irreversibility_fraction()
        
        assert result['formula'] == '1 - exp(-1)'
        assert result['timescale'] == 't_H = H^{-1}'


class TestOmegaLambda:
    """Test dark energy fraction calculation."""
    
    def test_omega_lambda_value(self):
        """Test that Ω_Λ = 0.6841."""
        result = calculate_omega_lambda()
        
        omega = result['omega_lambda']
        expected = 0.6841
        
        assert np.abs(omega - expected) < 0.0001
    
    def test_components(self):
        """Test that components are included."""
        result = calculate_omega_lambda()
        
        assert 'S_geom' in result
        assert 'f_irrev' in result
        assert 'components' in result
        assert 'geometric_entropy' in result['components']
        assert 'irreversibility_fraction' in result['components']
    
    def test_formula(self):
        """Test that formula is correct."""
        result = calculate_omega_lambda()
        
        assert result['formula'] == 'S_geom × f_irrev'
        
        # Verify calculation
        S_geom = result['S_geom']
        f_irrev = result['f_irrev']
        omega = result['omega_lambda']
        
        assert np.abs(omega - S_geom * f_irrev) < 1e-10


class TestLambda:
    """Test cosmological constant calculation."""
    
    def test_lambda_calculation(self):
        """Test Λ = 3Ω_Λ H²/c²."""
        result = calculate_lambda()
        
        assert 'lambda' in result
        assert 'lambda_units' in result
        assert result['lambda_units'] == 'm^-2'
        
        # Verify formula
        omega = result['omega_lambda']
        H0 = result['H0']
        c = result['c']
        Lambda = result['lambda']
        
        expected = 3 * omega * H0**2 / c**2
        assert np.abs(Lambda - expected) < 1e-20
    
    def test_lambda_with_custom_H0(self):
        """Test Lambda calculation with custom H0."""
        H0_custom = 2.0e-18  # s^-1
        result = calculate_lambda(H0_custom)
        
        assert result['H0'] == H0_custom
        
        # Verify scaling
        result_default = calculate_lambda()
        ratio = result['lambda'] / result_default['lambda']
        expected_ratio = (H0_custom / result_default['H0'])**2
        
        assert np.abs(ratio - expected_ratio) < 1e-10
    
    def test_omega_lambda_included(self):
        """Test that omega_lambda calculation is included."""
        result = calculate_lambda()
        
        assert 'omega_lambda' in result
        assert 'omega_lambda_calculation' in result


class TestNumericalPrecision:
    """Test numerical precision of calculations."""
    
    def test_high_precision(self):
        """Test that calculations maintain high precision."""
        geom = calculate_geometric_entropy()
        irrev = calculate_irreversibility_fraction()
        omega = calculate_omega_lambda()
        
        # All should be finite and well-defined
        assert np.isfinite(geom['S_geom'])
        assert np.isfinite(irrev['f_irrev'])
        assert np.isfinite(omega['omega_lambda'])
        
        # Values should be in expected ranges
        assert 1.0 < geom['S_geom'] < 1.1
        assert 0.6 < irrev['f_irrev'] < 0.7
        assert 0.68 < omega['omega_lambda'] < 0.69
