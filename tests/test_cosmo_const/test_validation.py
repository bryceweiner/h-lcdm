"""
Tests for cosmological constant validation infrastructure.
"""

import pytest
import numpy as np
from pipeline.cosmo_const.error_propagation import ErrorPropagation
from pipeline.cosmo_const.monte_carlo import CosmoConstMonteCarloValidator
from pipeline.cosmo_const.sensitivity import SensitivityAnalysis


class TestErrorPropagation:
    """Test error propagation analysis."""
    
    def test_omega_lambda_uncertainty(self):
        """Test that theoretical uncertainty is zero."""
        error_prop = ErrorPropagation()
        result = error_prop.propagate_omega_lambda_uncertainty()
        
        assert result['delta_omega_lambda_theory'] == 0.0
        assert result['note'] == 'Parameter-free prediction - no theoretical uncertainty'
    
    def test_lambda_uncertainty(self):
        """Test Lambda uncertainty propagation."""
        error_prop = ErrorPropagation()
        result = error_prop.propagate_lambda_uncertainty()
        
        assert 'delta_lambda' in result
        assert 'relative_uncertainty' in result
        assert result['delta_lambda'] > 0  # Should have uncertainty from H0
    
    def test_comparison_with_observation(self):
        """Test comparison with Planck observation."""
        error_prop = ErrorPropagation()
        result = error_prop.compare_with_observation()
        
        assert 'predicted' in result
        assert 'observed' in result
        assert 'deviation_sigma' in result
        assert 'consistent_2sigma' in result
        
        # Should be consistent within 2σ
        assert result['consistent_2sigma'] == True
        assert abs(result['deviation_sigma']) < 2.0
    
    def test_full_error_analysis(self):
        """Test complete error analysis."""
        error_prop = ErrorPropagation()
        result = error_prop.full_error_analysis()
        
        assert 'omega_lambda_uncertainty' in result
        assert 'lambda_uncertainty' in result
        assert 'comparison_with_observation' in result
        assert 'summary' in result


class TestMonteCarloValidation:
    """Test Monte Carlo uncertainty quantification."""
    
    def test_mc_validation(self):
        """Test Monte Carlo validation."""
        validator = CosmoConstMonteCarloValidator(n_samples=10000, random_state=42)
        result = validator.run_mc_validation()
        
        assert 'prediction' in result
        assert 'consistency_fractions' in result
        assert 'deviation_sigma' in result
        assert 'interpretation' in result
        
        # Consistency fractions should be reasonable
        consistency = result['consistency_fractions']
        assert 0 <= consistency['p_consistent_2sigma'] <= 1
    
    def test_consistency_fractions(self):
        """Test that consistency fractions are calculated correctly."""
        validator = CosmoConstMonteCarloValidator(n_samples=100000, random_state=42)
        result = validator.run_mc_validation()
        
        consistency = result['consistency_fractions']
        
        # Should have high consistency within 2σ
        assert consistency['p_consistent_2sigma'] > 0.9
    
    def test_deviation_statistics(self):
        """Test deviation statistics."""
        validator = CosmoConstMonteCarloValidator(n_samples=10000, random_state=42)
        result = validator.run_mc_validation()
        
        assert 'deviation_statistics' in result
        stats = result['deviation_statistics']
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert 'percentiles' in stats
    
    def test_sensitivity_to_uncertainty(self):
        """Test sensitivity to Planck uncertainty."""
        validator = CosmoConstMonteCarloValidator(n_samples=10000, random_state=42)
        result = validator.sensitivity_to_planck_uncertainty()
        
        assert 'sigma_values' in result
        assert 'consistency_fractions' in result
        assert 'sensitivity' in result


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""
    
    def test_dimension_weight_sensitivity(self):
        """Test sensitivity to dimension weights."""
        sensitivity = SensitivityAnalysis()
        result = sensitivity.sensitivity_to_dimension_weights(n_points=3)
        
        assert 'base_omega_lambda' in result
        assert 'parameter_variations' in result
        assert 'sensitivity_metrics' in result
        
        # Base should be around 0.6841
        assert abs(result['base_omega_lambda'] - 0.6841) < 0.01
    
    def test_timescale_sensitivity(self):
        """Test sensitivity to decoherence timescale."""
        sensitivity = SensitivityAnalysis()
        result = sensitivity.sensitivity_to_decoherence_timescale()
        
        assert 'base_omega_lambda' in result
        assert 'variations' in result
        assert 'sensitivity_metrics' in result
        
        # Base should be around 0.6841
        assert abs(result['base_omega_lambda'] - 0.6841) < 0.01
    
    def test_causal_structure_sensitivity(self):
        """Test sensitivity to causal structure."""
        sensitivity = SensitivityAnalysis()
        result = sensitivity.sensitivity_to_causal_structure()
        
        assert 'base_structure' in result
        assert 'base_omega_lambda' in result
        assert 'alternatives' in result
        
        # Base should be around 0.6841
        assert abs(result['base_omega_lambda'] - 0.6841) < 0.01
    
    def test_full_sensitivity_analysis(self):
        """Test complete sensitivity analysis."""
        sensitivity = SensitivityAnalysis()
        result = sensitivity.full_sensitivity_analysis()
        
        assert 'dimension_weights' in result
        assert 'decoherence_timescale' in result
        assert 'causal_structure' in result
        assert 'overall_robustness' in result


class TestStatisticalProperties:
    """Test statistical properties of validations."""
    
    def test_monte_carlo_reproducibility(self):
        """Test that Monte Carlo is reproducible with same seed."""
        validator1 = CosmoConstMonteCarloValidator(n_samples=1000, random_state=42)
        validator2 = CosmoConstMonteCarloValidator(n_samples=1000, random_state=42)
        
        result1 = validator1.run_mc_validation()
        result2 = validator2.run_mc_validation()
        
        # Should be identical with same seed
        assert result1['prediction'] == result2['prediction']
        assert abs(result1['deviation_sigma'] - result2['deviation_sigma']) < 1e-10
    
    def test_error_propagation_consistency(self):
        """Test that error propagation is consistent."""
        error_prop = ErrorPropagation()
        
        result1 = error_prop.propagate_omega_lambda_uncertainty()
        result2 = error_prop.propagate_omega_lambda_uncertainty()
        
        # Should be identical (deterministic)
        assert result1 == result2
