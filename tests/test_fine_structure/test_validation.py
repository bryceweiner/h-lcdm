"""
Tests for fine structure constant validation infrastructure.
"""

import pytest
import numpy as np
from pipeline.fine_structure.error_propagation import FineStructureErrorPropagation
from pipeline.fine_structure.monte_carlo import FineStructureMonteCarloValidator
from pipeline.fine_structure.sensitivity import FineStructureSensitivityAnalysis


class TestErrorPropagation:
    """Test error propagation analysis."""
    
    def test_alpha_inverse_uncertainty(self):
        """Test that uncertainty is propagated from H0."""
        error_prop = FineStructureErrorPropagation()
        result = error_prop.propagate_alpha_inverse_uncertainty()
        
        assert 'delta_alpha_inverse' in result
        assert 'relative_uncertainty' in result
        assert result['delta_alpha_inverse'] > 0  # Should have uncertainty from H0
    
    def test_comparison_with_observation(self):
        """Test comparison with CODATA observation."""
        error_prop = FineStructureErrorPropagation()
        result = error_prop.compare_with_observation()
        
        assert 'predicted' in result
        assert 'observed' in result
        assert 'deviation_sigma' in result
        assert 'consistent_2sigma' in result
        assert 'relative_difference_percent' in result
        
        # Should be consistent within 2σ
        assert result['consistent_2sigma'] == True
        assert abs(result['deviation_sigma']) < 2.0
    
    def test_full_error_analysis(self):
        """Test complete error analysis."""
        error_prop = FineStructureErrorPropagation()
        result = error_prop.full_error_analysis()
        
        assert 'alpha_inverse_uncertainty' in result
        assert 'comparison_with_observation' in result
        assert 'summary' in result


class TestMonteCarloValidation:
    """Test Monte Carlo uncertainty quantification."""
    
    def test_mc_validation(self):
        """Test Monte Carlo validation."""
        validator = FineStructureMonteCarloValidator(n_samples=10000, random_state=42)
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
        validator = FineStructureMonteCarloValidator(n_samples=100000, random_state=42)
        result = validator.run_mc_validation()
        
        consistency = result['consistency_fractions']
        
        # Should have high consistency within 2σ
        assert consistency['p_consistent_2sigma'] > 0.9
    
    def test_deviation_statistics(self):
        """Test deviation statistics."""
        validator = FineStructureMonteCarloValidator(n_samples=10000, random_state=42)
        result = validator.run_mc_validation()
        
        assert 'deviation_statistics' in result
        stats = result['deviation_statistics']
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert 'percentiles' in stats
    
    def test_sensitivity_to_h0(self):
        """Test sensitivity to H0 uncertainty."""
        validator = FineStructureMonteCarloValidator(n_samples=10000, random_state=42)
        result = validator.sensitivity_to_h0_uncertainty()
        
        assert 'H0_values' in result
        assert 'alpha_inverse_values' in result
        assert 'sensitivity' in result


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""
    
    def test_h0_sensitivity(self):
        """Test sensitivity to H0 measurement."""
        sensitivity = FineStructureSensitivityAnalysis()
        result = sensitivity.sensitivity_to_h0(n_points=3)
        
        assert 'base_alpha_inverse' in result
        assert 'parameter_variations' in result
        assert 'sensitivity_metrics' in result
        
        # Base should be around 137.032
        assert abs(result['base_alpha_inverse'] - 137.032) < 1.0
    
    def test_fundamental_constants_sensitivity(self):
        """Test sensitivity to fundamental constants."""
        sensitivity = FineStructureSensitivityAnalysis()
        result = sensitivity.sensitivity_to_fundamental_constants()
        
        assert 'base_alpha_inverse' in result
        assert 'variations' in result
        assert 'max_deviation' in result
    
    def test_formula_components_sensitivity(self):
        """Test sensitivity to formula components."""
        sensitivity = FineStructureSensitivityAnalysis()
        result = sensitivity.sensitivity_to_formula_components()
        
        assert 'base_alpha_inverse' in result
        assert 'variations' in result
        assert 'max_deviation' in result
    
    def test_full_sensitivity_analysis(self):
        """Test complete sensitivity analysis."""
        sensitivity = FineStructureSensitivityAnalysis()
        result = sensitivity.full_sensitivity_analysis()
        
        assert 'H0_sensitivity' in result
        assert 'fundamental_constants_sensitivity' in result
        assert 'formula_components_sensitivity' in result
        assert 'overall_robustness' in result


class TestStatisticalProperties:
    """Test statistical properties of validations."""
    
    def test_monte_carlo_reproducibility(self):
        """Test that Monte Carlo is reproducible with same seed."""
        validator1 = FineStructureMonteCarloValidator(n_samples=1000, random_state=42)
        validator2 = FineStructureMonteCarloValidator(n_samples=1000, random_state=42)
        
        result1 = validator1.run_mc_validation()
        result2 = validator2.run_mc_validation()
        
        # Should be identical with same seed
        assert result1['prediction'] == result2['prediction']
        assert abs(result1['deviation_sigma'] - result2['deviation_sigma']) < 1e-10
    
    def test_error_propagation_consistency(self):
        """Test that error propagation is consistent."""
        error_prop = FineStructureErrorPropagation()
        
        result1 = error_prop.propagate_alpha_inverse_uncertainty()
        result2 = error_prop.propagate_alpha_inverse_uncertainty()
        
        # Should be identical (deterministic)
        assert abs(result1['delta_alpha_inverse'] - result2['delta_alpha_inverse']) < 1e-10
