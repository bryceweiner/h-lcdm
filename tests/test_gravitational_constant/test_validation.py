"""
Tests for gravitational constant validation infrastructure.

The corrected formula G = πc⁵/(ℏH²N_P) has uncertainty dominated by H₀.
No f_quantum correction is used.
"""

import pytest
import numpy as np
from pipeline.gravitational_constant.error_propagation import GravitationalConstantErrorPropagation
from pipeline.gravitational_constant.monte_carlo import GravitationalConstantMonteCarloValidator
from pipeline.gravitational_constant.sensitivity import GravitationalConstantSensitivityAnalysis


class TestErrorPropagation:
    """Test error propagation analysis."""
    
    def test_g_uncertainty(self):
        """Test that uncertainty is propagated from inputs."""
        error_prop = GravitationalConstantErrorPropagation()
        result = error_prop.propagate_g_uncertainty()
        
        assert 'delta_G' in result
        assert 'relative_uncertainty' in result
        assert 'uncertainty_breakdown' in result
        assert result['delta_G'] > 0  # Should have uncertainty
    
    def test_h0_dominates_uncertainty(self):
        """Test that H₀ uncertainty dominates."""
        error_prop = GravitationalConstantErrorPropagation()
        result = error_prop.propagate_g_uncertainty()
        
        breakdown = result['uncertainty_breakdown']
        
        # H0 should dominate
        h0_contribution = breakdown['H0']['contribution_percent']
        alpha_contribution = breakdown['alpha_inverse']['contribution_percent']
        
        assert h0_contribution > 99.0  # H0 should dominate (>99%)
        assert alpha_contribution < 1.0  # Alpha should be negligible
    
    def test_no_f_quantum_in_breakdown(self):
        """Test that f_quantum is not in uncertainty breakdown."""
        error_prop = GravitationalConstantErrorPropagation()
        result = error_prop.propagate_g_uncertainty()
        
        breakdown = result['uncertainty_breakdown']
        
        # f_quantum should not be in breakdown
        assert 'f_quantum' not in breakdown
    
    def test_comparison_with_observation(self):
        """Test comparison with CODATA observation."""
        error_prop = GravitationalConstantErrorPropagation()
        result = error_prop.compare_with_observation()
        
        assert 'predicted' in result
        assert 'observed' in result
        assert 'relative_difference_percent' in result
        assert 'agreement' in result
        
        # Should agree within 2% relative difference
        assert result['relative_difference_percent'] < 2.0
    
    def test_agreement_quality(self):
        """Test that agreement quality is appropriate for parameter-free prediction."""
        error_prop = GravitationalConstantErrorPropagation()
        result = error_prop.compare_with_observation()
        
        # For ~1% agreement, should be 'good' or better
        assert result['agreement'] in ['excellent', 'very_good', 'good']
    
    def test_full_error_analysis(self):
        """Test complete error analysis."""
        error_prop = GravitationalConstantErrorPropagation()
        result = error_prop.full_error_analysis()
        
        assert 'g_uncertainty' in result
        assert 'comparison_with_observation' in result
        assert 'summary' in result


class TestMonteCarloValidation:
    """Test Monte Carlo uncertainty quantification."""
    
    def test_mc_validation(self):
        """Test Monte Carlo validation."""
        validator = GravitationalConstantMonteCarloValidator(n_samples=10000, random_state=42)
        result = validator.run_mc_validation()
        
        assert 'prediction' in result
        assert 'consistency_fractions' in result
        assert 'central_relative_difference_percent' in result
        assert 'interpretation' in result
        
        # Consistency fractions should be reasonable
        consistency = result['consistency_fractions']
        assert 0 <= consistency['p_consistent_1_percent'] <= 1
    
    def test_consistency_fractions(self):
        """Test that consistency fractions are calculated correctly."""
        validator = GravitationalConstantMonteCarloValidator(n_samples=100000, random_state=42)
        result = validator.run_mc_validation()
        
        consistency = result['consistency_fractions']
        
        # With H0 uncertainty ~1%, most samples should be within 2%
        assert consistency['p_consistent_2_percent'] > 0.5
    
    def test_no_f_quantum_parameter(self):
        """Test that f_quantum is not used in MC validation."""
        validator = GravitationalConstantMonteCarloValidator(n_samples=1000, random_state=42)
        
        # Should work without f_quantum parameters
        result = validator.run_mc_validation(
            alpha_inverse=137.036,
            delta_alpha_inverse=0.000000021
        )
        
        assert 'prediction' in result
        assert 'note' in result
        # Note should mention no correction factors
        assert 'correction' in result['note'].lower()
    
    def test_deviation_statistics(self):
        """Test deviation statistics."""
        validator = GravitationalConstantMonteCarloValidator(n_samples=10000, random_state=42)
        result = validator.run_mc_validation()
        
        assert 'deviation_statistics' in result
        stats = result['deviation_statistics']
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert 'percentiles' in stats
    
    def test_mc_1sigma_range(self):
        """Test that 1σ range is provided."""
        validator = GravitationalConstantMonteCarloValidator(n_samples=10000, random_state=42)
        result = validator.run_mc_validation()
        
        assert 'mc_1sigma_range' in result
        assert 'low' in result['mc_1sigma_range']
        assert 'high' in result['mc_1sigma_range']


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""
    
    def test_alpha_sensitivity(self):
        """Test sensitivity to α⁻¹ measurement."""
        sensitivity = GravitationalConstantSensitivityAnalysis()
        result = sensitivity.sensitivity_to_alpha_inverse(n_points=5)
        
        assert 'base_G' in result
        assert 'parameter_variations' in result
        assert 'sensitivity_metrics' in result
    
    def test_alpha_sensitivity_is_negligible(self):
        """Test that α⁻¹ sensitivity is negligible."""
        sensitivity = GravitationalConstantSensitivityAnalysis()
        result = sensitivity.sensitivity_to_alpha_inverse(n_points=5)
        
        # Relative sensitivity should be tiny
        rel_sens = result['sensitivity_metrics']['relative_sensitivity']
        assert rel_sens < 0.01  # Less than 1%
    
    def test_h0_sensitivity(self):
        """Test sensitivity to H₀ measurement."""
        sensitivity = GravitationalConstantSensitivityAnalysis()
        result = sensitivity.sensitivity_to_h0(n_points=5)
        
        assert 'base_G' in result
        assert 'parameter_variations' in result
        assert 'sensitivity_metrics' in result
        assert 'note' in result  # Should explain H0 dominance
    
    def test_h0_sensitivity_is_dominant(self):
        """Test that H₀ sensitivity is dominant."""
        sensitivity = GravitationalConstantSensitivityAnalysis()
        result = sensitivity.sensitivity_to_h0(n_points=5)
        
        # Should have significant sensitivity
        rel_sens = result['sensitivity_metrics']['relative_sensitivity']
        assert rel_sens > 0.01  # More than 1%
        
        # Elasticity should be ~-2 (G ∝ H⁻²)
        elasticity = result['sensitivity_metrics']['elasticity']
        assert np.abs(elasticity - (-2.0)) < 0.5  # Close to -2
    
    def test_hubble_tension_analysis(self):
        """Test Hubble tension sensitivity analysis."""
        sensitivity = GravitationalConstantSensitivityAnalysis()
        result = sensitivity.sensitivity_to_hubble_tension()
        
        assert 'planck_cmb' in result
        assert 'sh0es_cepheids' in result
        assert 'hubble_tension_effect' in result
        assert 'interpretation' in result
        
        # Planck should give better agreement than SH0ES
        planck_diff = result['planck_cmb']['relative_difference_percent']
        sh0es_diff = result['sh0es_cepheids']['relative_difference_percent']
        
        assert planck_diff < sh0es_diff
    
    def test_full_sensitivity_analysis(self):
        """Test complete sensitivity analysis."""
        sensitivity = GravitationalConstantSensitivityAnalysis()
        result = sensitivity.full_sensitivity_analysis()
        
        assert 'alpha_inverse_sensitivity' in result
        assert 'H0_sensitivity' in result
        assert 'hubble_tension_analysis' in result
        assert 'overall_assessment' in result
        
        # Overall assessment should mention H0 dominance
        assert 'H' in result['overall_assessment']


class TestStatisticalProperties:
    """Test statistical properties of validations."""
    
    def test_monte_carlo_reproducibility(self):
        """Test that Monte Carlo is reproducible with same seed."""
        validator1 = GravitationalConstantMonteCarloValidator(n_samples=1000, random_state=42)
        validator2 = GravitationalConstantMonteCarloValidator(n_samples=1000, random_state=42)
        
        result1 = validator1.run_mc_validation()
        result2 = validator2.run_mc_validation()
        
        # Should be identical with same seed
        assert abs(result1['prediction'] - result2['prediction']) < 1e-20
    
    def test_error_propagation_consistency(self):
        """Test that error propagation is consistent."""
        error_prop = GravitationalConstantErrorPropagation()
        
        result1 = error_prop.propagate_g_uncertainty()
        result2 = error_prop.propagate_g_uncertainty()
        
        # Should be identical (deterministic)
        assert abs(result1['delta_G'] - result2['delta_G']) < 1e-25
    
    def test_sensitivity_consistency(self):
        """Test that sensitivity analysis is consistent."""
        sens1 = GravitationalConstantSensitivityAnalysis()
        sens2 = GravitationalConstantSensitivityAnalysis()
        
        result1 = sens1.sensitivity_to_h0(n_points=3)
        result2 = sens2.sensitivity_to_h0(n_points=3)
        
        # Should be identical (deterministic)
        assert abs(result1['base_G'] - result2['base_G']) < 1e-25


class TestPhysicalConsistency:
    """Test physical consistency of validation results."""
    
    def test_prediction_is_positive(self):
        """Test that all predictions are positive."""
        error_prop = GravitationalConstantErrorPropagation()
        result = error_prop.compare_with_observation()
        
        assert result['predicted'] > 0
        assert result['observed'] > 0
    
    def test_uncertainty_is_positive(self):
        """Test that uncertainties are positive."""
        error_prop = GravitationalConstantErrorPropagation()
        result = error_prop.propagate_g_uncertainty()
        
        assert result['delta_G'] > 0
        assert result['relative_uncertainty'] > 0
    
    def test_consistency_fractions_sum_correctly(self):
        """Test that consistency fractions are ordered correctly."""
        validator = GravitationalConstantMonteCarloValidator(n_samples=10000, random_state=42)
        result = validator.run_mc_validation()
        
        consistency = result['consistency_fractions']
        
        # Larger thresholds should have higher fractions
        assert consistency['p_consistent_0_1_percent'] <= consistency['p_consistent_0_5_percent']
        assert consistency['p_consistent_0_5_percent'] <= consistency['p_consistent_1_percent']
        assert consistency['p_consistent_1_percent'] <= consistency['p_consistent_2_percent']
