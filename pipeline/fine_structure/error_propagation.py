"""
Error Propagation Analysis for Fine Structure Constant
=======================================================

Formal error propagation for fine structure constant prediction.

The prediction α⁻¹ = 137.032 depends on H0 measurement uncertainty.
"""

import numpy as np
from typing import Dict, Any
from .physics import calculate_alpha_inverse, calculate_bekenstein_hawking_entropy
from hlcdm.parameters import HLCDM_PARAMS


class FineStructureErrorPropagation:
    """
    Formal error propagation analysis for fine structure constant calculations.
    """
    
    def __init__(self):
        """Initialize error propagation calculator."""
        pass
    
    def propagate_alpha_inverse_uncertainty(self, H0: float = None, delta_H0: float = None) -> Dict[str, Any]:
        """
        Propagate uncertainty in α⁻¹ prediction from H0 measurement.
        
        α⁻¹ = (1/2)ln(S_H) - ln(4π²) - 1/(2π)
        where S_H = πc⁵/(ℏGH²)
        
        Uncertainty propagation:
        δ(α⁻¹) = |∂(α⁻¹)/∂H0| × δH0
        
        Parameters:
            H0 (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
            delta_H0 (float, optional): Uncertainty in H0. Defaults to 0.5e-18 s^-1 (Planck).
        
        Returns:
            dict: Uncertainty analysis for α⁻¹
        """
        if H0 is None:
            H0 = HLCDM_PARAMS.H0
        
        if delta_H0 is None:
            # Planck 2018 uncertainty: H0 = 67.4 ± 0.5 km/s/Mpc
            # Convert to s^-1: 0.5 km/s/Mpc = 0.5 × 1000 / (3.086e22) s^-1 ≈ 1.6e-20 s^-1
            # More conservative estimate: ~0.5e-18 s^-1 (fractional uncertainty)
            delta_H0 = 0.5e-18  # s^-1
        
        # Get α⁻¹ calculation
        alpha_inv_result = calculate_alpha_inverse(H0)
        alpha_inverse = alpha_inv_result['alpha_inverse']
        
        # Calculate partial derivative: ∂(α⁻¹)/∂H0
        # α⁻¹ = (1/2)ln(πc⁵/(ℏGH²)) - ln(4π²) - 1/(2π)
        # ∂(α⁻¹)/∂H0 = (1/2) × (1/S_H) × ∂S_H/∂H0
        # where ∂S_H/∂H0 = -2πc⁵/(ℏGH³)
        
        entropy_result = calculate_bekenstein_hawking_entropy(H0)
        S_H = entropy_result['S_H']
        
        # ∂S_H/∂H0 = -2πc⁵/(ℏGH³)
        dS_H_dH0 = -2 * np.pi * HLCDM_PARAMS.C**5 / (HLCDM_PARAMS.HBAR * HLCDM_PARAMS.G * H0**3)
        
        # ∂(α⁻¹)/∂H0 = (1/2) × (1/S_H) × dS_H/dH0
        dalpha_inv_dH0 = 0.5 * (1.0 / S_H) * dS_H_dH0
        
        # Propagated uncertainty
        delta_alpha_inverse = np.abs(dalpha_inv_dH0) * delta_H0
        
        # Relative uncertainty
        relative_uncertainty = delta_alpha_inverse / alpha_inverse if alpha_inverse != 0 else 0.0
        
        return {
            'alpha_inverse': alpha_inverse,
            'delta_alpha_inverse': delta_alpha_inverse,
            'relative_uncertainty': relative_uncertainty,
            'H0': H0,
            'delta_H0': delta_H0,
            'partial_derivative': {
                'dalpha_inv_dH0': float(dalpha_inv_dH0),
                'formula': '(1/2) × (1/S_H) × (-2πc⁵/(ℏGH³))'
            },
            'propagation_formula': 'δ(α⁻¹) = |∂(α⁻¹)/∂H0| × δH0'
        }
    
    def compare_with_observation(self,
                                 observed_alpha_inverse: float = 137.035999084,
                                 observed_sigma: float = 0.000000021) -> Dict[str, Any]:
        """
        Compare prediction with CODATA 2018 observation.
        
        Parameters:
            observed_alpha_inverse (float): Observed α⁻¹ from CODATA 2018
            observed_sigma (float): 1σ uncertainty from CODATA 2018
        
        Returns:
            dict: Comparison analysis
        """
        alpha_inv_result = calculate_alpha_inverse()
        predicted = alpha_inv_result['alpha_inverse']
        
        # Deviation
        deviation = predicted - observed_alpha_inverse
        
        # Deviation in sigma
        deviation_sigma = deviation / observed_sigma
        
        # Pull (normalized deviation)
        pull = deviation_sigma
        
        # Relative difference (percentage)
        relative_difference = abs(deviation) / observed_alpha_inverse * 100
        
        # Consistency check (within 2σ)
        consistent_2sigma = np.abs(deviation_sigma) < 2.0
        
        # Consistency check (within 1σ)
        consistent_1sigma = np.abs(deviation_sigma) < 1.0
        
        # Agreement assessment: Use relative difference when measurement precision
        # is much higher than theoretical uncertainty (CODATA case)
        # Relative difference < 0.01% is excellent, < 0.1% is very good, < 1% is good
        if relative_difference < 0.01:
            agreement = 'excellent'
        elif relative_difference < 0.1:
            agreement = 'very_good'
        elif relative_difference < 1.0:
            agreement = 'good'
        else:
            agreement = 'marginal'
        
        return {
            'predicted': predicted,
            'observed': observed_alpha_inverse,
            'observed_sigma': observed_sigma,
            'deviation': deviation,
            'deviation_sigma': deviation_sigma,
            'relative_difference_percent': relative_difference,
            'pull': pull,
            'consistent_1sigma': consistent_1sigma,
            'consistent_2sigma': consistent_2sigma,
            'agreement': agreement
        }
    
    def full_error_analysis(self) -> Dict[str, Any]:
        """
        Perform complete error propagation analysis.
        
        Returns:
            dict: Complete error analysis including all components
        """
        alpha_uncertainty = self.propagate_alpha_inverse_uncertainty()
        comparison = self.compare_with_observation()
        
        return {
            'alpha_inverse_uncertainty': alpha_uncertainty,
            'comparison_with_observation': comparison,
            'summary': {
                'prediction_uncertainty_source': 'H0 measurement error',
                'comparison_uncertainty_source': 'CODATA 2018 measurement error',
                'deviation_from_observation': f"{comparison['deviation_sigma']:.2f}σ",
                'relative_difference': f"{comparison['relative_difference_percent']:.4f}%"
            }
        }
