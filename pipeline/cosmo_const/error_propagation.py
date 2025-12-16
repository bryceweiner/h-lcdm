"""
Error Propagation Analysis
==========================

Formal error propagation for cosmological constant prediction.

The prediction Ω_Λ = 0.6841 is parameter-free (no input uncertainty).
Uncertainty in comparison comes from Planck 2018 measurement error.
"""

import numpy as np
from typing import Dict, Any
from .physics import calculate_omega_lambda, calculate_lambda
from hlcdm.parameters import HLCDM_PARAMS


class ErrorPropagation:
    """
    Formal error propagation analysis for cosmological constant calculations.
    """
    
    def __init__(self):
        """Initialize error propagation calculator."""
        pass
    
    def propagate_omega_lambda_uncertainty(self) -> Dict[str, Any]:
        """
        Propagate uncertainty in Ω_Λ prediction.
        
        Since the prediction is parameter-free (mathematical constants only),
        the uncertainty is zero. The comparison uncertainty comes entirely
        from the observational measurement error.
        
        Returns:
            dict: Uncertainty analysis for Ω_Λ
        """
        omega_result = calculate_omega_lambda()
        Omega_Lambda = omega_result['omega_lambda']
        
        # Input uncertainties (all exact mathematical constants)
        delta_ln2 = 0.0  # Exact
        delta_ln3 = 0.0  # Exact
        delta_e = 0.0  # Exact
        
        # The prediction has zero theoretical uncertainty
        delta_Omega_Lambda_theory = 0.0
        
        return {
            'omega_lambda': Omega_Lambda,
            'delta_omega_lambda_theory': delta_Omega_Lambda_theory,
            'input_uncertainties': {
                'delta_ln2': delta_ln2,
                'delta_ln3': delta_ln3,
                'delta_e': delta_e
            },
            'note': 'Parameter-free prediction - no theoretical uncertainty'
        }
    
    def propagate_lambda_uncertainty(self, H0: float = None, delta_H0: float = None) -> Dict[str, Any]:
        """
        Propagate uncertainty in Λ calculation from H0 measurement.
        
        Λ = 3Ω_Λ H²/c²
        
        Uncertainty propagation:
        δΛ = |∂Λ/∂H0| × δH0 = |6Ω_Λ H/c²| × δH0
        
        Parameters:
            H0 (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
            delta_H0 (float, optional): Uncertainty in H0. Defaults to 0.5e-18 s^-1 (Planck).
        
        Returns:
            dict: Uncertainty analysis for Λ
        """
        if H0 is None:
            H0 = HLCDM_PARAMS.H0
        
        if delta_H0 is None:
            # Planck 2018 uncertainty: H0 = 67.4 ± 0.5 km/s/Mpc
            # Convert to s^-1: 0.5 km/s/Mpc = 0.5 × 1000 / (3.086e22) s^-1 ≈ 1.6e-20 s^-1
            # More conservative estimate: ~0.5e-18 s^-1 (fractional uncertainty)
            delta_H0 = 0.5e-18  # s^-1
        
        # Get Ω_Λ (zero uncertainty)
        omega_result = calculate_omega_lambda()
        Omega_Lambda = omega_result['omega_lambda']
        delta_Omega_Lambda = 0.0
        
        # Calculate Λ
        lambda_result = calculate_lambda(H0)
        Lambda = lambda_result['lambda']
        
        # Partial derivative: ∂Λ/∂H0 = 6Ω_Λ H/c²
        dLambda_dH0 = 6 * Omega_Lambda * H0 / HLCDM_PARAMS.C**2
        
        # Propagated uncertainty
        delta_Lambda_H0 = np.abs(dLambda_dH0) * delta_H0
        
        # Relative uncertainty
        relative_uncertainty = delta_Lambda_H0 / Lambda if Lambda != 0 else 0.0
        
        return {
            'lambda': Lambda,
            'delta_lambda': delta_Lambda_H0,
            'relative_uncertainty': relative_uncertainty,
            'H0': H0,
            'delta_H0': delta_H0,
            'omega_lambda': Omega_Lambda,
            'delta_omega_lambda': delta_Omega_Lambda,
            'partial_derivative': {
                'dLambda_dH0': float(dLambda_dH0),
                'formula': '6Ω_Λ H/c²'
            },
            'propagation_formula': 'δΛ = |6Ω_Λ H/c²| × δH0'
        }
    
    def compare_with_observation(self, 
                                 observed_omega_lambda: float = 0.6847,
                                 observed_sigma: float = 0.0073) -> Dict[str, Any]:
        """
        Compare prediction with Planck 2018 observation.
        
        Parameters:
            observed_omega_lambda (float): Observed Ω_Λ from Planck 2018
            observed_sigma (float): 1σ uncertainty from Planck 2018
        
        Returns:
            dict: Comparison analysis
        """
        omega_result = calculate_omega_lambda()
        predicted = omega_result['omega_lambda']
        
        # Deviation
        deviation = predicted - observed_omega_lambda
        
        # Deviation in sigma
        deviation_sigma = deviation / observed_sigma
        
        # Pull (normalized deviation)
        pull = deviation_sigma
        
        # Consistency check (within 2σ)
        consistent_2sigma = np.abs(deviation_sigma) < 2.0
        
        # Consistency check (within 1σ)
        consistent_1sigma = np.abs(deviation_sigma) < 1.0
        
        return {
            'predicted': predicted,
            'observed': observed_omega_lambda,
            'observed_sigma': observed_sigma,
            'deviation': deviation,
            'deviation_sigma': deviation_sigma,
            'pull': pull,
            'consistent_1sigma': consistent_1sigma,
            'consistent_2sigma': consistent_2sigma,
            'agreement': 'excellent' if np.abs(deviation_sigma) < 0.2 else 'good' if np.abs(deviation_sigma) < 1.0 else 'marginal'
        }
    
    def full_error_analysis(self) -> Dict[str, Any]:
        """
        Perform complete error propagation analysis.
        
        Returns:
            dict: Complete error analysis including all components
        """
        omega_uncertainty = self.propagate_omega_lambda_uncertainty()
        lambda_uncertainty = self.propagate_lambda_uncertainty()
        comparison = self.compare_with_observation()
        
        return {
            'omega_lambda_uncertainty': omega_uncertainty,
            'lambda_uncertainty': lambda_uncertainty,
            'comparison_with_observation': comparison,
            'summary': {
                'prediction_uncertainty': 'zero (parameter-free)',
                'comparison_uncertainty_source': 'Planck 2018 measurement error',
                'deviation_from_observation': f"{comparison['deviation_sigma']:.2f}σ"
            }
        }
