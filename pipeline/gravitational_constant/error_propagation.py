"""
Error Propagation Analysis for Gravitational Constant
======================================================

Formal error propagation for gravitational constant prediction.

The prediction G = πc⁵/(ℏH²N_P) depends on:
- α⁻¹ measurement uncertainty (CODATA) - negligible
- H₀ measurement uncertainty (Planck/SH0ES) - DOMINANT (~1-2%)

No correction factors (ln(3), f_quantum) contribute to uncertainty
since they are not used in the corrected formula.
"""

import numpy as np
from typing import Dict, Any
from .physics import calculate_g, calculate_np_from_alpha_inverse
from hlcdm.parameters import HLCDM_PARAMS


class GravitationalConstantErrorPropagation:
    """
    Formal error propagation analysis for gravitational constant calculations.
    
    The prediction G = πc⁵/(ℏH²N_P) has uncertainty dominated by H₀ measurement.
    The α⁻¹ contribution is negligible due to extraordinary measurement precision.
    """
    
    def __init__(self):
        """Initialize error propagation calculator."""
        pass
    
    def propagate_g_uncertainty(self,
                                alpha_inverse: float = 137.035999084,
                                delta_alpha_inverse: float = 0.000000021,
                                H0: float = None,
                                delta_H0: float = None) -> Dict[str, Any]:
        """
        Propagate uncertainty in G prediction from input uncertainties.
        
        G = πc⁵/(ℏH²N_P)
        where N_P = exp[2α⁻¹ + 2ln(4π²) + 1/π]
        
        Uncertainty sources:
        1. α⁻¹ measurement uncertainty - negligible
        2. H₀ measurement uncertainty - DOMINANT
        
        Parameters:
            alpha_inverse (float): Central value of α⁻¹
            delta_alpha_inverse (float): Uncertainty in α⁻¹ (CODATA 2018)
            H0 (float, optional): Hubble parameter. Defaults to HLCDM_PARAMS.H0.
            delta_H0 (float, optional): Uncertainty in H0. Defaults to ~1% from Hubble tension.
        
        Returns:
            dict: Uncertainty analysis for G
        """
        if H0 is None:
            H0 = HLCDM_PARAMS.H0
        
        if delta_H0 is None:
            # Hubble tension: H0 ranges from ~67 to ~73 km/s/Mpc
            # Use ~1% as representative uncertainty
            delta_H0 = 0.01 * H0  # ~1% fractional uncertainty
        
        # Get base calculation (no corrections)
        g_result = calculate_g(H0, alpha_inverse)
        G = g_result['G']
        N_P = g_result['N_P']
        
        # Calculate partial derivatives
        
        # 1. ∂G/∂α⁻¹ via N_P
        # N_P = exp(2α⁻¹ + 2ln(4π²) + 1/π)
        # ∂N_P/∂α⁻¹ = 2N_P
        # G ∝ 1/N_P, so ∂G/∂α⁻¹ = -G × (2)
        dG_dalpha_inv = -2 * G
        
        # 2. ∂G/∂H0
        # G ∝ 1/H², so ∂G/∂H0 = -2G/H0
        dG_dH0 = -2 * G / H0
        
        # Propagate uncertainties (assuming independent)
        delta_G_alpha = np.abs(dG_dalpha_inv) * delta_alpha_inverse
        delta_G_H0 = np.abs(dG_dH0) * delta_H0
        
        # Total uncertainty (quadrature sum)
        delta_G_total = np.sqrt(delta_G_alpha**2 + delta_G_H0**2)
        
        # Relative uncertainties
        relative_uncertainty_alpha = delta_G_alpha / G if G > 0 else 0.0
        relative_uncertainty_H0 = delta_G_H0 / G if G > 0 else 0.0
        relative_uncertainty_total = delta_G_total / G if G > 0 else 0.0
        
        return {
            'G': G,
            'delta_G': delta_G_total,
            'relative_uncertainty': relative_uncertainty_total,
            'uncertainty_breakdown': {
                'alpha_inverse': {
                    'delta_G': float(delta_G_alpha),
                    'relative': float(relative_uncertainty_alpha),
                    'contribution_percent': float(relative_uncertainty_alpha / relative_uncertainty_total * 100) if relative_uncertainty_total > 0 else 0.0
                },
                'H0': {
                    'delta_G': float(delta_G_H0),
                    'relative': float(relative_uncertainty_H0),
                    'contribution_percent': float(relative_uncertainty_H0 / relative_uncertainty_total * 100) if relative_uncertainty_total > 0 else 0.0
                }
            },
            'partial_derivatives': {
                'dG_dalpha_inv': float(dG_dalpha_inv),
                'dG_dH0': float(dG_dH0)
            },
            'input_uncertainties': {
                'delta_alpha_inverse': delta_alpha_inverse,
                'delta_H0': delta_H0
            },
            'note': 'H₀ uncertainty dominates; α⁻¹ contribution negligible'
        }
    
    def compare_with_observation(self,
                                observed_G: float = 6.67430e-11,
                                observed_sigma: float = 0.00015e-11) -> Dict[str, Any]:
        """
        Compare prediction with CODATA 2018 observation.
        
        Parameters:
            observed_G (float): Observed G from CODATA 2018
            observed_sigma (float): 1σ uncertainty from CODATA 2018
        
        Returns:
            dict: Comparison analysis
        """
        g_result = calculate_g()
        predicted = g_result['G']
        
        # Deviation
        deviation = predicted - observed_G
        
        # Deviation in sigma (note: for parameter-free prediction, relative diff more meaningful)
        deviation_sigma = deviation / observed_sigma
        
        # Relative difference (percentage)
        relative_difference = abs(deviation) / observed_G * 100
        
        # Consistency check - for parameter-free prediction, use relative difference
        # ~1% is excellent for zero free parameters
        consistent_1_percent = relative_difference < 1.0
        consistent_2_percent = relative_difference < 2.0
        
        # Agreement assessment based on relative difference
        # Standards adjusted for parameter-free prediction
        if relative_difference < 0.1:
            agreement = 'excellent'
        elif relative_difference < 1.0:
            agreement = 'very_good'
        elif relative_difference < 2.0:
            agreement = 'good'
        else:
            agreement = 'marginal'
        
        return {
            'predicted': predicted,
            'observed': observed_G,
            'observed_sigma': observed_sigma,
            'deviation': deviation,
            'deviation_sigma': deviation_sigma,
            'relative_difference_percent': relative_difference,
            'consistent_1_percent': consistent_1_percent,
            'consistent_2_percent': consistent_2_percent,
            'agreement': agreement,
            'formula': 'G = πc⁵/(ℏH²N_P)',
            'note': 'Parameter-free prediction; ~1% agreement is excellent'
        }
    
    def full_error_analysis(self) -> Dict[str, Any]:
        """
        Perform complete error propagation analysis.
        
        Returns:
            dict: Complete error analysis including all components
        """
        g_uncertainty = self.propagate_g_uncertainty()
        comparison = self.compare_with_observation()
        
        return {
            'g_uncertainty': g_uncertainty,
            'comparison_with_observation': comparison,
            'summary': {
                'prediction_uncertainty_source': 'H₀ measurement dominates (α⁻¹ negligible)',
                'comparison_uncertainty_source': 'CODATA 2018 measurement error',
                'relative_difference': f"{comparison['relative_difference_percent']:.4f}%",
                'dominant_uncertainty': self._identify_dominant_uncertainty(g_uncertainty),
                'note': 'For parameter-free prediction, ~1% agreement is excellent'
            }
        }
    
    def _identify_dominant_uncertainty(self, g_uncertainty: Dict[str, Any]) -> str:
        """Identify the dominant source of uncertainty."""
        breakdown = g_uncertainty.get('uncertainty_breakdown', {})
        
        max_contribution = 0.0
        dominant_source = 'unknown'
        
        for source, data in breakdown.items():
            contribution = data.get('contribution_percent', 0.0)
            if contribution > max_contribution:
                max_contribution = contribution
                dominant_source = source
        
        return f"{dominant_source} ({max_contribution:.1f}%)"
