"""
Sensitivity Analysis for Fine Structure Constant
=================================================

Analyze sensitivity of α⁻¹ prediction to:
1. H0 measurement uncertainty
2. Fundamental constant uncertainties
3. Formula component variations
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from .physics import calculate_alpha_inverse, calculate_bekenstein_hawking_entropy


class FineStructureSensitivityAnalysis:
    """
    Sensitivity analysis for fine structure constant prediction.
    
    Tests robustness of prediction to variations in:
    - H0 measurement
    - Fundamental constants (c, ℏ, G)
    - Formula components
    """
    
    def __init__(self):
        """Initialize sensitivity analyzer."""
        pass
    
    def sensitivity_to_h0(self,
                          H0_range: Tuple[float, float] = (2.15e-18, 2.22e-18),
                          n_points: int = 20) -> Dict[str, Any]:
        """
        Analyze sensitivity to H0 measurement.
        
        Base case: H0 = 2.184e-18 s^-1 (Planck 2018)
        
        Parameters:
            H0_range (tuple): Range for H0 values
            n_points (int): Number of points in range
        
        Returns:
            dict: Sensitivity analysis results
        """
        H0_values = np.linspace(H0_range[0], H0_range[1], n_points)
        
        results = []
        base_alpha_inv = None
        
        for H0 in H0_values:
            alpha_inv_result = calculate_alpha_inverse(H0)
            alpha_inv = alpha_inv_result['alpha_inverse']
            
            if abs(H0 - 2.184e-18) < 1e-20:  # Base case
                base_alpha_inv = alpha_inv
            
            results.append({
                'H0': float(H0),
                'alpha_inverse': float(alpha_inv),
                'deviation_from_base': float(alpha_inv - base_alpha_inv) if base_alpha_inv is not None else 0.0
            })
        
        # Calculate sensitivity metrics
        alpha_inv_values = [r['alpha_inverse'] for r in results]
        deviations = [r['deviation_from_base'] for r in results if base_alpha_inv is not None]
        
        return {
            'base_H0': 2.184e-18,
            'base_alpha_inverse': float(base_alpha_inv) if base_alpha_inv is not None else None,
            'parameter_variations': results,
            'sensitivity_metrics': {
                'min_alpha_inverse': float(np.min(alpha_inv_values)),
                'max_alpha_inverse': float(np.max(alpha_inv_values)),
                'range': float(np.max(alpha_inv_values) - np.min(alpha_inv_values)),
                'std_alpha_inverse': float(np.std(alpha_inv_values)),
                'max_deviation': float(np.max(np.abs(deviations))) if deviations else 0.0
            },
            'interpretation': self._interpret_h0_sensitivity(deviations) if deviations else 'N/A'
        }
    
    def sensitivity_to_fundamental_constants(self) -> Dict[str, Any]:
        """
        Analyze sensitivity to fundamental constant uncertainties.
        
        Tests variations in c, ℏ, G within their measurement uncertainties.
        
        Returns:
            dict: Sensitivity analysis results
        """
        from hlcdm.parameters import HLCDM_PARAMS
        
        # Base calculation
        base_result = calculate_alpha_inverse()
        base_alpha_inv = base_result['alpha_inverse']
        
        # Typical uncertainties (fractional)
        delta_c_c = 0.0  # c is exact (definition)
        delta_hbar_hbar = 1e-8  # ~10^-8 relative uncertainty
        delta_G_G = 2.2e-5  # ~2.2×10^-5 relative uncertainty (CODATA)
        
        variations = []
        
        # Vary ℏ
        for factor in [1 - delta_hbar_hbar, 1.0, 1 + delta_hbar_hbar]:
            # Note: This requires modifying HLCDM_PARAMS, so we'll use approximation
            # In practice, would need to pass modified constants to calculation
            # For now, estimate sensitivity analytically
            # ∂(α⁻¹)/∂ℏ ≈ (1/2) × (1/S_H) × (-πc⁵/(ℏ²GH²))
            # Relative change ≈ -delta_hbar/hbar
            relative_change = -(factor - 1.0)
            alpha_inv_variation = base_alpha_inv * (1 + relative_change * 0.5)  # Approximate
            variations.append({
                'constant': 'hbar',
                'factor': float(factor),
                'alpha_inverse': float(alpha_inv_variation),
                'deviation': float(alpha_inv_variation - base_alpha_inv)
            })
        
        # Vary G
        for factor in [1 - delta_G_G, 1.0, 1 + delta_G_G]:
            relative_change = -(factor - 1.0)
            alpha_inv_variation = base_alpha_inv * (1 + relative_change * 0.5)  # Approximate
            variations.append({
                'constant': 'G',
                'factor': float(factor),
                'alpha_inverse': float(alpha_inv_variation),
                'deviation': float(alpha_inv_variation - base_alpha_inv)
            })
        
        # Calculate maximum sensitivity
        max_deviation = max(abs(v['deviation']) for v in variations)
        
        return {
            'base_alpha_inverse': base_alpha_inv,
            'variations': variations,
            'max_deviation': float(max_deviation),
            'interpretation': self._interpret_constant_sensitivity(max_deviation)
        }
    
    def sensitivity_to_formula_components(self) -> Dict[str, Any]:
        """
        Analyze sensitivity to formula component variations.
        
        Tests variations in:
        - Holographic term coefficient (1/2)
        - Geometric term (ln(4π²))
        - Vacuum term (1/(2π))
        
        Returns:
            dict: Sensitivity analysis results
        """
        base_result = calculate_alpha_inverse()
        base_alpha_inv = base_result['alpha_inverse']
        
        entropy_result = calculate_bekenstein_hawking_entropy()
        ln_S_H = entropy_result['ln_S_H']
        
        variations = []
        
        # Vary holographic term coefficient
        for coeff in [0.45, 0.5, 0.55]:
            alpha_inv = coeff * ln_S_H - np.log(4 * np.pi**2) - 1.0 / (2 * np.pi)
            variations.append({
                'component': 'holographic_coefficient',
                'value': float(coeff),
                'alpha_inverse': float(alpha_inv),
                'deviation': float(alpha_inv - base_alpha_inv)
            })
        
        # Vary geometric term
        base_geometric = np.log(4 * np.pi**2)
        for factor in [0.95, 1.0, 1.05]:
            geometric_term = base_geometric * factor
            alpha_inv = 0.5 * ln_S_H - geometric_term - 1.0 / (2 * np.pi)
            variations.append({
                'component': 'geometric_term',
                'factor': float(factor),
                'alpha_inverse': float(alpha_inv),
                'deviation': float(alpha_inv - base_alpha_inv)
            })
        
        # Vary vacuum term
        base_vacuum = 1.0 / (2 * np.pi)
        for factor in [0.95, 1.0, 1.05]:
            vacuum_term = base_vacuum * factor
            alpha_inv = 0.5 * ln_S_H - np.log(4 * np.pi**2) - vacuum_term
            variations.append({
                'component': 'vacuum_term',
                'factor': float(factor),
                'alpha_inverse': float(alpha_inv),
                'deviation': float(alpha_inv - base_alpha_inv)
            })
        
        max_deviation = max(abs(v['deviation']) for v in variations)
        
        return {
            'base_alpha_inverse': base_alpha_inv,
            'variations': variations,
            'max_deviation': float(max_deviation),
            'interpretation': self._interpret_component_sensitivity(max_deviation)
        }
    
    def _interpret_h0_sensitivity(self, deviations: List[float]) -> str:
        """Interpret sensitivity to H0 variations."""
        max_dev = np.max(np.abs(deviations))
        if max_dev < 0.001:
            return "Highly robust: <0.001 variation"
        elif max_dev < 0.01:
            return "Robust: <0.01 variation"
        elif max_dev < 0.1:
            return "Moderately sensitive: <0.1 variation"
        else:
            return f"Sensitive: {max_dev:.3f} variation"
    
    def _interpret_constant_sensitivity(self, max_deviation: float) -> str:
        """Interpret sensitivity to fundamental constant variations."""
        if max_deviation < 0.001:
            return "Robust to fundamental constant uncertainties"
        elif max_deviation < 0.01:
            return "Moderately sensitive to fundamental constants"
        else:
            return f"Sensitive to fundamental constants: {max_deviation:.4f} variation"
    
    def _interpret_component_sensitivity(self, max_deviation: float) -> str:
        """Interpret sensitivity to formula component variations."""
        if max_deviation < 0.1:
            return "Robust to formula component variations"
        elif max_deviation < 1.0:
            return "Moderately sensitive to formula components"
        else:
            return f"Sensitive to formula components: {max_deviation:.2f} variation"
    
    def full_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Perform complete sensitivity analysis.
        
        Returns:
            dict: Complete sensitivity analysis results
        """
        h0_sensitivity = self.sensitivity_to_h0()
        constant_sensitivity = self.sensitivity_to_fundamental_constants()
        component_sensitivity = self.sensitivity_to_formula_components()
        
        return {
            'H0_sensitivity': h0_sensitivity,
            'fundamental_constants_sensitivity': constant_sensitivity,
            'formula_components_sensitivity': component_sensitivity,
            'overall_robustness': self._assess_overall_robustness(
                h0_sensitivity,
                constant_sensitivity,
                component_sensitivity
            )
        }
    
    def _assess_overall_robustness(self, *sensitivity_results) -> str:
        """Assess overall robustness from all sensitivity tests."""
        # Check if all variations are small
        all_robust = True
        for result in sensitivity_results:
            if 'sensitivity_metrics' in result:
                max_dev = result['sensitivity_metrics'].get('max_deviation', 1.0)
                if max_dev > 0.01:  # >0.01 variation
                    all_robust = False
            elif 'max_deviation' in result:
                max_dev = result['max_deviation']
                if max_dev > 0.01:
                    all_robust = False
        
        if all_robust:
            return "Prediction is robust to parameter variations"
        else:
            return "Prediction shows some sensitivity to parameter variations"
