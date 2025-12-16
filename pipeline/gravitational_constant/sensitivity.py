"""
Sensitivity Analysis for Gravitational Constant
================================================

Analyze sensitivity of G prediction to:
1. α⁻¹ measurement uncertainty
2. H₀ measurement uncertainty (DOMINANT)

No correction factors (ln(3), f_quantum) are used in the formula.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from .physics import calculate_g


class GravitationalConstantSensitivityAnalysis:
    """
    Sensitivity analysis for gravitational constant prediction.
    
    Tests robustness of prediction G = πc⁵/(ℏH²N_P) to variations in:
    - α⁻¹ measurement (negligible effect)
    - H₀ measurement (DOMINANT effect due to G ∝ H⁻²)
    """
    
    def __init__(self):
        """Initialize sensitivity analyzer."""
        pass
    
    def sensitivity_to_alpha_inverse(self,
                                     alpha_range: Tuple[float, float] = (137.035, 137.037),
                                     n_points: int = 20) -> Dict[str, Any]:
        """
        Analyze sensitivity to α⁻¹ measurement.
        
        Parameters:
            alpha_range (tuple): Range for α⁻¹ values
            n_points (int): Number of points in range
        
        Returns:
            dict: Sensitivity analysis results
        """
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_points)
        
        results = []
        base_G = None
        base_alpha = 137.035999084
        
        for alpha_inv in alpha_values:
            g_result = calculate_g(alpha_inverse=alpha_inv)
            G = g_result['G']
            
            if abs(alpha_inv - base_alpha) < 1e-6:  # Base case
                base_G = G
            
            results.append({
                'alpha_inverse': float(alpha_inv),
                'G': float(G),
                'deviation_from_base': float(G - base_G) if base_G is not None else 0.0
            })
        
        # If we didn't hit the base case exactly, calculate it
        if base_G is None:
            g_result = calculate_g(alpha_inverse=base_alpha)
            base_G = g_result['G']
            # Update deviations
            for r in results:
                r['deviation_from_base'] = float(r['G'] - base_G)
        
        # Calculate sensitivity metrics
        G_values = [r['G'] for r in results]
        deviations = [r['deviation_from_base'] for r in results]
        
        return {
            'base_alpha_inverse': base_alpha,
            'base_G': float(base_G),
            'parameter_variations': results,
            'sensitivity_metrics': {
                'min_G': float(np.min(G_values)),
                'max_G': float(np.max(G_values)),
                'range': float(np.max(G_values) - np.min(G_values)),
                'std_G': float(np.std(G_values)),
                'max_deviation': float(np.max(np.abs(deviations))),
                'relative_sensitivity': float(np.std(G_values) / base_G) if base_G > 0 else 0.0
            },
            'interpretation': self._interpret_alpha_sensitivity(deviations, base_G)
        }
    
    def sensitivity_to_h0(self,
                         H0_range: Tuple[float, float] = (2.15e-18, 2.37e-18),
                         n_points: int = 20) -> Dict[str, Any]:
        """
        Analyze sensitivity to H₀ measurement.
        
        The range covers the Hubble tension: ~67 to ~73 km/s/Mpc.
        
        Parameters:
            H0_range (tuple): Range for H₀ values in s⁻¹
            n_points (int): Number of points in range
        
        Returns:
            dict: Sensitivity analysis results
        """
        from hlcdm.parameters import HLCDM_PARAMS
        
        H0_values = np.linspace(H0_range[0], H0_range[1], n_points)
        base_H0 = HLCDM_PARAMS.H0
        
        results = []
        base_G = None
        
        for H0 in H0_values:
            g_result = calculate_g(H=H0)
            G = g_result['G']
            
            if abs(H0 - base_H0) < 1e-20:  # Base case
                base_G = G
            
            results.append({
                'H0': float(H0),
                'H0_km_s_Mpc': float(H0 * 3.086e22 / 1000),  # Convert to km/s/Mpc
                'G': float(G),
                'deviation_from_base': float(G - base_G) if base_G is not None else 0.0
            })
        
        # If we didn't hit the base case exactly, calculate it
        if base_G is None:
            g_result = calculate_g(H=base_H0)
            base_G = g_result['G']
            # Update deviations
            for r in results:
                r['deviation_from_base'] = float(r['G'] - base_G)
        
        G_values = [r['G'] for r in results]
        deviations = [r['deviation_from_base'] for r in results]
        
        # Calculate relative sensitivity: dG/G per dH/H
        # Since G ∝ H⁻², we expect dG/G = -2 dH/H
        H0_frac_range = (H0_range[1] - H0_range[0]) / base_H0
        G_frac_range = (np.max(G_values) - np.min(G_values)) / base_G
        elasticity = G_frac_range / H0_frac_range if H0_frac_range > 0 else 0
        
        return {
            'base_H0': float(base_H0),
            'base_H0_km_s_Mpc': float(base_H0 * 3.086e22 / 1000),
            'base_G': float(base_G),
            'parameter_variations': results,
            'sensitivity_metrics': {
                'min_G': float(np.min(G_values)),
                'max_G': float(np.max(G_values)),
                'range': float(np.max(G_values) - np.min(G_values)),
                'std_G': float(np.std(G_values)),
                'max_deviation': float(np.max(np.abs(deviations))),
                'relative_sensitivity': float(np.std(G_values) / base_G) if base_G > 0 else 0.0,
                'elasticity': float(elasticity),
                'expected_elasticity': -2.0  # G ∝ H⁻²
            },
            'interpretation': self._interpret_h0_sensitivity(deviations, base_G, elasticity),
            'note': 'H₀ is the DOMINANT source of uncertainty due to G ∝ H⁻²'
        }
    
    def sensitivity_to_hubble_tension(self) -> Dict[str, Any]:
        """
        Specifically analyze how the Hubble tension affects G prediction.
        
        Compares predictions using:
        - Planck CMB: H₀ = 67.4 km/s/Mpc
        - SH0ES Cepheids: H₀ = 73.0 km/s/Mpc
        
        Returns:
            dict: Hubble tension sensitivity analysis
        """
        # Convert km/s/Mpc to s⁻¹
        # H0 [s⁻¹] = H0 [km/s/Mpc] × (1000 m/km) / (3.086e22 m/Mpc)
        H0_planck = 67.4 * 1000 / 3.086e22  # s⁻¹
        H0_sh0es = 73.0 * 1000 / 3.086e22  # s⁻¹
        
        G_codata = 6.67430e-11  # m³/(kg·s²)
        
        # Calculate G for each
        g_planck = calculate_g(H=H0_planck)
        g_sh0es = calculate_g(H=H0_sh0es)
        
        G_planck = g_planck['G']
        G_sh0es = g_sh0es['G']
        
        return {
            'planck_cmb': {
                'H0_km_s_Mpc': 67.4,
                'H0_s_inv': float(H0_planck),
                'G_predicted': float(G_planck),
                'G_codata': G_codata,
                'relative_difference_percent': float(abs(G_planck - G_codata) / G_codata * 100)
            },
            'sh0es_cepheids': {
                'H0_km_s_Mpc': 73.0,
                'H0_s_inv': float(H0_sh0es),
                'G_predicted': float(G_sh0es),
                'G_codata': G_codata,
                'relative_difference_percent': float(abs(G_sh0es - G_codata) / G_codata * 100)
            },
            'hubble_tension_effect': {
                'H0_difference_percent': float(abs(73.0 - 67.4) / 67.4 * 100),
                'G_difference_percent': float(abs(G_sh0es - G_planck) / G_planck * 100),
                'elasticity': float((G_sh0es - G_planck) / G_planck / ((H0_sh0es - H0_planck) / H0_planck))
            },
            'interpretation': self._interpret_hubble_tension(G_planck, G_sh0es, G_codata)
        }
    
    def _interpret_alpha_sensitivity(self, deviations: List[float], base_G: float) -> str:
        """Interpret sensitivity to α⁻¹ variations."""
        max_dev = np.max(np.abs(deviations))
        relative = max_dev / base_G if base_G > 0 else 0.0
        if relative < 0.001:
            return "Negligible: <0.1% variation (α⁻¹ measurement precision is extraordinary)"
        elif relative < 0.01:
            return f"Small: {relative*100:.3f}% variation"
        else:
            return f"Significant: {relative*100:.2f}% variation"
    
    def _interpret_h0_sensitivity(self, deviations: List[float], base_G: float, elasticity: float) -> str:
        """Interpret sensitivity to H₀ variations."""
        max_dev = np.max(np.abs(deviations))
        relative = max_dev / base_G if base_G > 0 else 0.0
        return f"DOMINANT: G ∝ H⁻² (elasticity = {elasticity:.2f}, expected -2.0). " \
               f"Range: {relative*100:.1f}% variation. Hubble tension directly affects G prediction."
    
    def _interpret_hubble_tension(self, G_planck: float, G_sh0es: float, G_codata: float) -> str:
        """Interpret how Hubble tension affects the prediction."""
        planck_diff = abs(G_planck - G_codata) / G_codata * 100
        sh0es_diff = abs(G_sh0es - G_codata) / G_codata * 100
        
        if planck_diff < sh0es_diff:
            return f"Framework FAVORS Planck H₀ (67.4 km/s/Mpc): {planck_diff:.2f}% vs {sh0es_diff:.2f}% deviation from CODATA G"
        else:
            return f"Framework favors SH0ES H₀ (73.0 km/s/Mpc): {sh0es_diff:.2f}% vs {planck_diff:.2f}% deviation from CODATA G"
    
    def full_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Perform complete sensitivity analysis.
        
        Returns:
            dict: Complete sensitivity analysis results
        """
        alpha_sensitivity = self.sensitivity_to_alpha_inverse()
        h0_sensitivity = self.sensitivity_to_h0()
        hubble_tension = self.sensitivity_to_hubble_tension()
        
        return {
            'alpha_inverse_sensitivity': alpha_sensitivity,
            'H0_sensitivity': h0_sensitivity,
            'hubble_tension_analysis': hubble_tension,
            'overall_assessment': self._assess_overall_sensitivity(
                alpha_sensitivity,
                h0_sensitivity,
                hubble_tension
            )
        }
    
    def _assess_overall_sensitivity(self, alpha_sensitivity: Dict, h0_sensitivity: Dict, 
                                    hubble_tension: Dict) -> str:
        """Assess overall sensitivity from all tests."""
        alpha_rel = alpha_sensitivity['sensitivity_metrics']['relative_sensitivity']
        h0_rel = h0_sensitivity['sensitivity_metrics']['relative_sensitivity']
        
        assessment = f"H₀ uncertainty DOMINATES (rel. sensitivity {h0_rel*100:.1f}% >> α⁻¹ {alpha_rel*100:.4f}%). "
        
        # Check Hubble tension effect
        ht = hubble_tension['hubble_tension_effect']
        assessment += f"Hubble tension (8% H₀ variation) causes {ht['G_difference_percent']:.1f}% G variation. "
        
        # Determine which H₀ is favored
        planck_diff = hubble_tension['planck_cmb']['relative_difference_percent']
        sh0es_diff = hubble_tension['sh0es_cepheids']['relative_difference_percent']
        
        if planck_diff < sh0es_diff:
            assessment += f"Planck H₀ gives better agreement ({planck_diff:.2f}% vs {sh0es_diff:.2f}%)."
        else:
            assessment += f"SH0ES H₀ gives better agreement ({sh0es_diff:.2f}% vs {planck_diff:.2f}%)."
        
        return assessment
