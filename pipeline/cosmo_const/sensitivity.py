"""
Sensitivity Analysis
====================

Analyze sensitivity of Ω_Λ prediction to:
1. Dimensional weight assignments
2. Decoherence timescale choice
3. Causal diamond geometry assumptions
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class SensitivityAnalysis:
    """
    Sensitivity analysis for cosmological constant prediction.
    
    Tests robustness of prediction to variations in:
    - Dimension weights (3, 3, 2)
    - Decoherence timescale
    - Causal structure assumptions
    """
    
    def __init__(self):
        """Initialize sensitivity analyzer."""
        pass
    
    def sensitivity_to_dimension_weights(self,
                                        w_null_range: Tuple[float, float] = (2.5, 3.5),
                                        w_screen_range: Tuple[float, float] = (1.5, 2.5),
                                        n_points: int = 5) -> Dict[str, Any]:
        """
        Analyze sensitivity to dimensional weight assignments.
        
        Base case: w_N^+ = w_N^- = 3, w_σ = 2
        
        Parameters:
            w_null_range (tuple): Range for null cone weights
            w_screen_range (tuple): Range for screen weight
            n_points (int): Number of points per dimension
        
        Returns:
            dict: Sensitivity analysis results
        """
        w_null_values = np.linspace(w_null_range[0], w_null_range[1], n_points)
        w_screen_values = np.linspace(w_screen_range[0], w_screen_range[1], n_points)
        
        results = []
        base_omega = None
        
        for w_null in w_null_values:
            for w_screen in w_screen_values:
                omega = self._compute_omega_lambda_custom_weights(w_null, w_null, w_screen)
                
                if w_null == 3.0 and w_screen == 2.0:
                    base_omega = omega
                
                results.append({
                    'w_null': float(w_null),
                    'w_screen': float(w_screen),
                    'omega_lambda': float(omega),
                    'deviation_from_base': float(omega - base_omega) if base_omega is not None else 0.0
                })
        
        # Calculate sensitivity metrics
        omegas = [r['omega_lambda'] for r in results]
        deviations = [r['deviation_from_base'] for r in results if base_omega is not None]
        
        return {
            'base_omega_lambda': float(base_omega) if base_omega is not None else None,
            'parameter_variations': results,
            'sensitivity_metrics': {
                'min_omega': float(np.min(omegas)),
                'max_omega': float(np.max(omegas)),
                'range': float(np.max(omegas) - np.min(omegas)),
                'std_omega': float(np.std(omegas)),
                'max_deviation': float(np.max(np.abs(deviations))) if deviations else 0.0
            },
            'interpretation': self._interpret_weight_sensitivity(deviations) if deviations else 'N/A'
        }
    
    def sensitivity_to_decoherence_timescale(self,
                                             tau_factors: List[float] = None) -> Dict[str, Any]:
        """
        Analyze sensitivity to decoherence timescale choice.
        
        Base case: f_irrev = 1 - exp(-1) (Hubble timescale)
        Variations: f_irrev = 1 - exp(-tau_factor)
        
        Parameters:
            tau_factors (list): List of timescale factors to test
        
        Returns:
            dict: Sensitivity analysis results
        """
        if tau_factors is None:
            tau_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        # Base geometric entropy (unchanged)
        S_geom = (11 * np.log(2) - 3 * np.log(3)) / 4
        
        results = []
        base_omega = None
        
        for tau_factor in tau_factors:
            f_irrev = 1 - np.exp(-tau_factor)
            omega = S_geom * f_irrev
            
            if tau_factor == 1.0:
                base_omega = omega
            
            results.append({
                'tau_factor': float(tau_factor),
                'f_irrev': float(f_irrev),
                'omega_lambda': float(omega),
                'deviation_from_base': float(omega - base_omega) if base_omega is not None else 0.0
            })
        
        # Calculate sensitivity
        omegas = [r['omega_lambda'] for r in results]
        deviations = [r['deviation_from_base'] for r in results if base_omega is not None]
        
        return {
            'base_omega_lambda': float(base_omega) if base_omega is not None else None,
            'base_tau_factor': 1.0,
            'variations': results,
            'sensitivity_metrics': {
                'min_omega': float(np.min(omegas)),
                'max_omega': float(np.max(omegas)),
                'range': float(np.max(omegas) - np.min(omegas)),
                'max_deviation': float(np.max(np.abs(deviations))) if deviations else 0.0
            },
            'interpretation': self._interpret_timescale_sensitivity(deviations) if deviations else 'N/A'
        }
    
    def sensitivity_to_causal_structure(self) -> Dict[str, Any]:
        """
        Analyze sensitivity to causal diamond structure assumptions.
        
        Tests variations in:
        - Number of causal sectors
        - Dimensionality assignments
        - Entropy calculation method
        
        Returns:
            dict: Sensitivity analysis results
        """
        # Base case: tripartite structure (N^+, N^-, σ)
        base_omega = (11 * np.log(2) - 3 * np.log(3)) / 4 * (1 - np.exp(-1))
        
        # Alternative structures
        alternatives = []
        
        # Bipartite structure (future/past only)
        p_future = 0.5
        p_past = 0.5
        S_bipartite = -p_future * np.log(p_future) - p_past * np.log(p_past)
        omega_bipartite = S_bipartite * (1 - np.exp(-1))
        alternatives.append({
            'structure': 'bipartite',
            'description': 'Future/past null cones only',
            'omega_lambda': float(omega_bipartite),
            'deviation': float(omega_bipartite - base_omega)
        })
        
        # Quadripartite structure (add extra dimension)
        w1, w2, w3, w4 = 2.0, 2.0, 2.0, 2.0
        w_tot = w1 + w2 + w3 + w4
        p1 = p2 = p3 = p4 = w1 / w_tot
        S_quad = -4 * p1 * np.log(p1)
        omega_quad = S_quad * (1 - np.exp(-1))
        alternatives.append({
            'structure': 'quadripartite',
            'description': 'Four equal-weight sectors',
            'omega_lambda': float(omega_quad),
            'deviation': float(omega_quad - base_omega)
        })
        
        return {
            'base_structure': 'tripartite (N^+, N^-, σ)',
            'base_omega_lambda': float(base_omega),
            'alternatives': alternatives,
            'interpretation': self._interpret_structure_sensitivity(alternatives)
        }
    
    def _compute_omega_lambda_custom_weights(self,
                                            w_N_plus: float,
                                            w_N_minus: float,
                                            w_sigma: float) -> float:
        """
        Compute Ω_Λ with custom dimension weights.
        
        Parameters:
            w_N_plus (float): Weight for future null cone
            w_N_minus (float): Weight for past null cone
            w_sigma (float): Weight for holographic screen
        
        Returns:
            float: Computed Ω_Λ
        """
        w_tot = w_N_plus + w_N_minus + w_sigma
        
        p_N_plus = w_N_plus / w_tot
        p_N_minus = w_N_minus / w_tot
        p_sigma = w_sigma / w_tot
        
        S_geom = -p_N_plus * np.log(p_N_plus) - p_N_minus * np.log(p_N_minus) - p_sigma * np.log(p_sigma)
        f_irrev = 1 - np.exp(-1)
        
        return S_geom * f_irrev
    
    def _interpret_weight_sensitivity(self, deviations: List[float]) -> str:
        """Interpret sensitivity to weight variations."""
        max_dev = np.max(np.abs(deviations))
        if max_dev < 0.001:
            return "Highly robust: <0.1% variation"
        elif max_dev < 0.01:
            return "Robust: <1% variation"
        elif max_dev < 0.05:
            return "Moderately sensitive: <5% variation"
        else:
            return f"Sensitive: {max_dev*100:.1f}% variation"
    
    def _interpret_timescale_sensitivity(self, deviations: List[float]) -> str:
        """Interpret sensitivity to timescale variations."""
        max_dev = np.max(np.abs(deviations))
        if max_dev < 0.01:
            return "Robust to timescale choice"
        elif max_dev < 0.05:
            return "Moderately sensitive to timescale"
        else:
            return f"Sensitive to timescale: {max_dev*100:.1f}% variation"
    
    def _interpret_structure_sensitivity(self, alternatives: List[Dict]) -> str:
        """Interpret sensitivity to causal structure."""
        max_dev = np.max([abs(a['deviation']) for a in alternatives])
        if max_dev < 0.01:
            return "Robust to structure variations"
        else:
            return f"Sensitive to structure: {max_dev*100:.1f}% variation"
    
    def full_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Perform complete sensitivity analysis.
        
        Returns:
            dict: Complete sensitivity analysis results
        """
        weight_sensitivity = self.sensitivity_to_dimension_weights()
        timescale_sensitivity = self.sensitivity_to_decoherence_timescale()
        structure_sensitivity = self.sensitivity_to_causal_structure()
        
        return {
            'dimension_weights': weight_sensitivity,
            'decoherence_timescale': timescale_sensitivity,
            'causal_structure': structure_sensitivity,
            'overall_robustness': self._assess_overall_robustness(
                weight_sensitivity,
                timescale_sensitivity,
                structure_sensitivity
            )
        }
    
    def _assess_overall_robustness(self, *sensitivity_results) -> str:
        """Assess overall robustness from all sensitivity tests."""
        # Check if all variations are small
        all_robust = True
        for result in sensitivity_results:
            if 'sensitivity_metrics' in result:
                max_dev = result['sensitivity_metrics'].get('max_deviation', 1.0)
                if max_dev > 0.01:  # >1% variation
                    all_robust = False
        
        if all_robust:
            return "Prediction is robust to parameter variations"
        else:
            return "Prediction shows some sensitivity to parameter variations"
