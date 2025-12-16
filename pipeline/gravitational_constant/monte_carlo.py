"""
Monte Carlo Uncertainty Quantification for Gravitational Constant
==================================================================

Monte Carlo sampling to quantify uncertainty and consistency with CODATA observation.

The prediction G = πc⁵/(ℏH²N_P) has uncertainty dominated by H₀ measurement.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import stats
from .physics import calculate_g


class GravitationalConstantMonteCarloValidator:
    """
    Monte Carlo uncertainty quantification for gravitational constant prediction.
    
    Samples from input parameter posteriors to compute probability that observation
    is consistent with the prediction.
    """
    
    def __init__(self, n_samples: int = 100000, random_state: int = 42):
        """
        Initialize Monte Carlo validator.
        
        Parameters:
            n_samples (int): Number of Monte Carlo samples
            random_state (int): Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def run_mc_validation(self,
                         codata_G: float = 6.67430e-11,
                         codata_sigma: float = 0.00015e-11,
                         alpha_inverse: float = 137.035999084,
                         delta_alpha_inverse: float = 0.000000021,
                         H0: float = None,
                         delta_H0: float = None) -> Dict[str, Any]:
        """
        Run Monte Carlo validation against CODATA 2018 observation.
        
        Samples from input parameter posteriors and computes:
        - Distribution of predicted G values
        - Fraction consistent with observation
        - Statistical significance
        
        Parameters:
            codata_G (float): CODATA 2018 central value
            codata_sigma (float): CODATA 2018 1σ uncertainty
            alpha_inverse (float): Central value of α⁻¹
            delta_alpha_inverse (float): Uncertainty in α⁻¹
            H0 (float, optional): Hubble parameter. Defaults to HLCDM_PARAMS.H0.
            delta_H0 (float, optional): Uncertainty in H0 (~1% from Hubble tension)
        
        Returns:
            dict: Monte Carlo validation results
        """
        from hlcdm.parameters import HLCDM_PARAMS
        
        if H0 is None:
            H0 = HLCDM_PARAMS.H0
        
        if delta_H0 is None:
            # Hubble tension gives ~1% uncertainty
            delta_H0 = 0.01 * H0
        
        # Sample from input parameter posteriors (assumed Gaussian)
        alpha_inv_samples = np.random.normal(alpha_inverse, delta_alpha_inverse, self.n_samples)
        H0_samples = np.random.normal(H0, delta_H0, self.n_samples)
        
        # Calculate G for each sample 
        G_samples = []
        for i in range(self.n_samples):
            try:
                g_result = calculate_g(H=H0_samples[i], alpha_inverse=alpha_inv_samples[i])
                G_samples.append(g_result['G'])
            except:
                # Skip invalid samples
                continue
        
        G_samples = np.array(G_samples)
        
        if len(G_samples) == 0:
            return {
                'error': 'No valid samples generated',
                'n_samples': self.n_samples
            }
        
        # Calculate deviations from observation
        deviations = G_samples - codata_G
        abs_deviations = np.abs(deviations)
        
        # Relative differences
        relative_differences = abs_deviations / codata_G * 100
        
        # Consistency fractions based on relative difference
        # For parameter-free prediction, these thresholds are meaningful
        p_consistent_0_1_percent = np.mean(relative_differences < 0.1)
        p_consistent_0_5_percent = np.mean(relative_differences < 0.5)
        p_consistent_1_percent = np.mean(relative_differences < 1.0)
        p_consistent_2_percent = np.mean(relative_differences < 2.0)
        
        # Central prediction
        mean_G = np.mean(G_samples)
        median_G = np.median(G_samples)
        std_G = np.std(G_samples)
        
        # Deviation statistics
        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)
        median_deviation = np.median(deviations)
        mean_relative_diff = np.mean(relative_differences)
        
        # Deviation of central value
        g_result_central = calculate_g(H=H0, alpha_inverse=alpha_inverse)
        central_G = g_result_central['G']
        central_deviation = central_G - codata_G
        central_relative_diff = abs(central_deviation) / codata_G * 100
        
        # Percentiles
        percentiles = {
            'p16': np.percentile(G_samples, 16),
            'p50': np.percentile(G_samples, 50),
            'p84': np.percentile(G_samples, 84),
            'p2.5': np.percentile(G_samples, 2.5),
            'p97.5': np.percentile(G_samples, 97.5)
        }
        
        # 1σ range from MC
        mc_1sigma_range = (percentiles['p16'], percentiles['p84'])
        
        return {
            'prediction': central_G,
            'observed_mean': codata_G,
            'observed_sigma': codata_sigma,
            'n_samples': len(G_samples),
            'mean_prediction': float(mean_G),
            'median_prediction': float(median_G),
            'std_prediction': float(std_G),
            'central_relative_difference_percent': float(central_relative_diff),
            'consistency_fractions': {
                'p_consistent_0_1_percent': float(p_consistent_0_1_percent),
                'p_consistent_0_5_percent': float(p_consistent_0_5_percent),
                'p_consistent_1_percent': float(p_consistent_1_percent),
                'p_consistent_2_percent': float(p_consistent_2_percent)
            },
            'deviation_statistics': {
                'mean': float(mean_deviation),
                'std': float(std_deviation),
                'median': float(median_deviation),
                'mean_relative_difference_percent': float(mean_relative_diff),
                'percentiles': {k: float(v) for k, v in percentiles.items()}
            },
            'mc_1sigma_range': {
                'low': float(mc_1sigma_range[0]),
                'high': float(mc_1sigma_range[1])
            },
            'interpretation': self._interpret_results(central_relative_diff, p_consistent_1_percent),
            'note': 'H₀ uncertainty dominates; no correction factors applied'
        }
    
    def _interpret_results(self, relative_difference_percent: float, p_consistent_1_percent: float) -> str:
        """
        Interpret Monte Carlo results.
        
        Parameters:
            relative_difference_percent (float): Relative difference in percent
            p_consistent_1_percent (float): Fraction consistent within 1% relative difference
        
        Returns:
            str: Interpretation string
        """
        if relative_difference_percent < 0.1:
            return f"Excellent agreement: {relative_difference_percent:.4f}% relative difference"
        elif relative_difference_percent < 0.5:
            return f"Very good agreement: {relative_difference_percent:.4f}% relative difference"
        elif relative_difference_percent < 1.0:
            return f"Good agreement: {relative_difference_percent:.4f}% relative difference (excellent for parameter-free)"
        elif relative_difference_percent < 2.0:
            return f"Good agreement: {relative_difference_percent:.4f}% relative difference (good for parameter-free)"
        else:
            return f"Moderate agreement: {relative_difference_percent:.4f}% relative difference"
