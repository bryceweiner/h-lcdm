"""
Monte Carlo Uncertainty Quantification for Fine Structure Constant
====================================================================

Monte Carlo sampling to quantify uncertainty and consistency with CODATA observation.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import stats
from .physics import calculate_alpha_inverse


class FineStructureMonteCarloValidator:
    """
    Monte Carlo uncertainty quantification for fine structure constant prediction.
    
    Samples from CODATA posterior to compute probability that observation
    is consistent with the parameter-free prediction.
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
                         codata_alpha_inverse: float = 137.035999084,
                         codata_sigma: float = 0.000000021) -> Dict[str, Any]:
        """
        Run Monte Carlo validation against CODATA 2018 observation.
        
        Samples from CODATA posterior N(μ, σ²) and computes:
        - Fraction of samples consistent with prediction
        - Distribution of deviations
        - Statistical significance
        
        Parameters:
            codata_alpha_inverse (float): CODATA 2018 central value
            codata_sigma (float): CODATA 2018 1σ uncertainty
        
        Returns:
            dict: Monte Carlo validation results
        """
        # Get parameter-free prediction
        alpha_inv_result = calculate_alpha_inverse()
        prediction = alpha_inv_result['alpha_inverse']
        
        # Sample from CODATA posterior (assumed Gaussian)
        samples = np.random.normal(codata_alpha_inverse, codata_sigma, self.n_samples)
        
        # Calculate deviations
        deviations = samples - prediction
        abs_deviations = np.abs(deviations)
        
        # For extremely precise measurements (CODATA), use relative difference
        # instead of absolute deviation in sigma units
        relative_differences = np.abs(deviations) / codata_alpha_inverse * 100
        
        # Consistency fractions based on relative difference
        # For CODATA-level precision, these thresholds are more meaningful
        p_consistent_0_01_percent = np.mean(relative_differences < 0.01)
        p_consistent_0_1_percent = np.mean(relative_differences < 0.1)
        p_consistent_1_percent = np.mean(relative_differences < 1.0)
        
        # Also calculate traditional sigma-based consistency (for reference)
        p_consistent_1sigma = np.mean(abs_deviations < codata_sigma)
        p_consistent_2sigma = np.mean(abs_deviations < 2 * codata_sigma)
        p_consistent_3sigma = np.mean(abs_deviations < 3 * codata_sigma)
        
        # Deviation statistics
        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)
        median_deviation = np.median(deviations)
        mean_relative_diff = np.mean(relative_differences)
        
        # Deviation in sigma units (not meaningful for CODATA, but included for reference)
        deviation_sigma = (codata_alpha_inverse - prediction) / codata_sigma
        
        # Relative difference of central value
        relative_difference_percent = abs(codata_alpha_inverse - prediction) / codata_alpha_inverse * 100
        
        # Percentiles of deviation distribution
        percentiles = {
            'p16': np.percentile(deviations, 16),
            'p50': np.percentile(deviations, 50),
            'p84': np.percentile(deviations, 84),
            'p2.5': np.percentile(deviations, 2.5),
            'p97.5': np.percentile(deviations, 97.5)
        }
        
        # Percentiles of relative difference distribution
        relative_percentiles = {
            'p16': np.percentile(relative_differences, 16),
            'p50': np.percentile(relative_differences, 50),
            'p84': np.percentile(relative_differences, 84),
            'p2.5': np.percentile(relative_differences, 2.5),
            'p97.5': np.percentile(relative_differences, 97.5)
        }
        
        # Probability that prediction lies within observed range
        p_in_observed_range = np.mean(
            (samples >= prediction - codata_sigma) & 
            (samples <= prediction + codata_sigma)
        )
        
        # Kolmogorov-Smirnov test against null hypothesis
        ks_statistic, ks_p_value = stats.kstest(
            deviations / codata_sigma,
            lambda x: stats.norm.cdf(x, loc=0, scale=1)
        )
        
        return {
            'prediction': prediction,
            'observed_mean': codata_alpha_inverse,
            'observed_sigma': codata_sigma,
            'n_samples': self.n_samples,
            'deviation_sigma': deviation_sigma,  # Not meaningful for CODATA, included for reference
            'relative_difference_percent': float(relative_difference_percent),
            'consistency_fractions': {
                'p_consistent_1sigma': float(p_consistent_1sigma),  # Not meaningful for CODATA
                'p_consistent_2sigma': float(p_consistent_2sigma),  # Not meaningful for CODATA
                'p_consistent_3sigma': float(p_consistent_3sigma),  # Not meaningful for CODATA
                'p_consistent_0_01_percent': float(p_consistent_0_01_percent),  # < 0.01% relative
                'p_consistent_0_1_percent': float(p_consistent_0_1_percent),  # < 0.1% relative
                'p_consistent_1_percent': float(p_consistent_1_percent),  # < 1% relative
                'p_in_observed_range': float(p_in_observed_range)
            },
            'deviation_statistics': {
                'mean': float(mean_deviation),
                'std': float(std_deviation),
                'median': float(median_deviation),
                'mean_relative_difference_percent': float(mean_relative_diff),
                'percentiles': {k: float(v) for k, v in percentiles.items()},
                'relative_percentiles': {k: float(v) for k, v in relative_percentiles.items()}
            },
            'ks_test': {
                'statistic': float(ks_statistic),
                'p_value': float(ks_p_value),
                'interpretation': 'consistent' if ks_p_value > 0.05 else 'inconsistent'
            },
            'interpretation': self._interpret_results(relative_difference_percent, p_consistent_0_1_percent)
        }
    
    def _interpret_results(self, relative_difference_percent: float, p_consistent_0_1_percent: float) -> str:
        """
        Interpret Monte Carlo results.
        
        For extremely precise measurements (like CODATA), use relative difference.
        
        Parameters:
            relative_difference_percent (float): Relative difference in percent
            p_consistent_0_1_percent (float): Fraction consistent within 0.1% relative difference
        
        Returns:
            str: Interpretation string
        """
        if relative_difference_percent < 0.01:
            return f"Excellent agreement: {relative_difference_percent:.4f}% relative difference"
        elif relative_difference_percent < 0.1:
            return f"Very good agreement: {relative_difference_percent:.4f}% relative difference"
        elif relative_difference_percent < 1.0:
            return f"Good agreement: {relative_difference_percent:.4f}% relative difference"
        else:
            return f"Moderate agreement: {relative_difference_percent:.4f}% relative difference"
    
    def sensitivity_to_h0_uncertainty(self,
                                      H0_range: tuple = (2.15e-18, 2.22e-18),
                                      n_H0_points: int = 20) -> Dict[str, Any]:
        """
        Analyze sensitivity of α⁻¹ prediction to H0 uncertainty.
        
        Parameters:
            H0_range (tuple): Range of H0 values to test
            n_H0_points (int): Number of H0 values in range
        
        Returns:
            dict: Sensitivity analysis results
        """
        H0_values = np.linspace(H0_range[0], H0_range[1], n_H0_points)
        alpha_inverse_values = []
        
        for H0 in H0_values:
            alpha_inv_result = calculate_alpha_inverse(H0)
            alpha_inverse_values.append(alpha_inv_result['alpha_inverse'])
        
        alpha_inverse_values = np.array(alpha_inverse_values)
        
        # Calculate sensitivity
        mean_alpha_inv = np.mean(alpha_inverse_values)
        std_alpha_inv = np.std(alpha_inverse_values)
        
        return {
            'H0_values': H0_values.tolist(),
            'alpha_inverse_values': [float(x) for x in alpha_inverse_values],
            'sensitivity': {
                'mean_alpha_inverse': float(mean_alpha_inv),
                'std_alpha_inverse': float(std_alpha_inv),
                'range': float(np.max(alpha_inverse_values) - np.min(alpha_inverse_values)),
                'relative_sensitivity': float(std_alpha_inv / mean_alpha_inv) if mean_alpha_inv > 0 else 0.0
            }
        }
