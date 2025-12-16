"""
Monte Carlo Uncertainty Quantification
========================================

Monte Carlo sampling to quantify uncertainty and consistency with observations.

Reuses patterns from BootstrapValidator for statistical validation.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import stats
from .physics import calculate_omega_lambda


class CosmoConstMonteCarloValidator:
    """
    Monte Carlo uncertainty quantification for cosmological constant prediction.
    
    Samples from Planck posterior to compute probability that observation
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
                         planck_omega_lambda: float = 0.6847,
                         planck_sigma: float = 0.0073) -> Dict[str, Any]:
        """
        Run Monte Carlo validation against Planck 2018 observation.
        
        Samples from Planck posterior N(μ, σ²) and computes:
        - Fraction of samples consistent with prediction
        - Distribution of deviations
        - Statistical significance
        
        Parameters:
            planck_omega_lambda (float): Planck 2018 central value
            planck_sigma (float): Planck 2018 1σ uncertainty
        
        Returns:
            dict: Monte Carlo validation results
        """
        # Get parameter-free prediction
        omega_result = calculate_omega_lambda()
        prediction = omega_result['omega_lambda']
        
        # Sample from Planck posterior (assumed Gaussian)
        samples = np.random.normal(planck_omega_lambda, planck_sigma, self.n_samples)
        
        # Calculate deviations
        deviations = samples - prediction
        abs_deviations = np.abs(deviations)
        
        # Consistency fractions
        p_consistent_1sigma = np.mean(abs_deviations < planck_sigma)
        p_consistent_2sigma = np.mean(abs_deviations < 2 * planck_sigma)
        p_consistent_3sigma = np.mean(abs_deviations < 3 * planck_sigma)
        
        # Deviation statistics
        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)
        median_deviation = np.median(deviations)
        
        # Deviation in sigma units
        deviation_sigma = (planck_omega_lambda - prediction) / planck_sigma
        
        # Percentiles of deviation distribution
        percentiles = {
            'p16': np.percentile(deviations, 16),
            'p50': np.percentile(deviations, 50),
            'p84': np.percentile(deviations, 84),
            'p2.5': np.percentile(deviations, 2.5),
            'p97.5': np.percentile(deviations, 97.5)
        }
        
        # Probability that prediction lies within observed range
        p_in_observed_range = np.mean(
            (samples >= prediction - planck_sigma) & 
            (samples <= prediction + planck_sigma)
        )
        
        # Kolmogorov-Smirnov test against null hypothesis (prediction = observation)
        ks_statistic, ks_p_value = stats.kstest(
            deviations / planck_sigma,
            lambda x: stats.norm.cdf(x, loc=0, scale=1)
        )
        
        return {
            'prediction': prediction,
            'observed_mean': planck_omega_lambda,
            'observed_sigma': planck_sigma,
            'n_samples': self.n_samples,
            'deviation_sigma': deviation_sigma,
            'consistency_fractions': {
                'p_consistent_1sigma': float(p_consistent_1sigma),
                'p_consistent_2sigma': float(p_consistent_2sigma),
                'p_consistent_3sigma': float(p_consistent_3sigma),
                'p_in_observed_range': float(p_in_observed_range)
            },
            'deviation_statistics': {
                'mean': float(mean_deviation),
                'std': float(std_deviation),
                'median': float(median_deviation),
                'percentiles': {k: float(v) for k, v in percentiles.items()}
            },
            'ks_test': {
                'statistic': float(ks_statistic),
                'p_value': float(ks_p_value),
                'interpretation': 'consistent' if ks_p_value > 0.05 else 'inconsistent'
            },
            'interpretation': self._interpret_results(deviation_sigma, p_consistent_2sigma)
        }
    
    def _interpret_results(self, deviation_sigma: float, p_consistent_2sigma: float) -> str:
        """
        Interpret Monte Carlo results.
        
        Parameters:
            deviation_sigma (float): Deviation in sigma units
            p_consistent_2sigma (float): Fraction consistent within 2σ
        
        Returns:
            str: Interpretation string
        """
        if np.abs(deviation_sigma) < 0.1:
            return "Excellent agreement: prediction within 0.1σ of observation"
        elif np.abs(deviation_sigma) < 0.5:
            return "Very good agreement: prediction within 0.5σ of observation"
        elif np.abs(deviation_sigma) < 1.0:
            return "Good agreement: prediction within 1σ of observation"
        elif np.abs(deviation_sigma) < 2.0:
            return "Acceptable agreement: prediction within 2σ of observation"
        else:
            return "Tension: prediction deviates by >2σ from observation"
    
    def sensitivity_to_planck_uncertainty(self,
                                         sigma_range: tuple = (0.005, 0.010),
                                         n_sigma_points: int = 20) -> Dict[str, Any]:
        """
        Analyze sensitivity of consistency to assumed Planck uncertainty.
        
        Parameters:
            sigma_range (tuple): Range of sigma values to test
            n_sigma_points (int): Number of sigma values in range
        
        Returns:
            dict: Sensitivity analysis results
        """
        planck_omega = 0.6847
        omega_result = calculate_omega_lambda()
        prediction = omega_result['omega_lambda']
        
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], n_sigma_points)
        consistency_fractions = []
        deviation_sigmas = []
        
        for sigma in sigma_values:
            # Sample from posterior
            samples = np.random.normal(planck_omega, sigma, self.n_samples)
            deviations = np.abs(samples - prediction)
            
            # Consistency fraction
            p_consistent = np.mean(deviations < 2 * sigma)
            consistency_fractions.append(p_consistent)
            
            # Deviation in sigma
            dev_sigma = (planck_omega - prediction) / sigma
            deviation_sigmas.append(dev_sigma)
        
        return {
            'sigma_values': sigma_values.tolist(),
            'consistency_fractions': [float(x) for x in consistency_fractions],
            'deviation_sigmas': [float(x) for x in deviation_sigmas],
            'sensitivity': {
                'min_consistency': float(np.min(consistency_fractions)),
                'max_consistency': float(np.max(consistency_fractions)),
                'mean_consistency': float(np.mean(consistency_fractions))
            }
        }
