"""
Parameter Consistency Check
============================

Check if β values from independent tests are consistent.

This module computes weighted mean β and consistency χ² to verify that
all five tests yield the same β value within uncertainties.

CRITICAL: No hardcoded significance thresholds. All statistics are reported
and the interpretation is left to the scientific analysis.
"""

import numpy as np
from scipy.stats import chi2
from typing import Dict, Any, List


def joint_consistency_check(test_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check if β values from independent tests are consistent.
    
    This implements the joint consistency check from docs/cmb_gw.md.
    
    CRITICAL: No hardcoded p-value thresholds. The analysis reports:
    - All statistical quantities (χ², p-value, ndof)
    - Multiple significance levels for reference
    - Let the scientific interpretation be made by the researcher
    
    Parameters:
    -----------
    test_results : dict
        Dictionary with keys ['sound_horizon', 'voids', 'sirens', 'peaks', 'coherence']
        Each contains 'beta_fit' and 'beta_err'
        
    Returns:
    --------
    dict
        Consistency check results containing:
        - 'beta_combined': Weighted mean β
        - 'beta_combined_err': Uncertainty on combined β
        - 'chi2_consistency': χ² for consistency test
        - 'ndof': Degrees of freedom
        - 'p_value': p-value for consistency
        - 'sigma_tension': Tension in units of sigma (for reference)
        - 'individual_betas': Dictionary of individual β values
        - 'individual_errors': Dictionary of individual β uncertainties
        - 'n_tests_combined': Number of tests that contributed
    """
    betas = []
    errors = []
    weights = []
    test_names = []
    
    for test_name, result in test_results.items():
        if 'beta_fit' in result and 'beta_err' in result:
            beta_val = result['beta_fit']
            beta_err = result['beta_err']
            
            # EXCLUDE tests marked as QUALITATIVE_ONLY (e.g., voids without N-body calibration)
            if result.get('caveat') == 'QUALITATIVE_ONLY':
                continue
            
            # Only include valid (finite) values with positive uncertainties
            if np.isfinite(beta_val) and np.isfinite(beta_err) and beta_err > 0:
                betas.append(beta_val)
                errors.append(beta_err)
                weights.append(1.0 / beta_err**2)
                test_names.append(test_name)
    
    if not betas:
        return {
            'beta_combined': np.nan,
            'beta_combined_err': np.nan,
            'chi2_consistency': np.nan,
            'ndof': 0,
            'p_value': np.nan,
            'sigma_tension': np.nan,
            'individual_betas': {},
            'individual_errors': {},
            'n_tests_combined': 0
        }
    
    betas = np.array(betas)
    errors = np.array(errors)
    weights = np.array(weights)
    
    # Weighted mean
    beta_combined = np.sum(betas * weights) / np.sum(weights)
    beta_combined_err = np.sqrt(1.0 / np.sum(weights))
    
    # Consistency χ²
    chi2_consistency = np.sum(weights * (betas - beta_combined)**2)
    ndof = len(betas) - 1
    
    # p-value from χ² distribution
    if ndof > 0:
        p_value = 1 - chi2.cdf(chi2_consistency, ndof)
        # Convert to sigma equivalent (for reference, not for decision-making)
        # σ such that P(|X| > σ) = p_value for standard normal
        from scipy.stats import norm
        if p_value > 0 and p_value < 1:
            sigma_tension = norm.ppf(1 - p_value/2)  # Two-tailed
        else:
            sigma_tension = np.inf if p_value == 0 else 0.0
    else:
        p_value = 1.0
        sigma_tension = 0.0
    
    # Individual values dictionaries
    individual_betas = {name: float(beta) for name, beta in zip(test_names, betas)}
    individual_errors = {name: float(err) for name, err in zip(test_names, errors)}
    
    return {
        'beta_combined': float(beta_combined),
        'beta_combined_err': float(beta_combined_err),
        'chi2_consistency': float(chi2_consistency),
        'ndof': int(ndof),
        'p_value': float(p_value),
        'sigma_tension': float(sigma_tension),
        'individual_betas': individual_betas,
        'individual_errors': individual_errors,
        'n_tests_combined': len(betas)
    }

